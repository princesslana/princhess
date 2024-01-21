use princhess::chess::{Board, Move};
use princhess::math::Rng;
use princhess::options::set_hash_size_mb;
use princhess::search::Search;
use princhess::state::State;
use princhess::tablebase::{self, Wdl};
use princhess::train::TrainingPosition;
use princhess::transposition_table::LRTable;

use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const THREADS: usize = 6;
const DATA_WRITE_RATE: usize = 16384;
const PLAYOUTS_PER_MOVE: usize = 2000;
const DFRC_PCT: u64 = 10;

struct Stats {
    start: Instant,
    games: AtomicU64,
    positions: AtomicU64,
    skipped: AtomicU64,
    white_wins: AtomicU64,
    black_wins: AtomicU64,
    draws: AtomicU64,
    blunders: AtomicU64,
}

impl Stats {
    pub fn zero() -> Self {
        Self {
            start: Instant::now(),
            games: AtomicU64::new(0),
            positions: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            white_wins: AtomicU64::new(0),
            black_wins: AtomicU64::new(0),
            draws: AtomicU64::new(0),
            blunders: AtomicU64::new(0),
        }
    }

    pub fn inc_games(&self) {
        self.games
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_positions(&self) {
        self.positions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_skipped(&self) {
        self.skipped
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_white_wins(&self) {
        self.white_wins
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_black_wins(&self) {
        self.black_wins
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_draws(&self) {
        self.draws
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn inc_blunders(&self) {
        self.blunders
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let games = self.games.load(std::sync::atomic::Ordering::Relaxed);
        let white_wins = self.white_wins.load(std::sync::atomic::Ordering::Relaxed);
        let draws = self.draws.load(std::sync::atomic::Ordering::Relaxed);
        let black_wins = self.black_wins.load(std::sync::atomic::Ordering::Relaxed);
        let positions = self.positions.load(std::sync::atomic::Ordering::Relaxed);
        let skipped = self.skipped.load(std::sync::atomic::Ordering::Relaxed);
        let blunders = self.blunders.load(std::sync::atomic::Ordering::Relaxed);
        let seconds = self.start.elapsed().as_secs().max(1);

        write!(
            f,
            "G: {:>7} | +{:>2} ={:>2} -{:>2} % | Bls: {:>2}% | S: {:>5.3}% | Pos: {:>5.1}m ({:>3}/g) ({:>4}/s)",
            games,
            white_wins * 100 / games,
            draws * 100 / games,
            black_wins * 100 / games,
            blunders * 100 / games,
            skipped as f32 / positions as f32,
            positions as f32 / 1000000.0,
            positions / games,
            positions / seconds
        )
    }
}

fn run_game(stats: &Stats, positions: &mut Vec<TrainingPosition>, rng: &mut Rng) {
    let startpos = if rng.next_u64() % 100 < DFRC_PCT {
        Board::dfrc(rng.next_usize() % 960, rng.next_usize() % 960)
    } else {
        Board::startpos()
    };

    let mut state = State::from_board(startpos);
    let mut table = LRTable::empty();

    let mut game_positions = Vec::with_capacity(256);

    let mut prev_moves = [Move::NONE; 4];

    for _ in 0..(8 + rng.next_u64() % 2) {
        let moves = state.available_moves();

        if moves.is_empty() {
            return;
        }

        let index = rng.next_usize() % moves.len();
        let best_move = moves[index];

        state.make_move(best_move);
    }

    if !state.is_available_move() {
        return;
    }

    let result = loop {
        let search = Search::new(state.clone(), table);

        search.playout_sync(PLAYOUTS_PER_MOVE);

        let best_move = search.best_move();

        let legal_moves = search.root_node().hots().len();

        if legal_moves > TrainingPosition::MAX_MOVES {
            stats.inc_skipped();
        } else {
            let mut position = TrainingPosition::from(search.tree());

            if position.evaluation() > 0.95 {
                break 1;
            } else if position.evaluation() < -0.95 {
                break -1;
            } else if let Some(wdl) = tablebase::probe_wdl(state.board()) {
                let result = match wdl {
                    Wdl::Win => 1,
                    Wdl::Draw => 0,
                    Wdl::Loss => -1,
                };

                break state.side_to_move().fold(result, -result);
            }

            position.set_previous_moves(prev_moves);
            game_positions.push(position);
            stats.inc_positions();
        }

        state.make_move(best_move);

        prev_moves.rotate_right(1);
        prev_moves[0] = best_move;

        if !state.is_available_move() {
            break if state.is_check() {
                // The stm has been checkmated. Convert to white relative result
                state.side_to_move().fold(-1, 1)
            } else {
                0
            };
        }

        if state.drawn_by_fifty_move_rule()
            || state.is_repetition()
            || state.board().is_insufficient_material()
        {
            break 0;
        }

        table = search.table();
    };

    let mut blunder = false;

    for position in game_positions.iter_mut() {
        position.set_result(result);

        blunder |= match result {
            1 => position.evaluation() < -0.5,
            -1 => position.evaluation() > 0.5,
            0 => position.evaluation().abs() > 0.75,
            _ => unreachable!(),
        }
    }

    positions.append(&mut game_positions);

    if blunder {
        stats.inc_blunders();
    }

    stats.inc_games();

    if result == 1 {
        stats.inc_white_wins();
    } else if result == -1 {
        stats.inc_black_wins();
    } else {
        stats.inc_draws();
    }
}

fn main() {
    set_hash_size_mb(128);

    tablebase::set_tablebase_directory("syzygy");

    let stats = Arc::new(Stats::zero());

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    thread::scope(|s| {
        for t in 0..THREADS {
            thread::sleep(Duration::from_millis(10 * t as u64));

            let mut writer = BufWriter::new(
                File::create(format!("data/princhess-{timestamp}-{t}.data")).unwrap(),
            );

            let stats = stats.clone();
            let mut rng = Rng::default();

            s.spawn(move || loop {
                let mut positions = Vec::new();

                while positions.len() < DATA_WRITE_RATE {
                    run_game(&stats, &mut positions, &mut rng);
                }

                TrainingPosition::write_batch(&mut writer, &positions).unwrap();

                if t == 0 {
                    print!("{}\r", stats);
                    io::stdout().flush().unwrap();
                }
            });
        }
    });

    println!();
}
