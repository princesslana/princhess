use princhess::chess::Move;
use princhess::math::Rng;
use princhess::options::set_hash_size_mb;
use princhess::search::Search;
use princhess::state::State;
use princhess::train::TrainingPosition;
use princhess::transposition_table::LRTable;

use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::BufWriter;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const THREADS: usize = 6;
const PLAYOUTS_PER_MOVE: usize = 1000;
const DATA_WRITE_RATE: usize = 16384;

struct Stats {
    start: Instant,
    games: AtomicU64,
    positions: AtomicU64,
    skipped: AtomicU64,
}

impl Stats {
    pub fn zero() -> Self {
        Self {
            start: Instant::now(),
            games: AtomicU64::new(0),
            positions: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
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
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let games = self.games.load(std::sync::atomic::Ordering::Relaxed);
        let positions = self.positions.load(std::sync::atomic::Ordering::Relaxed);
        let skipped = self.skipped.load(std::sync::atomic::Ordering::Relaxed);
        let seconds = self.start.elapsed().as_secs().max(1);

        write!(
            f,
            "Games: {}, Positions: {}, Skipped: {}, Positions/sec: {}",
            games,
            positions,
            skipped,
            positions / seconds
        )
    }
}

fn run_game(stats: &Stats, positions: &mut Vec<TrainingPosition>) {
    let mut state = State::default();
    let mut table = LRTable::empty();

    let mut prev_moves = [Move::NONE; 4];
    let mut result = 0;

    let mut rng = Rng::default();

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

    loop {
        let search = Search::new(state.clone(), table);

        search.playout_sync(PLAYOUTS_PER_MOVE);

        let best_move = search.best_move();

        prev_moves.rotate_right(1);
        prev_moves[0] = best_move;

        let legal_moves = search.root_node().hots().len();

        if legal_moves > TrainingPosition::MAX_MOVES {
            stats.inc_skipped();
            break;
        } else {
            let mut position = TrainingPosition::from(search.tree());
            position.set_previous_moves(prev_moves);
            positions.push(position);

            stats.inc_positions();
        }

        state.make_move(best_move);

        if !state.is_available_move() {
            if state.is_check() {
                // The stm has been checkmated. Convert to white relative result
                result = state.side_to_move().fold(-1, 1);
            } else {
                result = 0;
            }
            break;
        }

        if state.drawn_by_fifty_move_rule() || state.is_repetition() {
            result = 0;
            break;
        }

        table = search.table();
    }

    for position in positions.iter_mut() {
        position.set_result(result);
    }

    stats.inc_games();
}

fn main() {
    set_hash_size_mb(128);

    let stats = Arc::new(Stats::zero());

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    thread::scope(|s| {
        for t in 0..THREADS {
            let mut writer = BufWriter::new(
                File::create(format!("data/princhess-{timestamp}-{t}.data")).unwrap(),
            );

            let stats = stats.clone();

            s.spawn(move || loop {
                let mut positions = Vec::new();

                while positions.len() < DATA_WRITE_RATE {
                    run_game(&stats, &mut positions);
                }

                TrainingPosition::write_batch(&mut writer, &positions).unwrap();

                if t == 0 {
                    println!("{}", stats);
                }
            });
        }
    });
}
