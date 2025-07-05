use bytemuck::allocation;
use princhess::chess::{Board, Move};
use princhess::engine::Engine;
use princhess::evaluation;
use princhess::math::Rng;
use princhess::options::{EngineOptions, MctsOptions};
use princhess::state::State;
use princhess::tablebase::{self, Wdl};
use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::ops::Neg;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use princhess_train::data::TrainingPosition;

const HASH_SIZE_MB: usize = 128;

const VISITS_PER_POSITION: u64 = 5000;
const THREADS: u64 = 5;
const DFRC_PCT: u64 = 10;

const MAX_POSITIONS_PER_FILE: u64 = 20_000_000;
const MAX_POSITIONS_TOTAL: u64 = MAX_POSITIONS_PER_FILE * THREADS;
const MAX_VARIATIONS: usize = 16;

struct Stats {
    start: Instant,
    games: AtomicU64,
    positions: AtomicU64,
    skipped: AtomicU64,
    aborted: AtomicU64,
    white_wins: AtomicU64,
    black_wins: AtomicU64,
    draws: AtomicU64,
    blunders: AtomicU64,
    variations: AtomicUsize,
    nodes: AtomicUsize,
    playouts: AtomicUsize,
    depth: AtomicUsize,
    seldepth: AtomicUsize,
}

struct GameStats {
    pub positions: u64,
    pub skipped: u64,
    pub blunders: u64,
    pub variations: usize,
    pub nodes: usize,
    pub playouts: usize,
    pub depth: usize,
    pub seldepth: usize,
}

impl Stats {
    pub fn zero() -> Self {
        Self {
            start: Instant::now(),
            games: AtomicU64::new(0),
            positions: AtomicU64::new(0),
            skipped: AtomicU64::new(0),
            aborted: AtomicU64::new(0),
            white_wins: AtomicU64::new(0),
            black_wins: AtomicU64::new(0),
            draws: AtomicU64::new(0),
            blunders: AtomicU64::new(0),
            variations: AtomicUsize::new(0),
            nodes: AtomicUsize::new(0),
            playouts: AtomicUsize::new(0),
            depth: AtomicUsize::new(0),
            seldepth: AtomicUsize::new(0),
        }
    }

    fn add_game(&self, game: &GameStats) {
        self.games.fetch_add(1, Ordering::Relaxed);
        self.positions.fetch_add(game.positions, Ordering::Relaxed);
        self.skipped.fetch_add(game.skipped, Ordering::Relaxed);
        self.blunders.fetch_add(game.blunders, Ordering::Relaxed);
        self.variations
            .fetch_add(game.variations, Ordering::Relaxed);
        self.nodes.fetch_add(game.nodes, Ordering::Relaxed);
        self.playouts.fetch_add(game.playouts, Ordering::Relaxed);
        self.depth.fetch_add(game.depth, Ordering::Relaxed);
        self.seldepth.fetch_add(game.seldepth, Ordering::Relaxed);
    }

    pub fn add_white_win(&self, game: &GameStats) {
        self.add_game(game);
        self.white_wins.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_black_win(&self, game: &GameStats) {
        self.add_game(game);
        self.black_wins.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_draw(&self, game: &GameStats) {
        self.add_game(game);
        self.draws.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_aborted(&self) {
        self.aborted.fetch_add(1, Ordering::Relaxed);
    }
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let games = self.games.load(Ordering::Relaxed);
        let white_wins = self.white_wins.load(Ordering::Relaxed);
        let draws = self.draws.load(Ordering::Relaxed);
        let black_wins = self.black_wins.load(Ordering::Relaxed);
        let positions = self.positions.load(Ordering::Relaxed);
        let skipped = self.skipped.load(Ordering::Relaxed);
        let aborted = self.aborted.load(Ordering::Relaxed);
        let blunders = self.blunders.load(Ordering::Relaxed);
        let variations = self.variations.load(Ordering::Relaxed);
        let nodes = self.nodes.load(Ordering::Relaxed);
        let playouts = self.playouts.load(Ordering::Relaxed);
        let depth = self.depth.load(Ordering::Relaxed);
        let seldepth = self.seldepth.load(Ordering::Relaxed);
        let seconds = self.start.elapsed().as_secs().max(1);

        write!(
            f,
            "G {:>7} | +{:>2}={:>2}-{:>2} | B {:>4.1} V {:>4.1} S {:>3.1} X {:>4.1} | N {:>4} P {:>4} | D {:>2}/{:>2} | P {:>5.1}m ({:>2}/g, {:>3.1}m/h)",
            games,
            white_wins * 100 / games,
            draws * 100 / games,
            black_wins * 100 / games,
            blunders  as f32 * 100. / games as f32,
            variations as f32 * 100. / positions as f32,
            skipped as f32 * 100. / positions as f32,
            aborted as f32 * 100. / games as f32,
            nodes / positions as usize,
            playouts / positions as usize,
            depth / positions as usize,
            seldepth / positions as usize,
            positions as f32 / 1000000.0,
            positions / games,
            (positions * 3600 / seconds) as f32 / 1000000.0
        )
    }
}

impl GameStats {
    pub fn zero() -> Self {
        Self {
            positions: 0,
            skipped: 0,
            blunders: 0,
            variations: 0,
            nodes: 0,
            playouts: 0,
            depth: 0,
            seldepth: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum GameResult {
    WhiteWin,
    Draw,
    BlackWin,
    Aborted,
}

impl From<GameResult> for i8 {
    fn from(result: GameResult) -> Self {
        match result {
            GameResult::WhiteWin => 1,
            GameResult::Draw => 0,
            GameResult::BlackWin => -1,
            GameResult::Aborted => unreachable!(),
        }
    }
}

impl Neg for GameResult {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            GameResult::WhiteWin => GameResult::BlackWin,
            GameResult::BlackWin => GameResult::WhiteWin,
            GameResult::Draw => GameResult::Draw,
            GameResult::Aborted => GameResult::Aborted,
        }
    }
}

fn run_game(stats: &Stats, positions: &mut Vec<TrainingPosition>, rng: &mut Rng) {
    let mut variations = Vec::with_capacity(MAX_VARIATIONS + 1);
    let mut variations_count = 0;
    let mut seen_positions = HashSet::new();

    variations.push(random_start(rng));

    let mcts_options = MctsOptions::default();

    let engine_options = EngineOptions {
        mcts_options,
        hash_size_mb: HASH_SIZE_MB,
        ..EngineOptions::default()
    };

    while let Some(mut state) = variations.pop() {
        let mut game_stats = GameStats::zero();
        let mut engine = Engine::new(state.clone(), engine_options);

        if !state.is_available_move() {
            continue;
        }

        if state.drawn_by_fifty_move_rule()
            || state.is_repetition()
            || state.board().is_insufficient_material()
        {
            continue;
        }

        if tablebase::probe_wdl(state.board()).is_some() {
            continue;
        }

        let mut game_positions = Vec::with_capacity(256);

        let result = loop {
            engine.set_root_state(state.clone());
            let legal_moves = engine.root_node().edges().len();

            if legal_moves > 1 {
                let visits = engine.root_node().visits();
                engine.playout_sync(VISITS_PER_POSITION.saturating_sub(visits));
            }

            let best_move = engine.best_move();

            if variations_count < MAX_VARIATIONS {
                let variation = engine.most_visited_move();

                if variation != best_move && state.phase() > 18 {
                    game_stats.variations += 1;
                    let mut state = state.clone();
                    state.make_move(variation);
                    variations.push(state);
                    variations_count += 1;
                }
            }

            if legal_moves == 1 || legal_moves > TrainingPosition::MAX_MOVES {
                game_stats.skipped += 1;
            } else {
                let position = TrainingPosition::from(engine.mcts());

                if position.evaluation() > 0.95 {
                    break GameResult::WhiteWin;
                } else if position.evaluation() < -0.95 {
                    break GameResult::BlackWin;
                }

                game_positions.push(position);

                game_stats.positions += 1;
                game_stats.nodes += engine.mcts().num_nodes();
                game_stats.playouts += engine.mcts().playouts();
                game_stats.depth += engine.mcts().depth();
                game_stats.seldepth += engine.mcts().max_depth();
            }

            state.make_move(best_move);

            if !state.is_available_move() {
                break if state.is_check() {
                    // The stm has been checkmated. Convert to white relative result
                    state
                        .side_to_move()
                        .fold(GameResult::BlackWin, GameResult::WhiteWin)
                } else {
                    GameResult::Draw
                };
            }

            if state.drawn_by_fifty_move_rule()
                || state.is_repetition()
                || state.board().is_insufficient_material()
            {
                break GameResult::Draw;
            }

            if let Some(wdl) = tablebase::probe_wdl(state.board()) {
                let result = match wdl {
                    Wdl::Win => GameResult::WhiteWin,
                    Wdl::Draw => GameResult::Draw,
                    Wdl::Loss => GameResult::BlackWin,
                };

                break state.side_to_move().fold(result, -result);
            }

            if !seen_positions.insert(state.hash()) {
                game_positions.clear();
                break GameResult::Aborted;
            }
        };

        let mut blunder = false;

        for position in game_positions.iter_mut() {
            position.set_result(i8::from(result));

            blunder |= match result {
                GameResult::WhiteWin => position.evaluation() < -0.5,
                GameResult::BlackWin => position.evaluation() > 0.5,
                GameResult::Draw => position.evaluation().abs() > 0.75,
                GameResult::Aborted => false,
            }
        }

        positions.append(&mut game_positions);

        if blunder {
            game_stats.blunders += 1;
        }

        match result {
            GameResult::WhiteWin => stats.add_white_win(&game_stats),
            GameResult::Draw => stats.add_draw(&game_stats),
            GameResult::BlackWin => stats.add_black_win(&game_stats),
            GameResult::Aborted => stats.add_aborted(),
        }
    }
}

fn random_start(rng: &mut Rng) -> State {
    let startpos = if rng.next_u64() % 100 < DFRC_PCT {
        Board::dfrc(rng.next_usize() % 960, rng.next_usize() % 960)
    } else {
        Board::startpos()
    };

    let mut state = State::from_board(startpos);

    for p in 0..16 {
        let t = 1. + ((p as f32) / 8.).powi(2);

        let best_move = select_weighted_random_move(&state, t, rng);

        if best_move == Move::NONE {
            return state;
        }

        state.make_move(best_move);
    }

    state
}

fn select_weighted_random_move(state: &State, t: f32, rng: &mut Rng) -> Move {
    let moves = state.available_moves();

    if moves.is_empty() {
        return Move::NONE;
    }

    let policy = evaluation::policy(state, &moves, t);

    moves[rng.weighted(&policy)]
}

fn main() {
    tablebase::set_tablebase_directory("syzygy");

    let stats = Arc::new(Stats::zero());

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    thread::scope(|s| {
        for t in 0..THREADS {
            thread::sleep(Duration::from_millis(10 * t));

            let mut writer = BufWriter::new(
                File::create(format!("data/princhess-{timestamp}-{t}.data")).unwrap(),
            );

            {
                let stats = stats.clone();
                let mut rng = Rng::default();

                s.spawn(move || {
                    let mut positions = Vec::new();

                    let mut buffer: Box<[TrainingPosition; TrainingPosition::BUFFER_COUNT]> =
                        allocation::zeroed_box();

                    while stats.positions.load(Ordering::Relaxed) < MAX_POSITIONS_TOTAL {
                        while positions.len() < TrainingPosition::BUFFER_COUNT {
                            run_game(&stats, &mut positions, &mut rng);
                        }

                        buffer.copy_from_slice(
                            positions.drain(..TrainingPosition::BUFFER_COUNT).as_slice(),
                        );

                        TrainingPosition::write_buffer(&mut writer, &buffer[..]);
                    }
                });
            }
        }

        let stats = stats.clone();

        s.spawn(move || {
            while stats.positions.load(Ordering::Relaxed) < MAX_POSITIONS_TOTAL {
                thread::sleep(Duration::from_secs(1));
                print!("{stats}\r");
                io::stdout().flush().unwrap();
            }
            println!("\nStopping...");
        });
    });

    println!("{stats}");
}
