use args::options;
use chess::{Color, MoveGen, Piece};
use evaluation::GooseEval;
use features::Model;
use mcts::GameState;
use mcts::{AsyncSearchOwned, CycleBehaviour, MCTSManager, MCTS};
use state::{Move, State};
use std::cmp::max;
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;
use transposition_table::ApproxTable;
use tree_policy::AlphaGoPolicy;
use uci::{Tokens, TIMEUP};

const DEFAULT_MOVE_TIME_SECS: u64 = 10;
const DEFAULT_MOVE_TIME_FRACTION: u32 = 15;

pub const SCALE: f32 = 1e9;

fn policy() -> AlphaGoPolicy {
    AlphaGoPolicy::new(5.0 * SCALE)
}

fn num_threads() -> usize {
    max(1, options().num_threads)
}

pub struct GooseMCTS;
pub struct ThreadSentinel;

impl Default for ThreadSentinel {
    fn default() -> Self {
        info!("Search thread created.");
        ThreadSentinel
    }
}
impl Drop for ThreadSentinel {
    fn drop(&mut self) {
        info!("Search thread destroyed.");
    }
}

impl MCTS for GooseMCTS {
    type State = State;
    type Eval = GooseEval;
    type TreePolicy = AlphaGoPolicy;
    type NodeData = ();
    type ExtraThreadData = ThreadSentinel;
    type TranspositionTable = ApproxTable<Self>;
    type PlayoutData = ();

    fn node_limit(&self) -> usize {
        4_000_000
    }
    fn virtual_loss(&self) -> i64 {
        SCALE as i64
    }
    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseThisEvalWhenCycleDetected(0)
    }
}

pub struct Search {
    search: AsyncSearchOwned<GooseMCTS>,
}

impl Search {
    pub fn create_manager(state: State) -> MCTSManager<GooseMCTS> {
        MCTSManager::new(
            state.freeze(),
            GooseMCTS,
            GooseEval::from(Model::new()),
            policy(),
            ApproxTable::enough_to_hold(GooseMCTS.node_limit()),
        )
    }
    pub fn new(state: State) -> Self {
        let search = Self::create_manager(state).into();
        Self { search }
    }
    fn stop_and_print_m(self) -> MCTSManager<GooseMCTS> {
        if self.search.num_threads() == 0 {
            return self.search.halt();
        }
        let manager = self.search.halt();
        if let Some(mov) = manager.best_move() {
            let info_str = format!(
                "info depth {} score cp {} pv{}",
                manager.tree().num_nodes(),
                manager
                    .principal_variation_info(1)
                    .get(0)
                    .map(|x| x.sum_rewards() / x.visits() as i64 / (SCALE / 100.) as i64)
                    .unwrap_or(0),
                get_pv(&manager)
            );
            info!("{}", info_str);
            println!("{}", info_str);
            println!("bestmove {}", to_uci(mov));
        }
        let manager = manager.reset();
        manager
    }
    pub fn stop_and_print(self) -> Self {
        Self {
            search: self.stop_and_print_m().into(),
        }
    }

    fn parse_ms(tokens: &mut Tokens) -> Option<Duration> {
        tokens
            .next()
            .unwrap_or("".into())
            .parse()
            .ok()
            .map(|ms| Duration::from_millis(ms))
    }

    pub fn go(self, mut tokens: Tokens, position_num: u64, sender: &Sender<String>) -> Self {
        let manager = self.stop_and_print_m();

        let state = manager.tree().root_state();
        let player = state.current_player();

        let mut mvs = MoveGen::new_legal(state.board());

        if mvs.len() == 1 {
            println!("bestmove {}", to_uci(mvs.next().unwrap()));
            return Self { search: manager.into() }
        }


        let mut move_time = None;
        let mut infinite = false;
        let mut remaining = None;
        let mut increment = Duration::from_secs(0);

        while let Some(s) = tokens.next() {
            match s {
                "movetime" => move_time = Self::parse_ms(&mut tokens),
                "wtime" => {
                    if player == Color::White {
                        remaining = Self::parse_ms(&mut tokens)
                    }
                }
                "btime" => {
                    if player == Color::Black {
                        remaining = Self::parse_ms(&mut tokens)
                    }
                }
                "winc" => {
                    if player == Color::White {
                        increment = Self::parse_ms(&mut tokens).unwrap_or(increment)
                    }
                }
                "binc" => {
                    if player == Color::Black {
                        increment = Self::parse_ms(&mut tokens).unwrap_or(increment)
                    }
                }
                "infinite" => infinite = true,
                _ => (),
            }
        }

        let mut think_time = Some(Duration::from_secs(DEFAULT_MOVE_TIME_SECS));

        if infinite {
            think_time = None
        } else if let Some(mt) = move_time {
            think_time = Some(mt)
        } else if let Some(r) = remaining {
            let mut t = r / DEFAULT_MOVE_TIME_FRACTION;

            if state.is_endgame() {
                debug!("Endgame");
                t += increment;

                t = t.min(r - increment / 2);
            } else if state.is_opening() {
                debug!("Opening");
                t /= 2;
            } else {
                debug!("Middlegame");
            }

            think_time = Some(t)
        }

        if let Some(t) = think_time {
            let sender = sender.clone();
            thread::spawn(move || {
                thread::sleep(t);
                let _ = sender.send(format!("{} {}", TIMEUP, position_num));
            });
        }
        Self {
            search: manager.into_playout_parallel_async(num_threads()),
        }
    }

    pub fn nodes_per_sec(self) -> Self {
        let mut manager = self.stop_and_print_m().reset();
        manager.perf_test_to_stderr(num_threads());
        Self {
            search: manager.into(),
        }
    }
}

fn to_uci(mov: Move) -> String {
    let promo = match mov.get_promotion() {
        Some(Piece::Queen) => "q",
        Some(Piece::Rook) => "r",
        Some(Piece::Knight) => "n",
        Some(Piece::Bishop) => "b",
        Some(_) => unreachable!(),
        None => "",
    };
    format!("{}{}{}", mov.get_source(), mov.get_dest(), promo)
}

fn get_pv(m: &MCTSManager<GooseMCTS>) -> String {
    m.principal_variation(10)
        .into_iter()
        .map(|x| format!(" {}", to_uci(x)))
        .collect()
}
