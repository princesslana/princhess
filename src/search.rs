use chess::{Color, MoveGen, Piece};
use evaluation::GooseEval;
use features::Model;
use float_ord::FloatOrd;
use mcts::{AsyncSearchOwned, CycleBehaviour, Evaluator, GameState, MCTSManager, MCTS};
use options::get_num_threads;
use policy_features::evaluate_single;
use search_tree::PreviousTable;
use shakmaty_syzygy::Syzygy;
use state::{Move, Outcome, State, StateBuilder};
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;
use tablebase::probe_tablebase_best_move;
use transposition_table::ApproxTable;
use tree_policy::AlphaGoPolicy;
use uci::{Tokens, TIMEUP};

const DEFAULT_MOVE_TIME_SECS: u64 = 10;
const DEFAULT_MOVE_TIME_FRACTION: u32 = 15;

pub const SCALE: f32 = 1e9;

fn policy() -> AlphaGoPolicy {
    AlphaGoPolicy::new(2.0 * SCALE)
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
    pub fn create_manager(
        state: State,
        prev_table: PreviousTable<GooseMCTS>,
    ) -> MCTSManager<GooseMCTS> {
        MCTSManager::new(
            state.freeze(),
            GooseMCTS,
            GooseEval::new(Model::new()),
            policy(),
            ApproxTable::enough_to_hold(GooseMCTS.node_limit()),
            prev_table,
        )
    }

    pub fn new(state: State, prev_table: PreviousTable<GooseMCTS>) -> Self {
        let search = Self::create_manager(state, prev_table).into();
        Self { search }
    }

    pub fn table(self) -> PreviousTable<GooseMCTS> {
        let manager = self.stop_and_print_m();
        manager.table()
    }
    fn stop_and_print_m(self) -> MCTSManager<GooseMCTS> {
        if self.search.num_threads() == 0 {
            return self.search.halt();
        }
        let manager = self.search.halt();
        if let Some(mov) = manager.best_move() {
            let info_str = format!(
                "info nodes {} score cp {} pv{}",
                manager.tree().num_nodes(),
                manager.eval_in_cp(),
                get_pv(&manager)
            );
            info!("{}", info_str);
            println!("{}", info_str);
            println!("bestmove {}", to_uci(mov));
            //manager.tree().display_moves();
        }
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
            .unwrap_or("")
            .parse()
            .ok()
            .map(Duration::from_millis)
    }

    pub fn go(self, mut tokens: Tokens, position_num: u64, sender: &Sender<String>) -> Self {
        let manager = self.stop_and_print_m();

        let state = manager.tree().root_state();
        let player = state.current_player();

        let mvs = state.available_moves();

        if mvs.len() == 1 {
            println!("bestmove {}", to_uci(mvs.as_slice()[0]));
            return Self {
                search: manager.into(),
            };
        } else if state.piece_count() < shakmaty::Chess::MAX_PIECES as u32 {
            if let Some(mv) = probe_tablebase_best_move(state.shakmaty_board()) {
                println!("info tbhits 1");
                println!("bestmove {}", mv.to_uci(shakmaty::CastlingMode::Standard));
                return Self {
                    search: manager.into(),
                };
            } else if state.outcome() != &Outcome::Ongoing {
                // Choose a capture that keeps the outcome the same,
                // if there's no capture, leave it to move eval
                // This is a special case for when the tablebase has determined a result
                // but not a best move
                let mut move_gen = MoveGen::new_legal(&state.board());

                let targets = state.board().combined();
                move_gen.set_iterator_mask(*targets);

                for mv in move_gen {
                    let mut new_state: State =
                        StateBuilder::from(state.shakmaty_board().clone()).into();
                    new_state.make_move(&mv);

                    if new_state.outcome() == state.outcome() {
                        println!("bestmove {}", to_uci(mv));
                        return Self {
                            search: manager.into(),
                        };
                    }
                }
            }
        }

        // This shouldn't happen (being asked to search with no moves available).
        //
        // One known case is the tablebase determiining a win, but not a best move.
        //
        // Just depend upon our move evaluation here.
        if mvs.len() == 0 {
            let move_gen = MoveGen::new_legal(&state.board());
            let mv = move_gen
                .max_by_key(|m| FloatOrd(evaluate_single(state, m)))
                .unwrap();

            println!("bestmove {}", to_uci(mv));
            return Self {
                search: manager.into(),
            };
        }

        let mut move_time = None;
        let mut infinite = false;
        let mut remaining = None;
        let mut sudden_death = true;

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
                        sudden_death = false;
                    }
                }
                "binc" => {
                    if player == Color::Black {
                        sudden_death = false;
                    }
                }
                "movestogo" => {
                    sudden_death = false;
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

            if sudden_death && r < Duration::from_millis(60000) {
                t = r / 60;
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
            search: manager.into_playout_parallel_async(get_num_threads()),
        }
    }

    pub fn print_eval(self) -> Self {
        let manager = self.stop_and_print_m();

        let state = manager.tree().root_state();

        let eval = GooseEval::new(Model::new());

        let moves = state.available_moves();
        let (move_eval, state_eval) = eval.evaluate_new_state(state, &moves);

        println!(
            "cp {} outcome {:?}",
            (state_eval as f32 / (SCALE / 100.)) as i64,
            state.outcome()
        );

        print!("moves ");
        for (i, e) in move_eval.iter().enumerate().take(moves.len()) {
            print!("{}:{:.3} ", moves.as_slice()[i], e);
        }
        println!();

        Self {
            search: manager.into(),
        }
    }

    pub fn nodes_per_sec(self) -> Self {
        let mut manager = self.stop_and_print_m();
        manager.perf_test_to_stderr(get_num_threads());
        Self {
            search: manager.into(),
        }
    }
}

pub fn to_uci(mov: Move) -> String {
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
