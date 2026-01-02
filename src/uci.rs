use std::io;
use std::str::SplitWhitespace;

use crate::engine::Engine;
use crate::graph;
use crate::math::Rng;
use crate::options::{EngineOptions, UciOption, UciOptionMap};
use crate::state::{self, State};
use crate::tablebase;
use crate::time_management::TimeManagement;

pub type Tokens<'a> = SplitWhitespace<'a>;

const ENGINE_NAME: &str = "Princhess";
const ENGINE_AUTHOR: &str = "Princess Lana";
const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

const BENCH_FENS: &str = include_str!("../resources/fens.txt");
const BENCH_PLAYOUTS_PER_POSITION: u64 = 5000;

pub struct Uci {
    options: UciOptionMap,
    engine_options: EngineOptions,
    engine: Engine,
}

impl Uci {
    #[must_use]
    pub fn new() -> Self {
        let options = UciOptionMap::default();
        let engine_options = EngineOptions::from(&options);
        let engine = Engine::new(State::default(), engine_options);

        Self {
            options,
            engine_options,
            engine,
        }
    }

    pub fn main_loop(&mut self) {
        let mut next_line: Option<String> = None;

        loop {
            let line = if let Some(line) = next_line.take() {
                line
            } else {
                read_stdin()
            };

            let (quit, returned_next_line) = self.handle_command(&line, true);
            next_line = returned_next_line;

            if quit {
                return;
            }
        }
    }

    pub fn handle_command(&mut self, line: &str, is_interactive: bool) -> (bool, Option<String>) {
        let mut tokens = line.split_whitespace();
        let mut next_line_from_go = None;
        let mut should_quit = false;

        if let Some(first_word) = tokens.next() {
            match first_word {
                "uci" => Self::uci_info(),
                "isready" => println!("readyok"),
                "setoption" => self.handle_setoption(tokens),
                "ucinewgame" => {
                    self.engine = Engine::new(State::default(), self.engine_options);
                }
                "position" => self.handle_position(tokens, line),
                "quit" => should_quit = true,
                "go" => {
                    next_line_from_go = self.handle_go(tokens, is_interactive);
                }
                "movelist" => self.engine.print_move_list(tokens),
                "sizelist" => graph::print_size_list(),
                "eval" => self.engine.print_eval(),
                "bench" => self.run_bench(),
                "randomopen" => self.generate_random_opening(),
                _ => (),
            }
        }
        (should_quit, next_line_from_go)
    }

    fn handle_setoption(&mut self, tokens: Tokens) {
        if let Some((name, value)) = parse_set_option(tokens) {
            let root_state = self.engine.root_state().clone();

            self.options.set(&name, &value);
            self.engine_options = EngineOptions::from(&self.options);

            if name.eq_ignore_ascii_case("syzygypath") {
                match tablebase::set_tablebase_directory(&value) {
                    Ok(()) => println!("info string Success initializing tablebase at {value}"),
                    Err(()) => println!("info string Error initializing tablebase at {value}"),
                }
            }

            self.engine = Engine::new(root_state, self.engine_options);
        }
    }

    fn handle_position(&mut self, tokens: Tokens, line: &str) {
        if let Some(state) = State::from_tokens(tokens, self.engine_options.is_chess960) {
            self.engine.set_root_state(state);
        } else {
            println!("info string Couldn't parse '{line}' as position");
        }
    }

    fn handle_go(&self, tokens: Tokens, is_interactive: bool) -> Option<String> {
        self.engine.go(tokens, is_interactive)
    }

    fn run_bench(&mut self) {
        let mut total_nodes = 0;
        let mut total_elapsed_time_ms = 0;

        for fen_line in BENCH_FENS.lines().filter(|line| !line.is_empty()) {
            println!("info string {fen_line}");

            let state = State::from_fen(fen_line);
            let local_engine = Engine::new(state, self.engine_options);
            let time_management = TimeManagement::infinite();

            local_engine.playout_sync(BENCH_PLAYOUTS_PER_POSITION);
            total_nodes += local_engine.mcts().num_nodes() as u64;
            total_elapsed_time_ms += time_management.elapsed().as_millis() as u64;

            local_engine
                .mcts()
                .print_info(&time_management, local_engine.table_full());
        }

        let nps = if total_elapsed_time_ms > 0 {
            (total_nodes * 1000) / total_elapsed_time_ms
        } else {
            0
        };

        println!("Bench: {total_nodes} nodes {nps} nps");
    }

    pub fn uci_info() {
        println!("id name {} {}", ENGINE_NAME, VERSION.unwrap_or("unknown"));
        println!("id author {ENGINE_AUTHOR}");

        UciOption::print_all();

        println!("uciok");
    }

    fn generate_random_opening(&mut self) {
        let mut rng = Rng::default();
        let (moves_played, state) = state::generate_random_opening(&mut rng, 0); // No DFRC

        let move_strs: Vec<String> = moves_played
            .iter()
            .map(|mv| mv.to_uci(self.engine_options.is_chess960))
            .collect();

        // Set the engine position to the generated opening
        self.engine.set_root_state(state);

        println!("info string {}", move_strs.join(" "));
    }
}

impl Default for Uci {
    fn default() -> Self {
        Self::new()
    }
}

/// Reads a line from standard input.
///
/// # Panics
///
/// Panics if reading from stdin fails.
#[must_use]
pub fn read_stdin() -> String {
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input
}

#[must_use]
pub fn parse_set_option(mut tokens: Tokens) -> Option<(String, String)> {
    if tokens.next() != Some("name") {
        return None;
    }

    let mut name = tokens.next()?.to_owned();

    for t in tokens.by_ref() {
        if t == "value" {
            break;
        }

        name = format!("{name} {t}");
    }

    let mut value = None;

    for t in tokens {
        value = match value {
            None => Some(t.to_owned()),
            Some(s) => Some(format!("{s} {t}")),
        }
    }

    Some((name, value?))
}
