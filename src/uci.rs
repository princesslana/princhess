use std::io;
use std::io::{BufRead, Read};

use crate::engine::Engine;
use crate::graph;
use crate::math::Rng;
use crate::nets;
use crate::options::{EngineOptions, UciOption, UciOptionMap};
use crate::state::{self, State};
use crate::tablebase;
use crate::time_management::TimeManagement;

pub mod command;
pub mod parser;

pub use command::{GoParams, PositionSpec, SetOptionParams, UciCommand};
pub use parser::ParseError;

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

    // parse line and dispatch uci command
    pub fn handle_command(&mut self, line: &str, is_interactive: bool) -> (bool, Option<String>) {
        let command = match UciCommand::parse(line) {
            Ok(Some(cmd)) => cmd,
            Ok(None) => return (false, None),
            Err(ParseError::InvalidPosition | ParseError::InvalidFen) => {
                println!("info string Couldn't parse '{line}' as position");
                return (false, None);
            }
            Err(ParseError::MissingOptionName | ParseError::MalformedCommand) => {
                return (false, None);
            }
        };

        self.execute(command, is_interactive)
    }

    /// dispatch a parsed command, returns (quit, next line for go)
    pub fn execute(
        &mut self,
        command: UciCommand<'_>,
        is_interactive: bool,
    ) -> (bool, Option<String>) {
        let mut next_line_from_go = None;
        let mut should_quit = false;

        match command {
            UciCommand::Uci => Self::uci_info(),
            UciCommand::Debug(_) => (),
            UciCommand::IsReady => println!("readyok"),
            UciCommand::SetOption(params) => {
                self.handle_setoption(params.name, params.value);
            }
            UciCommand::UciNewGame => {
                self.engine = Engine::new(State::default(), self.engine_options);
            }
            UciCommand::Position { spec, moves } => {
                self.handle_position(spec, moves);
            }
            UciCommand::Quit => should_quit = true,
            UciCommand::Stop | UciCommand::PonderHit => (),
            UciCommand::Go(params) => {
                next_line_from_go = self.handle_go(&params, is_interactive);
            }
            UciCommand::MoveList(moves) => self.engine.print_move_list(moves.split_whitespace()),
            UciCommand::SizeList => graph::print_size_list(),
            UciCommand::Eval => self.engine.print_eval(),
            UciCommand::Bench => self.run_bench(),
            UciCommand::RandomOpen => self.generate_random_opening(),
            UciCommand::Fingerprint => Self::fingerprint(),
            UciCommand::UnknownCommand(_) => (),
        }

        (should_quit, next_line_from_go)
    }

    // apply setoption, syzygypath updates tablebase dir
    fn handle_setoption(&mut self, name: &str, value: Option<&str>) {
        let Some(value) = value else {
            println!("info string Option '{name}' is not a button and requires a value");
            return;
        };

        let root_state = self.engine.root_state().clone();

        self.options.set(name, value);
        self.engine_options = EngineOptions::from(&self.options);

        if name.eq_ignore_ascii_case("syzygypath") {
            match tablebase::set_tablebase_directory(value) {
                Ok(()) => println!("info string Success initializing tablebase at {value}"),
                Err(()) => println!("info string Error initializing tablebase at {value}"),
            }
        }

        self.engine = Engine::new(root_state, self.engine_options);
    }

    // set up root state from startpos/fen and replay moves without allocating
    fn handle_position(&mut self, spec: PositionSpec<'_>, moves: &str) {
        let mut state = match spec {
            PositionSpec::Startpos => State::default(),
            PositionSpec::Fen(fen) => State::from_fen(fen),
        };

        for mov_str in moves.split_whitespace() {
            let mut applied = false;
            for mov in state.available_moves() {
                if mov.matches_uci(mov_str, self.engine_options.is_chess960) {
                    state.make_move(mov);
                    applied = true;
                    break;
                }
            }
            if !applied {
                println!("info string Couldn't parse '{mov_str}' as move");
                return;
            }
        }

        self.engine.set_root_state(state);
    }

    fn handle_go(&self, params: &GoParams, is_interactive: bool) -> Option<String> {
        self.engine.go(params, is_interactive)
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

        let nps = (total_nodes * 1000)
            .checked_div(total_elapsed_time_ms)
            .unwrap_or(0);

        println!("Bench: {total_nodes} nodes {nps} nps");
    }

    pub fn uci_info() {
        println!("id name {} {}", ENGINE_NAME, VERSION.unwrap_or("unknown"));
        println!("id author {ENGINE_AUTHOR}");

        UciOption::print_all();

        println!("uciok");
    }

    fn fingerprint() {
        println!(
            "info string git {}",
            option_env!("PRINCHESS_GIT_DESCRIBE").unwrap_or("unknown")
        );
        println!(
            "info string rustc {}",
            option_env!("PRINCHESS_RUSTC_VERSION").unwrap_or("unknown")
        );
        println!(
            "info string target-cpu {}",
            option_env!("PRINCHESS_TARGET_CPU").unwrap_or("unknown")
        );
        println!("info string net-md5-value {}", nets::NET_MD5_VALUE);
        println!("info string net-md5-mg-policy {}", nets::NET_MD5_MG_POLICY);
        println!("info string net-md5-eg-policy {}", nets::NET_MD5_EG_POLICY);
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
    read_bounded_line(&mut io::stdin().lock())
}

// max bytes kept per input line; uci lines are tiny, this stops hostile input growing memory
const MAX_UCI_LINE_BYTES: u64 = 1 << 20;

// read one line capped at MAX_UCI_LINE_BYTES; overlong lines are drained and ignored
fn read_bounded_line<R: BufRead>(reader: &mut R) -> String {
    let mut input = String::new();
    let bytes_read = (&mut *reader)
        .take(MAX_UCI_LINE_BYTES + 1)
        .read_line(&mut input)
        .unwrap();

    if bytes_read as u64 > MAX_UCI_LINE_BYTES {
        // over the cap: drain the rest of the line so the next read starts clean
        let mut drain = Vec::new();
        let _ = reader.read_until(b'\n', &mut drain);
        println!("info string ignoring oversized input line");
        return String::new();
    }
    input
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_bounded_line_keeps_normal_lines() {
        let mut cursor = Cursor::new(b"isready\n".to_vec());
        assert_eq!(read_bounded_line(&mut cursor), "isready\n");
    }

    #[test]
    fn test_bounded_line_drains_oversized_line() {
        let big = "x".repeat(2 * 1024 * 1024);
        let data = format!("{big}\nquit\n");
        let mut cursor = Cursor::new(data.into_bytes());
        assert_eq!(read_bounded_line(&mut cursor), "");
        // rest of the overlong line was drained, next read gets the following line
        assert_eq!(read_bounded_line(&mut cursor), "quit\n");
    }
}

