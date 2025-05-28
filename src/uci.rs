use scoped_threadpool::Pool;
use std::io::stdin;
use std::str::SplitWhitespace;

use crate::options::{SearchOptions, UciOption, UciOptionMap};
use crate::search::{Search, TimeManagement};
use crate::search_tree::print_size_list;
use crate::state::State;
use crate::tablebase::set_tablebase_directory;

pub type Tokens<'a> = SplitWhitespace<'a>;

const ENGINE_NAME: &str = "Princhess";
const ENGINE_AUTHOR: &str = "Princess Lana";
const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

const BENCH_FENS: &str = include_str!("../resources/fens.txt");
const BENCH_PLAYOUTS_PER_POSITION: u64 = 5000;

pub struct Uci {
    options: UciOptionMap,
    search_options: SearchOptions,
    search: Search,
    search_threads: Pool,
}

impl Uci {
    #[must_use]
    pub fn new() -> Self {
        let options = UciOptionMap::default();
        let search_options = SearchOptions::from(&options);
        let search_threads = Pool::new(1);
        let search = Search::new(State::default(), search_options);

        Self {
            options,
            search_options,
            search,
            search_threads,
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
                "setoption" => {
                    if let Some((name, value)) = parse_set_option(tokens) {
                        let root_state = self.search.root_state().clone();

                        self.options.set(&name, &value);
                        self.search_options = SearchOptions::from(&self.options);

                        match name.to_lowercase().as_str() {
                            "threads" => {
                                self.search_threads = Pool::new(self.search_options.threads);
                            }
                            "syzygypath" => {
                                set_tablebase_directory(&value);
                            }
                            _ => (),
                        }

                        self.search = Search::new(root_state, self.search_options);
                    }
                }
                "ucinewgame" => {
                    self.search = Search::new(State::default(), self.search_options);
                }
                "position" => {
                    if let Some(state) = State::from_tokens(tokens, self.search_options.is_chess960)
                    {
                        self.search.set_root_state(state);
                    } else {
                        println!("info string Couldn't parse '{line}' as position");
                    }
                }
                "quit" => should_quit = true,
                "go" => {
                    next_line_from_go =
                        self.search
                            .go(&mut self.search_threads, tokens, is_interactive);
                }
                "movelist" => self.search.print_move_list(),
                "sizelist" => print_size_list(),
                "eval" => self.search.print_eval(),
                "bench" => {
                    self.run_bench();
                }
                _ => (),
            }
        }
        (should_quit, next_line_from_go)
    }

    fn run_bench(&mut self) {
        let mut total_nodes = 0;
        let mut total_elapsed_time_ms = 0;

        for fen_line in BENCH_FENS.lines().filter(|line| !line.is_empty()) {
            println!("info string {fen_line}");

            let state = State::from_fen(fen_line);
            let local_search = Search::new(state, self.search_options);
            let time_management = TimeManagement::infinite();

            local_search.playout_sync(BENCH_PLAYOUTS_PER_POSITION);
            total_nodes += local_search.tree().num_nodes() as u64;
            total_elapsed_time_ms += time_management.elapsed().as_millis() as u64;

            local_search
                .tree()
                .print_info(&time_management, local_search.table_full());
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
}

impl Default for Uci {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use]
pub fn read_stdin() -> String {
    let mut input = String::new();
    stdin().read_line(&mut input).unwrap();
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
