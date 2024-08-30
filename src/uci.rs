use scoped_threadpool::Pool;
use std::io::stdin;
use std::str::SplitWhitespace;

use crate::options::{SearchOptions, UciOption, UciOptionMap};
use crate::search::Search;
use crate::search_tree::print_size_list;
use crate::state::State;
use crate::tablebase::set_tablebase_directory;

pub type Tokens<'a> = SplitWhitespace<'a>;

const ENGINE_NAME: &str = "Princhess";
const ENGINE_AUTHOR: &str = "Princess Lana";
const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

pub fn main() {
    let mut uci_options = UciOptionMap::default();
    let mut search_options = SearchOptions::from(&uci_options);
    let mut search = Search::with_empty_table(State::default(), search_options);
    let mut search_threads = Pool::new(1);

    let mut next_line: Option<String> = None;

    loop {
        let line = if let Some(line) = next_line.take() {
            line
        } else {
            read_stdin()
        };

        let mut tokens = line.split_whitespace();
        if let Some(first_word) = tokens.next() {
            match first_word {
                "uci" => uci(),
                "isready" => println!("readyok"),
                "setoption" => {
                    if let Some((name, value)) = parse_set_option(tokens) {
                        uci_options.set(&name, &value);

                        if name.to_lowercase().as_str() == "syzygypath" {
                            set_tablebase_directory(&value);
                        }

                        search_options = SearchOptions::from(&uci_options);

                        search =
                            Search::with_empty_table(search.root_state().clone(), search_options);
                    }
                }
                "ucinewgame" => {
                    search = Search::with_empty_table(State::default(), search_options);
                }
                "position" => {
                    if let Some(state) = State::from_tokens(tokens, search_options.is_chess960) {
                        let prev_table = search.table();
                        search = Search::new(state, prev_table, search_options);
                    } else {
                        println!("info string Couldn't parse '{line}' as position");
                    }
                }
                "quit" => return,
                "go" => {
                    if search_threads.thread_count() != search_options.threads {
                        search_threads = Pool::new(search_options.threads);
                    }
                    search.go(&mut search_threads, tokens, &mut next_line);
                }
                "movelist" => search.print_move_list(),
                "sizelist" => print_size_list(),
                "eval" => search.print_eval(),
                _ => (),
            }
        }
    }
}

#[must_use]
pub fn read_stdin() -> String {
    let mut input = String::new();
    stdin().read_line(&mut input).unwrap();
    input
}

pub fn uci() {
    println!("id name {} {}", ENGINE_NAME, VERSION.unwrap_or("unknown"));
    println!("id author {ENGINE_AUTHOR}");

    UciOption::print_all();

    println!("uciok");
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
