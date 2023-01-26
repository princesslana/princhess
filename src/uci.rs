use options::{set_chess960, set_cpuct, set_hash_size_mb, set_num_threads};
use search::Search;
use search_tree::print_size_list;
use state::State;
use std::io::{stdin, BufRead};
use std::str::{FromStr, SplitWhitespace};
use std::sync::mpsc::{channel, SendError};
use std::thread;
use tablebase::set_tablebase_directory;
use transposition_table::TranspositionTable;

pub type Tokens<'a> = SplitWhitespace<'a>;

const ENGINE_NAME: &str = "Princhess";
const ENGINE_AUTHOR: &str = "Princess Lana";
const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

pub fn main(commands: Vec<String>) {
    let mut search = Search::new(State::default(), TranspositionTable::empty());
    let (sender, receiver) = channel();
    for cmd in commands {
        sender.send(cmd).unwrap();
    }
    {
        let sender = sender.clone();
        thread::spawn(move || -> Result<(), SendError<String>> {
            let stdin = stdin();
            for line in stdin.lock().lines() {
                sender.send(line.unwrap_or_else(|_| "".into()))?;
            }
            sender.send("quit".into())?;
            Ok(())
        });
    }
    for line in receiver {
        debug!("Received '{}'.", line);
        let mut tokens = line.split_whitespace();
        if let Some(first_word) = tokens.next() {
            match first_word {
                "uci"        => uci(),
                "isready"    => println!("readyok"),
                "setoption"  => {
                    let option = UciOption::parse(tokens);

                    if let Some(opt) = option {
                        opt.set();
                    }
                }
                "ucinewgame" => {
                    search = Search::new(State::default(), TranspositionTable::empty());
                }
                "position"   => {
                    if let Some(state) = State::from_tokens(tokens) {
                        debug!("\n{:?}", state.board());
                        let prev_table = search.table();
                        search = Search::new(state, prev_table);
                    } else {
                        error!("Couldn't parse '{}' as position", line);
                    }
                },
                "stop"       => search = search.stop_and_print(),
                "quit"       => return,
                "go"         => search = search.go(tokens, &sender),
                "movelist"   => search.print_move_list(),
                "sizelist"   => print_size_list(),
                "info"       => search.print_info(),
                "features"   => search.print_features(),
                _ => error!("Unknown command: {} (this engine uses a reduced set of commands from the UCI protocol)", first_word)
            }
        }
    }
}

pub fn uci() {
    println!("id name {} {}", ENGINE_NAME, VERSION.unwrap_or("unknown"));
    println!("id author {}", ENGINE_AUTHOR);
    println!("option name Hash type spin min 8 max 65536 default 16");
    println!("option name Threads type spin min 1 max 255 default 1");
    println!("option name SyzygyPath type string");
    println!("option name CPuct type string default 1.79");
    println!("option name UCI_Chess960 type check default false");

    println!("uciok");
}

struct UciOption {
    name: String,
    value: Option<String>,
}

impl UciOption {
    pub fn parse(mut tokens: Tokens) -> Option<UciOption> {
        if tokens.next() != Some("name") {
            return None;
        }

        let mut name = tokens.next()?.to_owned();

        for t in tokens.by_ref() {
            if t == "value" {
                break;
            }

            name = format!("{} {}", name, t);
        }

        let mut value = None;

        for t in tokens {
            value = match value {
                None => Some(t.to_owned()),
                Some(s) => Some(format!("{} {}", s, t)),
            }
        }

        Some(Self {
            name: name.to_lowercase(),
            value,
        })
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn value(&self) -> &Option<String> {
        &self.value
    }

    pub fn set(&self) {
        match self.name().as_str() {
            "syzygypath" => {
                if let Some(path) = self.value() {
                    set_tablebase_directory(path)
                }
            }
            "threads" => self.set_option(set_num_threads),
            "hash" => self.set_option(set_hash_size_mb),
            "cpuct" => self.set_option(set_cpuct),
            "uci_chess960" => self.set_option(set_chess960),
            _ => warn!("Badly formatted or unknown option"),
        }
    }

    fn set_option<F, T>(&self, f: F)
    where
        F: FnOnce(T),
        T: FromStr,
    {
        if let Some(v) = self.value() {
            if let Ok(t) = v.parse() {
                f(t);
            }
        }
    }
}
