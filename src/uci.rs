use options::{set_hash_size_mb, set_num_threads};
use search::Search;
use search_tree::empty_previous_table;
use state::State;
use std::io::{stdin, BufRead};
use std::str::SplitWhitespace;
use std::sync::mpsc::{channel, SendError};
use std::thread;
use tablebase::set_tablebase_directory;

pub type Tokens<'a> = SplitWhitespace<'a>;

pub const TIMEUP: &str = "timeup";
const ENGINE_NAME: &str = "Princhess";
const ENGINE_AUTHOR: &str = "Princess Lana";
const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

pub fn main(commands: Vec<String>) {
    let mut search = Search::new(State::default(), empty_previous_table());
    let mut position_num: u64 = 0;
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

                    match option {
                        Some(opt) if opt.name() == "syzygypath" => {
                            if let Some(path) = opt.value() {
                                set_tablebase_directory(path)
                            }
                        }
                        Some(opt) if opt.name() == "threads" => {
                            if let Some(v) = opt.value() {
                                if let Some(t) = v.parse().ok() {
                                    set_num_threads(t)
                                }
                            }
                        }
                        Some(opt) if opt.name() == "hash" => {
                            if let Some(v) = opt.value() {
                                if let Some(t) = v.parse().ok() {
                                    set_hash_size_mb(t)
                                }
                            }
                        }
                        _ => warn!("Badly formatted or unknown option"),
                       }
                }
                "ucinewgame" => {
                    position_num += 1;
                    search = Search::new(State::default(), empty_previous_table());
                }
                "position"   => {
                    position_num += 1;
                    if let Some(state) = State::from_tokens(tokens) {
                        debug!("\n{}", state.board());
                        let prev_table = search.table();
                        search = Search::new(state, prev_table);
                    } else {
                        error!("Couldn't parse '{}' as position", line);
                    }
                },
                "stop"       => search = search.stop_and_print(),
                TIMEUP       => {
                    let old_position_num = tokens.next().and_then(|x| x.parse().ok()).unwrap_or(0);
                    if position_num == old_position_num {
                        search = search.stop_and_print();
                    }
                }
                "quit"       => return,
                "n/s"        => search = search.nodes_per_sec(),
                "go"         => {
                    search = search.go(tokens, position_num, &sender);
                },
                "eval"       => search = search.print_eval(),
                "info"       => search.print_info(),
                _ => error!("Unknown command: {} (this engine uses a reduced set of commands from the UCI protocol)", first_word)
            }
        }
    }
}

pub fn uci() {
    println!("id name {} {}", ENGINE_NAME, VERSION.unwrap_or("unknown"));
    println!("id author {}", ENGINE_AUTHOR);
    println!("option name Hash type spin min 1 max 65536 default 1");
    println!("option name Threads type spin min 1 max 255 default 1");
    println!("option name SyzygyPath type string");
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

        while let Some(t) = tokens.next() {
            if t == "value" {
                break;
            }

            name = format!("{} {}", name, t);
        }

        let mut value = None;

        while let Some(t) = tokens.next() {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn en_passant() {
        let s = String::from("startpos moves g1f3 g8f6 d2d4 b8c6 b1c3 e7e6 c1g5 h7h6 g5h4 g7g5 h4g3 f8b4 d1d3 g5g4 f3d2 e8g8 e1c1 d7d6 a2a3 b4a5 d2c4 a5c3 d3c3 c8d7 c3b3 b7b6 g3h4 a7a5 a3a4 d8e7 c4e3 h6h5 g2g3 d6d5 f1g2 f8b8 c2c4 b6b5 c4b5 c6b4 c1b1 c7c6 b5c6 b4c6 b3c3 b8b4 c3a3 a8b8 d1d3 b4b2 a3b2 b8b2 b1b2 e7b4 d3b3 b4d4 b2b1 c6b4 h1d1 d4e5 d1c1 d7a4 b3a3 a4b5 f2f4 g4f3");
        let tokens = s.split_whitespace();
        State::from_tokens(tokens).unwrap();
    }
}
