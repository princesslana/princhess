use options::{
    set_cpuct, set_cpuct_base, set_cpuct_factor, set_hash_size_mb, set_mate_score, set_num_threads,
    set_policy_bad_capture_factor, set_policy_good_capture_factor, set_policy_softmax_temp,
};
use search::Search;
use search_tree::{empty_previous_table, print_size_list};
use state::State;
use std::io::{stdin, BufRead};
use std::str::{FromStr, SplitWhitespace};
use std::sync::mpsc::{channel, SendError};
use std::thread;
use tablebase::set_tablebase_directory;

pub type Tokens<'a> = SplitWhitespace<'a>;

const ENGINE_NAME: &str = "Princhess";
const ENGINE_AUTHOR: &str = "Princess Lana";
const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

pub fn main(commands: Vec<String>) {
    let mut search = Search::new(State::default(), empty_previous_table());
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
                    search = Search::new(State::default(), empty_previous_table());
                }
                "position"   => {
                    if let Some(state) = State::from_tokens(tokens) {
                        debug!("\n{}", state.board());
                        let prev_table = search.table();
                        search = Search::new(state, prev_table);
                    } else {
                        error!("Couldn't parse '{}' as position", line);
                    }
                },
                "stop"       => search = search.stop_and_print(),
                "quit"       => return,
                "go"         => search = search.go(tokens, &sender),
                "eval"       => search.print_eval(),
                "movelist"   => search.print_move_list(),
                "sizelist"   => print_size_list(),
                "info"       => search.print_info(),
                "features"   => search.print_features(),
                "bench"      => search.bench(),
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
    println!("option name CPuct type string default 2.15");
    println!("option name CPuctBase type spin min 1 max 65536 default 18368");
    println!("option name CPuctFactor type string default 2.82");
    println!("option name MateScore type string default 1.1");

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
            "cpuctbase" => self.set_option(set_cpuct_base),
            "cpuctfactor" => self.set_option(set_cpuct_factor),
            "matescore" => self.set_option(set_mate_score),
            "policysoftmaxtemp" => self.set_option(set_policy_softmax_temp),
            "policygoodcapturefactor" => self.set_option(set_policy_good_capture_factor),
            "policybadcapturefactor" => self.set_option(set_policy_bad_capture_factor),
            _ => warn!("Badly formatted or unknown option"),
        }
    }

    fn set_option<F, T>(&self, f: F)
    where
        F: FnOnce(T) -> (),
        T: FromStr,
    {
        if let Some(v) = self.value() {
            if let Ok(t) = v.parse() {
                f(t);
            }
        }
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
