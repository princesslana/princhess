use std::io::{stdin, BufRead};
use search::Search;
use state::State;
use std::str::SplitWhitespace;
use std::sync::mpsc::{SendError, channel};
use std::thread;

pub type Tokens<'a> = SplitWhitespace<'a>;

pub const TIMEUP: &'static str = "timeup";
const ENGINE_NAME: &'static str = "Sashimi";
const ENGINE_AUTHOR: &'static str = "Jacob Jackson";
const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

pub fn main(commands: Vec<String>) {
    let mut search = Search::new(State::default());
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
                sender.send(line.unwrap_or("".into()))?;
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
                "setoption"  => (),
                "ucinewgame" => position_num += 1,
                "position"   => {
                    position_num += 1;
                    if let Some(state) = State::from_tokens(tokens) {
                        debug!("\n{}", state.board());
                        search = Search::new(state);
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
                _ => error!("Unknown command: {} (this engine uses a reduced set of commands from the UCI protocol)", first_word)
            }
        }
    }
}

pub fn uci() {
    println!("id name {} {}", ENGINE_NAME, VERSION.unwrap_or("unknown"));
    println!("id author {}", ENGINE_AUTHOR);
    println!("uciok");
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
