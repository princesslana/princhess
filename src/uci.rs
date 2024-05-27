use std::io::stdin;
use std::str::{FromStr, SplitWhitespace};

use crate::options::{
    set_chess960, set_cpuct, set_cpuct_root, set_cvisits_selection, set_hash_size_mb,
    set_num_threads, set_policy_only, set_policy_temperature, set_policy_temperature_root,
};
use crate::search::Search;
use crate::search_tree::print_size_list;
use crate::state::State;
use crate::tablebase::set_tablebase_directory;
use crate::transposition_table::LRTable;

pub type Tokens<'a> = SplitWhitespace<'a>;

const ENGINE_NAME: &str = "Princhess";
const ENGINE_AUTHOR: &str = "Princess Lana";
const VERSION: Option<&'static str> = option_env!("CARGO_PKG_VERSION");

pub fn main() {
    let mut search = Search::new(State::default(), LRTable::empty());

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
                    let option = UciOption::parse(tokens);

                    if let Some(opt) = option {
                        opt.set(&mut search);
                    }
                }
                "ucinewgame" => {
                    search = Search::new(State::default(), LRTable::empty());
                }
                "position" => {
                    if let Some(state) = State::from_tokens(tokens) {
                        let prev_table = search.table();
                        search = Search::new(state, prev_table);
                    } else {
                        println!("info string Couldn't parse '{line}' as position");
                    }
                }
                "quit" => return,
                "go" => search.go(tokens, &mut next_line),
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
    println!("option name Hash type spin min 8 max 65536 default 16");
    println!("option name Threads type spin min 1 max 255 default 1");
    println!("option name SyzygyPath type string default <empty>");
    println!("option name CPuct type string default 1.06");
    println!("option name CPuctRoot type string default 3.17");
    println!("option name CVisitsSelection type string default 0.01");
    println!("option name PolicyTemperature type string default 1.29");
    println!("option name PolicyTemperatureRoot type string default 5.25");
    println!("option name UCI_Chess960 type check default false");
    println!("option name PolicyOnly type check default false");

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

            name = format!("{name} {t}");
        }

        let mut value = None;

        for t in tokens {
            value = match value {
                None => Some(t.to_owned()),
                Some(s) => Some(format!("{s} {t}")),
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

    pub fn set(&self, search: &mut Search) {
        match self.name().as_str() {
            "syzygypath" => {
                if let Some(path) = self.value() {
                    set_tablebase_directory(path);
                }
            }
            "threads" => self.set_option(set_num_threads),
            "hash" => {
                self.set_option(set_hash_size_mb);
                search.reset_table();
            }
            "cpuct" => self.set_option(set_cpuct),
            "cpuctroot" => self.set_option(set_cpuct_root),
            "cvisitsselection" => self.set_option(set_cvisits_selection),
            "policytemperature" => self.set_option(set_policy_temperature),
            "policytemperatureroot" => self.set_option(set_policy_temperature_root),
            "uci_chess960" => self.set_option(set_chess960),
            "policyonly" => self.set_option(set_policy_only),
            _ => println!("info string Badly formatted or unknown option"),
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
