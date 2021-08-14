#[macro_use]
extern crate log;
extern crate chess;
extern crate crossbeam;
extern crate float_ord;
extern crate memmap;
extern crate pod;
extern crate shakmaty;
extern crate simplelog;
extern crate smallvec;

use simplelog::{CombinedLogger, Config, LevelFilter, TermLogger, WriteLogger};
use std::fs::OpenOptions;

mod arena;
mod atomics;
mod mcts;
mod search_tree;
mod transposition_table;
mod tree_policy;

mod args;
mod evaluation;
mod features;
mod features_common;
mod policy_features;
mod search;
mod state;
mod training;
mod uci;

fn main() {
    args::init();
    let options = args::options();
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&options.log_file_path)
        .unwrap();
    CombinedLogger::init(vec![WriteLogger::new(
        LevelFilter::Debug,
        Config::default(),
        log_file,
    )])
    .unwrap();
    if let Some(ref train_pgn) = options.train_pgn {
        training::train(&train_pgn, &options.train_output_path, options.policy);
    } else {
        info!("Init.");
        uci::main(options.extra.clone());
        info!("Exit.");
    }
}
