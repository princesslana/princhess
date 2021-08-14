#[macro_use]
extern crate log;
extern crate float_ord;
extern crate smallvec;
extern crate simplelog;
extern crate shakmaty;
extern crate chess;
extern crate crossbeam;
extern crate memmap;
extern crate pod;

use simplelog::{WriteLogger, CombinedLogger, LevelFilter, Config, TermLogger};
use std::fs::OpenOptions;

mod arena;
mod search_tree;
mod atomics;
mod tree_policy;
mod transposition_table;
mod mcts;

mod uci;
mod search;
mod state;
mod evaluation;
mod features;
mod policy_features;
mod features_common;
mod args;
mod training;

fn main() {
    args::init();
    let options = args::options();
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&options.log_file_path)
        .unwrap();
    CombinedLogger::init(vec![
        WriteLogger::new(
            LevelFilter::Debug,
            Config::default(),
            log_file)
        ]).unwrap();
    if let Some(ref train_pgn) = options.train_pgn {
        training::train(&train_pgn, &options.train_output_path, options.policy);
    } else {
        info!("Init.");
        uci::main(options.extra.clone());
        info!("Exit.");
    }
}
