#[macro_use]
extern crate log;
extern crate chess;
extern crate crossbeam;
extern crate float_ord;
extern crate memmap;
extern crate pod;
extern crate pretty_env_logger;
extern crate shakmaty;
extern crate smallvec;

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

    pretty_env_logger::init();

    if let Some(ref train_pgn) = options.train_pgn {
        training::train(&train_pgn, &options.train_output_path, options.policy);
    } else {
        info!("Init.");
        uci::main(options.extra.clone());
        info!("Exit.");
    }
}
