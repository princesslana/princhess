#![warn(clippy::all, clippy::pedantic)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

#[macro_use]
extern crate log;
extern crate arc_swap;
extern crate arrayvec;
extern crate crossbeam;
extern crate dashmap;
extern crate fastapprox;
extern crate memmap;
extern crate nohash_hasher;
extern crate once_cell;
extern crate pretty_env_logger;
extern crate rand;
extern crate shakmaty;
extern crate shakmaty_syzygy;

mod arena;
mod math;
mod mcts;
mod options;
mod search_tree;
mod tablebase;
mod transposition_table;
mod tree_policy;

mod args;
mod evaluation;
mod search;
mod state;
mod training;
mod uci;

fn main() {
    args::init();
    let options = args::options();

    pretty_env_logger::init();

    if let Some(ref train_pgn) = options.train_pgn {
        training::train(train_pgn, &options.train_output_path);
    } else {
        info!("Init.");
        uci::main(options.extra.clone());
        info!("Exit.");
    }
}
