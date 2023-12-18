#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

#[macro_use]
extern crate log;
extern crate arc_swap;
extern crate arrayvec;
extern crate dashmap;
extern crate fastapprox;
extern crate memmap;
extern crate nohash_hasher;
extern crate once_cell;
extern crate pretty_env_logger;
extern crate rand;
extern crate shakmaty;

mod arena;
mod args;
mod chess;
mod evaluation;
mod math;
mod options;
mod search;
mod search_tree;
mod state;
mod tablebase;
mod training;
mod transposition_table;
mod tree_policy;
mod uci;

fn main() {
    args::init();
    let options = args::options();

    pretty_env_logger::init();

    if let Some(ref syzygy_path) = options.syzygy_path {
        tablebase::set_tablebase_directory(syzygy_path);
    }

    if let Some(ref train_pgn) = options.train_pgn {
        training::train(train_pgn, &options.train_output_path);
    } else {
        info!("Init.");
        uci::main();
        info!("Exit.");
    }
}
