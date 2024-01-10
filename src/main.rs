#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

#[macro_use]
extern crate arc_swap;
extern crate arrayvec;
extern crate dashmap;
extern crate fastapprox;
extern crate memmap;
extern crate nohash_hasher;
extern crate once_cell;

mod arena;
mod chess;
mod evaluation;
mod math;
mod options;
mod search;
mod search_tree;
mod state;
mod tablebase;
mod transposition_table;
mod tree_policy;
mod uci;

fn main() {
    uci::main();
}
