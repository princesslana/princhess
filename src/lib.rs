#![warn(clippy::all, clippy::pedantic, clippy::cargo)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

extern crate arc_swap;
extern crate arrayvec;
extern crate fastapprox;
extern crate memmap;
extern crate nohash_hasher;
extern crate scc;

mod arena;
mod mem;
mod nets;
mod search_tree;
mod tree_policy;

pub mod chess;
pub mod evaluation;
pub mod math;
pub mod options;
pub mod policy;
pub mod search;
pub mod state;
pub mod tablebase;
pub mod train;
pub mod transposition_table;
pub mod uci;
pub mod value;
