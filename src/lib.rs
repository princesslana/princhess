#![warn(clippy::all, clippy::pedantic, clippy::cargo)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

mod arena;
mod graph;
mod mcts;
mod mem;
mod nets;
mod subnets;
mod threadpool;

pub mod chess;
pub mod engine;
pub mod evaluation;
pub mod math;
pub mod options;
pub mod policy;
pub mod state;
pub mod tablebase;
pub mod time_management;
pub mod train;
pub mod transposition_table;
pub mod uci;
pub mod value;
