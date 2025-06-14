#![warn(clippy::all, clippy::pedantic, clippy::cargo)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod chess;
pub mod engine;
pub mod evaluation;
pub mod math;
pub mod nets;
pub mod options;
pub mod quantized_policy;
pub mod quantized_value;
pub mod state;
pub mod tablebase;
pub mod time_management;
pub mod train;
pub mod transposition_table;
pub mod uci;

mod arena;
mod graph;
mod mcts;
mod mem;
mod threadpool;
