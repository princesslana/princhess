#![warn(clippy::all, clippy::pedantic, clippy::cargo)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
#![allow(clippy::multiple_crate_versions)]

pub mod analysis_utils;
pub mod args;
pub mod data;
pub mod neural;
pub mod policy;
pub mod system;
pub mod tui;
pub mod value;

mod nets;
