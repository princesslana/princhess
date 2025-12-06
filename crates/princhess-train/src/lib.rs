#![warn(clippy::all, clippy::pedantic, clippy::cargo)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::multiple_crate_versions)]

pub mod analysis_utils;
pub mod data;
pub mod neural;
pub mod policy;
pub mod value;

mod nets;
