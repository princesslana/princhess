extern crate slurp;
use slurp::*;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    build_policy();
}

fn build_policy() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let magic_path = Path::new(&out_dir).join("policy_feature_const.rs");
    let mut f = File::create(&magic_path).unwrap();
    let offset = "NUM_ENCODED";
    let names = write_feature_names("policy_feature_list.txt", &mut f, offset);
    let num_names = names.len();
    writeln!(
        f,
        "pub const NUM_POLICY_FEATURES: usize = {} + {};",
        offset, num_names
    )
    .unwrap();
    write!(f, "#[allow(dead_code)] ").unwrap();
    writeln!(f, "const INDEX_NAMES: [&str; {}] = [", num_names).unwrap();
    for x in &names {
        writeln!(f, "    \"{}\",", x).unwrap();
    }
    writeln!(f, "];").unwrap();
    writeln!(
        f,
        "const NUM_MODEL_FEATURES: usize = {};",
        read_all_lines("policy_model").unwrap().len()
    )
    .unwrap();
    writeln!(
        f,
        "#[allow(clippy::excessive_precision)] const COEF: [f32; NUM_MODEL_FEATURES] = {};",
        read_all_to_string("policy_model").unwrap()
    )
    .unwrap();
}

fn write_feature_names(from: &str, f: &mut File, offset: &str) -> Vec<String> {
    let names = read_all_to_string(from).unwrap();
    let names: Vec<String> = names.split_whitespace().map(|x| x.into()).collect();
    for (i, x) in names.iter().enumerate() {
        if exempt(x) {
            write!(f, "#[allow(dead_code)] ").unwrap();
        }
        if i == 0 {
            write!(f, "#[allow(clippy::identity_op)]").unwrap();
        }
        writeln!(f, "const {}: usize = {} + {};", x, offset, i).unwrap();
    }
    names
}

fn exempt(name: &str) -> bool {
    if name.contains("PAWN_TO_RANK") || name.contains("_TO_") || name.contains("ISOLATED_PAWN") {
        true
    } else if name.contains("NUM") {
        false
    } else {
        name.contains("KNIGHT")
            || name.contains("BISHOP")
            || name.contains("ROOK")
            || name.contains("QUEEN")
            || name.contains("KING")
    }
}
