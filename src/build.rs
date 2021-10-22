extern crate slurp;
use slurp::*;

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    build_value();
    build_policy();
}

fn build_value() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let magic_path = Path::new(&out_dir).join("feature_const.rs");
    let mut f = File::create(&magic_path).unwrap();
    let names = read_all_to_string("feature_list.txt").unwrap();
    let mut nxt = 0;
    let phases = ["Midgame", "Endgame"];
    writeln!(f, "#[repr(u8)]").unwrap();
    writeln!(f, "#[derive(Copy, Clone, Debug, Eq, PartialEq)]").unwrap();
    writeln!(f, "enum Phase {{").unwrap();
    for (i, p) in phases.iter().enumerate() {
        writeln!(f, "    {} = {},", p, i).unwrap()
    }
    writeln!(f, "}}").unwrap();
    let colours = 2;
    let names = expand_macros(
        names
            .split_whitespace()
            .map(|x| x.as_bytes().to_vec())
            .collect(),
    );
    for x in &names {
        if exempt(x) {
            write!(f, "#[allow(dead_code)]\n").unwrap();
        }
        write!(f, "const {}: usize = {};\n", x, nxt).unwrap();
        nxt += 1
    }
    writeln!(f, "const NUM_COLORS: usize = 2;").unwrap();
    writeln!(f, "const NUM_NAMES: usize = {};", nxt).unwrap();
    writeln!(f, "const NUM_PHASES: usize = {};", phases.len()).unwrap();
    let tot = nxt * phases.len() * colours;
    writeln!(f, "pub const NUM_DENSE_FEATURES: usize = {};", tot).unwrap();
    writeln!(f, "const INDEX_NAMES: [&str; {}] = [", nxt).unwrap();
    for x in &names {
        writeln!(f, "    \"{}\",", x).unwrap();
    }
    writeln!(f, "];").unwrap();
    writeln!(
        f,
        "const NUM_MODEL_FEATURES: usize = {};",
        read_all_lines("model").unwrap().len()
    )
    .unwrap();
    writeln!(
        f,
        "#[allow(clippy::excessive_precision)] const COEF: [[f32; NUM_OUTCOMES]; NUM_MODEL_FEATURES] = {};",
        read_all_to_string("model").unwrap()
    )
    .unwrap();
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

fn expand_macros(mut x: Vec<Vec<u8>>) -> Vec<String> {
    let mut result = Vec::new();
    while !x.is_empty() {
        let mut buf = Vec::new();
        {
            let s = &x[0];
            if let Some(a) = s.iter().position(|c| *c == b'[') {
                let b = s.iter().position(|c| *c == b']').unwrap();
                for term in terms(s[(a + 1)..b].to_vec()) {
                    let mut crnt = Vec::new();
                    crnt.extend(s[..a].to_vec());
                    crnt.extend(term);
                    crnt.extend(s[(b + 1)..].to_vec());
                    buf.push(crnt);
                }
            }
        }
        if buf.is_empty() {
            result.push(x.remove(0));
        } else {
            x.remove(0);
        }
        buf.append(&mut x);
        x = buf;
    }
    result
        .into_iter()
        .map(|x| String::from_utf8(x).unwrap())
        .collect()
}

fn terms(x: Vec<u8>) -> Vec<Vec<u8>> {
    if x == b"piece" {
        vec![
            b"PAWN".to_vec(),
            b"KNIGHT".to_vec(),
            b"BISHOP".to_vec(),
            b"ROOK".to_vec(),
            b"QUEEN".to_vec(),
            b"KING".to_vec(),
        ]
    } else {
        String::from_utf8(x)
            .unwrap()
            .split('|')
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .into_iter()
            .map(|s| s.as_bytes().to_vec())
            .collect()
    }
}
