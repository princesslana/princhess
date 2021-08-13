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
    write!(f, "#[repr(u8)]\n").unwrap();
    write!(f, "#[derive(Copy, Clone, Debug, Eq, PartialEq)]\n").unwrap();
    write!(f, "enum Phase {{\n").unwrap();
    for (i, p) in phases.iter().enumerate() {
        write!(f, "    {} = {},\n", p, i).unwrap()
    }
    write!(f, "}}\n").unwrap();
    let colours = 2;
    let names = expand_macros(
        names.split_whitespace()
            .map(|x| x.as_bytes().to_vec())
            .collect());
    for x in &names {
        if exempt(x) {
            write!(f, "#[allow(dead_code)]\n").unwrap();
        }
        write!(f, "const {}: usize = {};\n", x, nxt).unwrap();
        nxt += 1
    }
    write!(f, "const NUM_COLORS: usize = 2;\n").unwrap();
    write!(f, "const NUM_NAMES: usize = {};\n", nxt).unwrap();
    write!(f, "const NUM_PHASES: usize = {};\n", phases.len()).unwrap();
    let tot = nxt * phases.len() * colours;
    write!(f, "pub const NUM_DENSE_FEATURES: usize = {};\n", tot).unwrap();
    write!(f, "const INDEX_NAMES: [&'static str; {}] = [\n", nxt).unwrap();
    for x in &names {
        write!(f, "    \"{}\",\n", x).unwrap();
    }
    write!(f, "];\n").unwrap();
    write!(f, "const NUM_MODEL_FEATURES: usize = {};\n",
        read_all_lines("model").unwrap().len()).unwrap();
    write!(f, "const COEF: [[f32; NUM_OUTCOMES]; NUM_MODEL_FEATURES] = {};\n",
        read_all_to_string("model").unwrap()).unwrap();
}

fn build_policy() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let magic_path = Path::new(&out_dir).join("policy_feature_const.rs");
    let mut f = File::create(&magic_path).unwrap();
    let offset = "NUM_ENCODED";
    let names = write_feature_names("policy_feature_list.txt", &mut f, offset);
    let num_names = names.len();
    write!(f, "pub const NUM_POLICY_FEATURES: usize = {} + {};\n", offset, num_names).unwrap();
    write!(f, "#[allow(dead_code)] ").unwrap();
    write!(f, "const INDEX_NAMES: [&'static str; {}] = [\n", num_names).unwrap();
    for x in &names {
        write!(f, "    \"{}\",\n", x).unwrap();
    }
    write!(f, "];\n").unwrap();
    write!(f, "const NUM_MODEL_FEATURES: usize = {};\n",
        read_all_lines("policy_model").unwrap().len()).unwrap();
    write!(f, "const COEF: [f32; NUM_MODEL_FEATURES] = {};\n",
        read_all_to_string("policy_model").unwrap()).unwrap();
}

fn write_feature_names(from: &str, f: &mut File, offset: &str) -> Vec<String> {
    let names = read_all_to_string(from).unwrap();
    let names: Vec<String> = names.split_whitespace().map(|x| x.into()).collect();
    for (i, x) in names.iter().enumerate() {
        if exempt(x) {
            write!(f, "#[allow(dead_code)] ").unwrap();
        }
        write!(f, "const {}: usize = {} + {};\n", x, offset, i).unwrap();
    }
    names
}

fn exempt(name: &str) -> bool {
    if name.contains("PAWN_TO_RANK") {
        true
    } else if name.contains("_TO_") {
        true
    } else if name.contains("NUM") {
        false
    } else {
        name.contains("KNIGHT") ||
        name.contains("BISHOP") ||
        name.contains("ROOK") ||
        name.contains("QUEEN") ||
        name.contains("KING")
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
                for term in terms(s[(a+1)..b].to_vec()) {
                    let mut crnt = Vec::new();
                    crnt.extend(s[..a].to_vec());
                    crnt.extend(term);
                    crnt.extend(s[(b+1)..].to_vec());
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
    result.into_iter().map(|x| String::from_utf8(x).unwrap()).collect()
}

fn terms(x: Vec<u8>) -> Vec<Vec<u8>> {
    if x == b"piece" {
        vec![
            b"PAWN".to_vec(),
            b"KNIGHT".to_vec(),
            b"BISHOP".to_vec(),
            b"ROOK".to_vec(),
            b"QUEEN".to_vec(),
            b"KING".to_vec(),]
    } else {
        String::from_utf8(x).unwrap()
            .split("|")
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .into_iter()
            .map(|s| s.as_bytes().to_vec())
            .collect()
    }
}
