use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    emit_fingerprint_vars();

    if env::var("CARGO_FEATURE_FATHOM").is_ok() {
        build_fathom();
        generate_bindings();
    }
}

fn emit_fingerprint_vars() {
    let rustc = Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=PRINCHESS_RUSTC_VERSION={rustc}");

    let git = Command::new("git")
        .args(["describe", "--tags", "--dirty", "--always"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=PRINCHESS_GIT_DESCRIBE={git}");

    let cpu = env::var("PRINCHESS_TARGET_CPU").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=PRINCHESS_PRINCHESS_TARGET_CPU={cpu}");

    for (name, path) in [
        ("VALUE", "src/nets/value.bin"),
        ("MG_POLICY", "src/nets/mg-policy.bin"),
        ("EG_POLICY", "src/nets/eg-policy.bin"),
    ] {
        println!("cargo:rerun-if-changed={path}");
        let md5 = net_md5(Path::new(path));
        println!("cargo:rustc-env=PRINCHESS_NET_MD5_{name}={md5}");
    }
}

fn net_md5(path: &Path) -> String {
    if !path.exists() {
        return "none".to_string();
    }

    Command::new("md5sum")
        .arg(path)
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8(o.stdout)
                .ok()
                .and_then(|s| s.split_whitespace().next().map(str::to_string))
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn build_fathom() {
    let cc = &mut cc::Build::new();

    cc.file("./deps/fathom/src/tbprobe.c");
    cc.include("./deps/fathom/src");
    cc.define("_CRT_SECURE_NO_WARNINGS", None);
    cc.flag("-Wno-sign-compare");

    let target_cpu = env::var("PRINCHESS_TARGET_CPU").unwrap_or("native".to_string());

    cc.flag(format!("-march={target_cpu}"));

    // MSVC doesn't support stdatomic.h, so use clang on Windows
    if env::consts::OS == "windows" {
        cc.compiler("clang");
    }

    cc.compile("fathom");
}

fn generate_bindings() {
    let bindings = bindgen::Builder::default()
        .header("./deps/fathom/src/tbprobe.h")
        .allowlist_function("tb_.*")
        .allowlist_type("TB_.*")
        .allowlist_var("TB_.*")
        .blocklist_item("__.*")
        .blocklist_item("_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("./src/tablebase/bindings.rs")
        .expect("Couldn't write bindings!");
}
