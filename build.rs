use std::env;

fn main() {
    if env::var("CARGO_FEATURE_FATHOM").is_ok() {
        build_fathom();
        generate_bindings();
    }
}

fn build_fathom() {
    let cc = &mut cc::Build::new();

    cc.file("./deps/fathom/src/tbprobe.c");
    cc.include("./deps/fathom/src");
    cc.define("_CRT_SECURE_NO_WARNINGS", None);

    let target_cpu = env::var("TARGET_CPU").unwrap_or("native".to_string());

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
