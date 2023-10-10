use std::env;

fn main() {
    build_fathom();
    generate_bindings();
}

fn build_fathom() {
    let cc = &mut cc::Build::new();

    cc.file("./deps/fathom/src/tbprobe.c");
    cc.include("./deps/fathom/src");
    cc.define("_CRT_SECURE_NO_WARNINGS", None);

    // MSVC doesn't support stdatomic.h, so use clang on Windows
    if env::consts::OS == "windows" {
        cc.compiler("clang");
    }

    cc.compile("fathom");
}

fn generate_bindings() {
    let bindings = bindgen::Builder::default()
        .header("./deps/fathom/src/tbprobe.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file("./src/tablebase/bindings.rs")
        .expect("Couldn't write bindings!");
}
