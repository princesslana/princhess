extern crate argparse;
use self::argparse::{ArgumentParser, StoreOption};

#[derive(Default)]
pub struct Options {
    pub syzygy_path: Option<String>,
}

pub fn init() {
    let mut options = Options::default();
    {
        let mut ap = ArgumentParser::new();
        ap.refer(&mut options.syzygy_path).add_option(
            &["-s", "--syzygy"],
            StoreOption,
            "path to syzygy tablebases",
        );
        ap.parse_args_or_exit();
    }
    unsafe {
        G_OPTIONS = Some(options);
    }
}

pub fn options() -> &'static Options {
    unsafe { G_OPTIONS.as_ref().unwrap() }
}

static mut G_OPTIONS: Option<Options> = None;
