extern crate argparse;
use self::argparse::*;

pub struct Options {
    pub train_pgn: Option<String>,
    pub train_output_path: String,
    pub extra: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            train_pgn: None,
            train_output_path: "train_data.libsvm".into(),
            extra: Vec::new(),
        }
    }
}

pub fn init() {
    let mut options = Options::default();
    {
        let mut ap = ArgumentParser::new();
        ap.refer(&mut options.train_pgn).add_option(
            &["-t", "--train"],
            StoreOption,
            "path to .pgn for training",
        );
        ap.refer(&mut options.train_output_path).add_option(
            &["-o", "--output"],
            Store,
            "train output path",
        );
        ap.refer(&mut options.extra).add_argument(
            "uci_commands",
            Collect,
            "additional arguments are interpreted as UCI commands",
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
