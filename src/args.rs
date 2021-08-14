extern crate argparse;
use self::argparse::*;

pub struct Options {
    pub log_file_path: String,
    pub train_pgn: Option<String>,
    pub train_output_path: String,
    pub policy: bool,
    pub extra: Vec<String>,
    pub num_threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        extern crate num_cpus;
        let num_threads = num_cpus::get();
        Options {
            log_file_path: "sashimi.log".into(),
            train_pgn: None,
            train_output_path: "train_data.libsvm".into(),
            policy: false,
            extra: Vec::new(),
            num_threads,
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
        ap.refer(&mut options.policy).add_option(
            &["-p", "--policy"],
            StoreTrue,
            "output policy data instead of value data",
        );
        ap.refer(&mut options.log_file_path)
            .add_option(&["--log"], Store, "log file path");
        ap.refer(&mut options.num_threads)
            .add_option(&["--threads"], Store, "number of threads");
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
