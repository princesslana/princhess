extern crate memmap;
extern crate pgn_reader;
extern crate rand;

use self::memmap::Mmap;
use self::pgn_reader::{BufferedReader, Outcome, RawHeader, SanPlus, Skip, Visitor};
use self::rand::{Rng, SeedableRng, XorShiftRng};

use chess;
use features::{featurize, name_feature, GameResult, NUM_DENSE_FEATURES, NUM_FEATURES};
use mcts::GameState;
use policy_features;
use policy_features::NUM_POLICY_FEATURES;
use shakmaty;
use state::StateBuilder;

use std;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::str;

const NUM_ROWS: usize = std::usize::MAX;
const MIN_ELO: i32 = 1700;
const MIN_ELO_POLICY: i32 = 2200;
const NUM_SAMPLES: usize = 4;

struct ValueDataGenerator {
    out_file: Option<BufWriter<File>>,
    state: StateBuilder,
    skip: bool,
    rows_written: usize,
    rng: XorShiftRng,
    freq: [u64; NUM_FEATURES],
    whitelist: [bool; NUM_FEATURES],
}

impl Visitor for ValueDataGenerator {
    type Result = ();

    fn begin_game(&mut self) {
        self.state = StateBuilder::default();
        self.skip = self.rows_written == NUM_ROWS;
    }

    fn san(&mut self, san: SanPlus) {
        if let Ok(m) = san.san.to_move(self.state.chess()) {
            self.state.make_move(m);
        }
    }

    fn end_headers(&mut self) -> Skip {
        Skip(self.skip)
    }

    fn header(&mut self, key: &[u8], value: RawHeader) {
        if key == b"WhiteElo" || key == b"BlackElo" {
            let elo: i32 = value.decode_utf8().unwrap().parse().unwrap();
            if elo < MIN_ELO {
                self.skip = true;
            }
        }
    }

    fn outcome(&mut self, outcome: Option<Outcome>) {
        let game_result = match outcome {
            Some(Outcome::Draw) => GameResult::Draw,
            Some(Outcome::Decisive { winner }) => {
                if winner == shakmaty::Color::White {
                    GameResult::WhiteWin
                } else {
                    GameResult::BlackWin
                }
            }
            None => return,
        };
        let (mut state, moves) = self.state.extract();
        let freq = NUM_SAMPLES as f64 / moves.len() as f64;
        for (i, m) in moves.into_iter().enumerate() {
            if i >= 2 && self.rng.gen_range(0., 1.) < freq {
                let mut f = featurize(&state);
                self.rows_written += 1;
                if let Some(out_file) = self.out_file.as_mut() {
                    let whitelist = &self.whitelist;
                    let crnt_result = if state.board().side_to_move() == chess::Color::White {
                        game_result
                    } else {
                        game_result.flip()
                    };
                    f.write_libsvm(out_file, crnt_result as usize, |x| whitelist[x]);
                }
                f.write_frequency(&mut self.freq);
            }
            state.make_move(&m);
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {}
}

fn write_feature_names() {
    let mut out_file = File::create("train_data_features.txt").expect("create");
    for i in 0..NUM_DENSE_FEATURES {
        write!(out_file, "{}\n", name_feature(i)).unwrap();
    }
}

fn write_policy_feature_names() {
    let mut out_file = File::create("policy_train_data_features.txt").expect("create");
    for i in 0..NUM_POLICY_FEATURES {
        write!(out_file, "{}\n", policy_features::name_feature(i)).unwrap();
    }
}

fn run_value_gen(
    in_path: &str,
    out_file: Option<BufWriter<File>>,
    whitelist: [bool; NUM_FEATURES],
) -> ValueDataGenerator {
    let mut generator = ValueDataGenerator {
        freq: [0; NUM_FEATURES],
        whitelist,
        out_file,
        state: StateBuilder::default(),
        skip: true,
        rows_written: 0,
        rng: SeedableRng::from_seed([1, 2, 3, 4]),
    };

    let file = File::open(in_path).expect("fopen");
    let pgn = unsafe { Mmap::map(&file).expect("mmap") };
    BufferedReader::new(&pgn[..])
        .read_all(&mut generator)
        .unwrap();

    generator
}

pub fn train_value(in_path: &str, out_path: &str) {
    let freq = run_value_gen(in_path, None, [true; NUM_FEATURES]).freq;
    let out_file = BufWriter::new(File::create(out_path).expect("create"));
    let mut whitelist = [false; NUM_FEATURES];
    for i in 0..NUM_FEATURES {
        whitelist[i] = freq[i] >= 500;
    }
    run_value_gen(in_path, Some(out_file), whitelist);
    let mut freq_file = File::create("frequencies.debug.txt").expect("create");
    let mut indices = (0..NUM_FEATURES).map(|x| (freq[x], x)).collect::<Vec<_>>();
    indices.sort_unstable();
    for &(freq, feature) in &indices {
        write!(freq_file, "{} {}\n", feature, freq).unwrap();
    }
    let mut whitelist_file = File::create("feature_whitelist.txt").expect("create");
    for feature in whitelist.iter().take(NUM_FEATURES) {
        write!(whitelist_file, "{}\n", feature).unwrap();
    }
}

pub fn train(in_path: &str, out_path: &str, policy: bool) {
    write_feature_names();
    write_policy_feature_names();
    if policy {
        train_policy(in_path, out_path);
    } else {
        train_value(in_path, out_path);
    }
}

pub fn train_policy(in_path: &str, out_path: &str) {
    let out_path = format!("policy_{}", out_path);

    let out_file = BufWriter::new(File::create(out_path).expect("create"));
    let key_file = BufWriter::new(File::create("policy_key.txt").expect("create"));
    let mut generator = PolicyDataGenerator {
        out_file,
        key_file,
        state: StateBuilder::default(),
        skip: true,
        rng: SeedableRng::from_seed([1, 2, 3, 4]),
    };
    let file = File::open(in_path).expect("fopen");
    let pgn = unsafe { Mmap::map(&file).expect("mmap") };
    BufferedReader::new(&pgn[..])
        .read_all(&mut generator)
        .unwrap();
}

struct PolicyDataGenerator {
    out_file: BufWriter<File>,
    key_file: BufWriter<File>,
    state: StateBuilder,
    skip: bool,
    rng: XorShiftRng,
}

impl Visitor for PolicyDataGenerator {
    type Result = ();

    fn begin_game(&mut self) {
        self.state = StateBuilder::default();
        self.skip = false;
    }

    fn san(&mut self, san: SanPlus) {
        if let Ok(m) = san.san.to_move(self.state.chess()) {
            self.state.make_move(m);
        }
    }

    fn end_headers(&mut self) -> Skip {
        Skip(self.skip)
    }

    fn header(&mut self, key: &[u8], value: RawHeader) {
        if key == b"WhiteElo" || key == b"BlackElo" {
            let elo: i32 = value.decode_utf8().unwrap().parse().unwrap();
            if elo < MIN_ELO_POLICY {
                self.skip = true;
            }
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {
        let (mut state, moves) = self.state.extract();
        let freq = NUM_SAMPLES as f64 / moves.len() as f64;
        for m in moves {
            if self.rng.gen_range(0., 1.) < freq {
                let legals = state.available_moves();
                let legals = legals.as_slice();
                let index = legals.iter().position(|x| m == *x).unwrap();
                writeln!(self.key_file, "{} {}", legals.len(), index).unwrap();
                for opt in legals {
                    policy_features::featurize(&state, opt).write_libsvm(
                        &mut self.out_file,
                        0,
                        |_| true,
                    );
                }
            }
            state.make_move(&m);
        }
    }
}
