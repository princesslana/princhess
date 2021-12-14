extern crate memmap;
extern crate pgn_reader;
extern crate rand;

use self::memmap::Mmap;
use self::pgn_reader::{BufferedReader, Outcome, RawHeader, SanPlus, Skip, Visitor};
use self::rand::rngs::SmallRng;
use self::rand::{Rng, SeedableRng};

use chess;
use features::{featurize, GameResult};
use mcts::GameState;
use policy_features;
use policy_features::NUM_POLICY_FEATURES;
use search::to_uci;
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
const VALIDATION_RATIO: f32 = 0.1;

struct ValueDataGenerator {
    out_file: Option<BufWriter<File>>,
    state: StateBuilder,
    skip: bool,
    rows_written: usize,
    rng: SmallRng,
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
            if i >= 8 && self.rng.gen_range(0., 1.) < freq {
                let mut f = featurize(&state);
                self.rows_written += 1;
                if let Some(out_file) = self.out_file.as_mut() {
                    let crnt_result = if state.board().side_to_move() == chess::Color::White {
                        game_result
                    } else {
                        game_result.flip()
                    };
                    f.write_libsvm(out_file, crnt_result as i16)
                }
            }
            state.make_move(&m);
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {}
}

fn write_policy_feature_names() {
    let mut out_file = File::create("policy_train_data_features.txt").expect("create");
    for i in 0..NUM_POLICY_FEATURES {
        write!(out_file, "{}\n", policy_features::name_feature(i)).unwrap();
    }
}

fn run_value_gen(in_path: &str, out_file: Option<BufWriter<File>>) -> ValueDataGenerator {
    let mut generator = ValueDataGenerator {
        out_file,
        state: StateBuilder::default(),
        skip: true,
        rows_written: 0,
        rng: SeedableRng::seed_from_u64(42),
    };

    let file = File::open(in_path).expect("fopen");
    let pgn = unsafe { Mmap::map(&file).expect("mmap") };
    BufferedReader::new(&pgn[..])
        .read_all(&mut generator)
        .unwrap();

    generator
}

pub fn train_value(in_path: &str, out_path: &str) {
    let out_file = BufWriter::new(File::create(out_path).expect("create"));
    run_value_gen(in_path, Some(out_file));
}

pub fn train(in_path: &str, out_path: &str, policy: bool) {
    write_policy_feature_names();
    if policy {
        train_policy(in_path, out_path);
    } else {
        train_value(in_path, out_path);
    }
}

pub fn train_policy(in_path: &str, out_path: &str) {
    let from_path = format!("policy_from_sq.libsvm");
    let to_path = format!("policy_to_sq.libsvm");

    let from_file = BufWriter::new(File::create(from_path).expect("create"));
    let to_file = BufWriter::new(File::create(to_path).expect("create"));
    let mut generator = PolicyDataGenerator {
        from_file,
        to_file,
        state: StateBuilder::default(),
        skip: true,
        rng: SeedableRng::seed_from_u64(42),
    };
    let file = File::open(in_path).expect("fopen");
    let pgn = unsafe { Mmap::map(&file).expect("mmap") };
    BufferedReader::new(&pgn[..])
        .read_all(&mut generator)
        .unwrap();
}

struct PolicyDataGenerator {
    from_file: BufWriter<File>,
    to_file: BufWriter<File>,
    state: StateBuilder,
    skip: bool,
    rng: SmallRng,
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
        for (i, m) in moves.into_iter().enumerate() {
            if i >= 8 && self.rng.gen_range(0., 1.) < freq {
                let mut f = policy_features::featurize(&state);
                let from_v = state.square_to_index(&m.get_source());
                let to_v = state.square_to_index(&m.get_dest());

                f.write_libsvm(&mut self.from_file, from_v as i16);
                f.write_libsvm(&mut self.to_file, to_v as i16);
            }
            state.make_move(&m);
        }
    }
}
