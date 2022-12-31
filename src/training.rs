extern crate memmap;
extern crate pgn_reader;
extern crate rand;

use self::memmap::Mmap;
use self::pgn_reader::{BufferedReader, Outcome, RawHeader, SanPlus, Skip, Visitor};
use self::rand::rngs::SmallRng;
use self::rand::{Rng, SeedableRng};

use chess;
use features::{featurize, FeatureVec, GameResult};
use shakmaty;
use state::StateBuilder;

use std;
use std::fs::File;
use std::io::BufWriter;
use std::str;

const NUM_ROWS: usize = std::usize::MAX;
const MIN_ELO: i32 = 1700;
const MIN_ELO_POLICY: i32 = 1700;
const NUM_SAMPLES: usize = 16;
const NUM_SAMPLES_POLICY: usize = 16;

struct ValueDataGenerator {
    out_file: BufWriter<File>,
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
                let crnt_result = if state.board().side_to_move() == chess::Color::White {
                    game_result
                } else {
                    game_result.flip()
                };
                let v = match crnt_result {
                    GameResult::WhiteWin => 1,
                    GameResult::BlackWin => -1,
                    GameResult::Draw => 0,
                };
                f.write_libsvm(&mut self.out_file, v);
            }
            state.make_move(&m);
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {}
}

fn run_value_gen(in_path: &str, out_file: BufWriter<File>) -> ValueDataGenerator {
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
    run_value_gen(in_path, out_file);
}

pub fn train(in_path: &str, out_path: &str, policy: bool) {
    if policy {
        train_policy(in_path, out_path);
    } else {
        train_value(in_path, out_path);
    }
}

pub fn train_policy(in_path: &str, out_path: &str) {
    let out_path = format!("policy_{}", out_path);

    let out_file = BufWriter::new(File::create(out_path).expect("create"));
    let mut generator = PolicyDataGenerator {
        out_file,
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
    out_file: BufWriter<File>,
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

    fn outcome(&mut self, _: Option<Outcome>) {
        let (mut state, moves) = self.state.extract();
        let freq = NUM_SAMPLES_POLICY as f64 / moves.len() as f64;
        for (i, made) in moves.into_iter().enumerate() {
            if i >= 8 && self.rng.gen_range(0., 1.) < freq {
                let board_features = state.features();
                let moves = state.available_moves();

                let mut move_features = [0f32; 1792];

                for m in moves.as_slice() {
                    move_features[state.move_to_index(m)] = 2.
                }

                move_features[state.move_to_index(&made)] = 1.;

                let mut f_vec = FeatureVec {
                    arr: move_features
                        .iter()
                        .chain(board_features.iter())
                        .map(|v| *v as i8)
                        .collect(),
                };

                f_vec.write_libsvm(&mut self.out_file, 0);
            }
            state.make_move(&made);
        }
    }
    fn end_game(&mut self) -> Self::Result {}
}
