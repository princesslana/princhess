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
const NUM_SAMPLES: usize = 2;
const VALIDATION_RATIO: f32 = 0.1;

struct ValueDataGenerator {
    out_file: Option<BufWriter<File>>,
    state: StateBuilder,
    skip: bool,
    rows_written: usize,
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
        for m in moves.iter() {
            let mut f = featurize(&state);
            self.rows_written += 1;
            if let Some(out_file) = self.out_file.as_mut() {
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
                f.write_libsvm(out_file, v)
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
    let out_path = format!("policy_{}", out_path);

    let out_file = BufWriter::new(File::create(out_path).expect("create"));
    let validation_file = BufWriter::new(File::create("policy_validation").expect("create"));
    let mut generator = PolicyDataGenerator {
        out_file,
        validation_file,
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
    validation_file: BufWriter<File>,
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
        for m in moves {
            if self.rng.gen_range(0., 1.) < freq {
                let legals = state.available_moves();
                let legals = legals.as_slice();
                //let index = legals.iter().position(|x| m == *x).unwrap();

                let chosen_vec = policy_features::featurize(&state, &m);

                for opt in legals {
                    if *opt == m {
                        continue;
                    }

                    let not_chosen_vec = policy_features::featurize(&state, opt);

                    if self.rng.gen_range(0., 1.) < 0.5 {
                        (chosen_vec.clone() - not_chosen_vec).write_libsvm(&mut self.out_file, 1)
                    } else {
                        (not_chosen_vec - chosen_vec.clone()).write_libsvm(&mut self.out_file, -1)
                    }
                }
                if self.rng.gen_range(0., 1.) < VALIDATION_RATIO {
                    let fen = shakmaty::fen::fen(state.shakmaty_board());
                    writeln!(self.validation_file, "{} : {}", to_uci(m), fen).unwrap();
                }
            }
            state.make_move(&m);
        }
    }
}
