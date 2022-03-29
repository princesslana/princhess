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
use shakmaty;
use state::StateBuilder;

use std;
use std::fs::File;
use std::io::BufWriter;
use std::str;

const NUM_ROWS: usize = std::usize::MAX;
const MIN_ELO: i32 = 1700;
const NUM_SAMPLES: usize = 8;

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
                    let v = match crnt_result {
                        GameResult::WhiteWin => 1,
                        GameResult::BlackWin => -1,
                        GameResult::Draw => 0,
                    };
                    f.write_libsvm(out_file, v);
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

pub fn train(in_path: &str, out_path: &str) {
    train_value(in_path, out_path);
}
