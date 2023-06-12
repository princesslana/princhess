extern crate memmap;
extern crate pgn_reader;
extern crate rand;

use memmap::Mmap;
use pgn_reader::{BufferedReader, Outcome, SanPlus, Skip, Visitor};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use shakmaty::{self, Color};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::str;

use crate::state::{self, Builder as StateBuilder};
use crate::mcts::Mcts;
use crate::transposition_table::TranspositionTable;

const NUM_SAMPLES: f64 = 16.;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

impl GameResult {
    pub fn flip(self) -> Self {
        match self {
            GameResult::WhiteWin => GameResult::BlackWin,
            GameResult::BlackWin => GameResult::WhiteWin,
            GameResult::Draw => GameResult::Draw,
        }
    }
}

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
        let freq = NUM_SAMPLES / moves.len() as f64;
        for (i, made) in moves.into_iter().enumerate() {
            if i >= 8 && self.rng.gen_range(0., 1.) < freq {
                self.rows_written += 1;

                if self.rows_written % 100000 == 0 {
                    println!("{} rows written", self.rows_written);
                }

                let mcts = Mcts::new(state.clone(), TranspositionTable::empty(), TranspositionTable::empty());

                mcts.playout_sync_n(1000);

                let eval = mcts.eval();

                let crnt_result = if state.side_to_move() == Color::White {
                    game_result
                } else {
                    game_result.flip()
                };
                let wdl = match crnt_result {
                    GameResult::WhiteWin => 1,
                    GameResult::BlackWin => -1,
                    GameResult::Draw => 0,
                };

                let moves = state.available_moves();

                let mut board_features = [0i8; state::NUMBER_FEATURES];
                let mut move_features = [0i8; state::NUMBER_MOVE_IDX];

                state.features_map(|idx| board_features[idx] = 1);

                for m in moves.as_slice() {
                    move_features[state.move_to_index(m)] = 2;
                }

                move_features[state.move_to_index(&made)] = 1;

                let mut f_vec = Vec::with_capacity(1 + state::NUMBER_MOVE_IDX + state::NUMBER_FEATURES);
                f_vec.push(wdl);
                f_vec.extend_from_slice(&move_features);
                f_vec.extend_from_slice(&board_features);

                /*
                let f_vec = [wdl].iter().chain(move_features)
                    .iter()
                    .chain(board_features.iter())
                    .copied()
                    .collect::<Vec<_>>();
                    */

//                let f_vec = vec![[wdl], move_features, board_features].iter().flat_map(|i| i.iter()).collect();

                write_libsvm(&f_vec, &mut self.out_file, eval);
            }
            state.make_move(&made);
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // stay in the mainline
    }

    fn end_game(&mut self) -> Self::Result {}
}

pub fn write_libsvm<W: Write>(features: &[i8], f: &mut W, label: f32) {
    write!(f, "{label}").unwrap();
    for (index, value) in features.iter().enumerate() {
        if *value != 0 {
            write!(f, " {}:{}", index + 1, value).unwrap();
        }
    }
    writeln!(f).unwrap();
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

pub fn train(in_path: &str, out_path: &str) {
    let out_file = BufWriter::new(File::create(out_path).expect("create"));
    run_value_gen(in_path, out_file);
}
