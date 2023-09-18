extern crate memmap;
extern crate pgn_reader;
extern crate rand;

use memmap::Mmap;
use pgn_reader::{BufferedReader, Outcome, RawHeader, SanPlus, Skip, Visitor};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use shakmaty::{self, Chess, Color, Setup};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::str;

use crate::mcts::Mcts;
use crate::options::set_chess960;
use crate::state::{self, Builder as StateBuilder};
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
    log_prefix: String,
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

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        if key == b"FEN" {
            let fen = str::from_utf8(value.as_bytes()).unwrap();
            self.state = StateBuilder::from_fen(fen).unwrap();
        }
    }

    fn san(&mut self, san: SanPlus) {
        if self.skip {
            return;
        }

        if let Ok(m) = san.san.to_move(self.state.chess()) {
            self.state.make_move(m);
        } else {
            self.skip = true;
            println!("{}: invalid move: {}", self.log_prefix, san.san);
        }
    }

    fn outcome(&mut self, outcome: Option<Outcome>) {
        if self.skip {
            return;
        }

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
        let skip_plies = if Chess::default().board().pawns() == state.board().board().pawns() {
            8
        } else {
            1
        };

        for (ply, made) in moves.into_iter().enumerate() {
            if ply >= skip_plies && self.rng.gen_range(0., 1.) < freq {
                let moves = state.available_moves();

                if moves.is_empty() {
                    continue;
                }

                self.rows_written += 1;

                if self.rows_written % 100_000 == 0 {
                    println!("{}: {} rows written", self.log_prefix, self.rows_written);
                }

                let mcts = Mcts::new(
                    state.clone(),
                    TranspositionTable::empty(),
                    TranspositionTable::zero(),
                );

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

                let mut board_features = [0i8; state::NUMBER_FEATURES];
                let mut move_features = [0i8; state::NUMBER_MOVE_IDX];

                state.features_map(|idx| board_features[idx] = 1);

                for m in moves.as_slice() {
                    move_features[state.move_to_index(m)] = 2;
                }

                move_features[state.move_to_index(&made)] = 1;

                let mut f_vec =
                    Vec::with_capacity(1 + state::NUMBER_MOVE_IDX + state::NUMBER_FEATURES);
                f_vec.push(wdl);
                f_vec.extend_from_slice(&move_features);
                f_vec.extend_from_slice(&board_features);

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
        log_prefix: in_path.to_string(),
        out_file,
        state: StateBuilder::default(),
        skip: false,
        rows_written: 0,
        rng: SeedableRng::seed_from_u64(42),
    };

    if in_path.contains("FRC") {
        set_chess960(true);
    }

    let file = File::open(in_path).expect("fopen");
    let pgn = unsafe { Mmap::map(&file).expect("mmap") };
    BufferedReader::new(&pgn[..])
        .read_all(&mut generator)
        .unwrap();

    generator
}

pub fn train(in_path: &str, out_path: &str) {
    let out_file = BufWriter::new(File::create(out_path).expect("create"));
    println!("Featurizing {in_path}...");
    run_value_gen(in_path, out_file);
}
