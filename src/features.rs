use chess::*;
use nn;
use nn::NN;
use state;
use state::State;
use std::io::Write;
use std::ops;

#[derive(Clone)]
pub struct FeatureVec {
    pub arr: Vec<i8>,
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum GameResult {
    WhiteWin,
    BlackWin,
    Draw,
}

impl GameResult {
    pub fn flip(&self) -> Self {
        match *self {
            GameResult::WhiteWin => GameResult::BlackWin,
            GameResult::BlackWin => GameResult::WhiteWin,
            GameResult::Draw => GameResult::Draw,
        }
    }
}

impl FeatureVec {
    pub fn write_libsvm<W: Write>(&mut self, f: &mut W, label: i64) {
        write!(f, "{}", label).unwrap();
        for (index, value) in self.arr.iter().enumerate() {
            if *value != 0 {
                write!(f, " {}:{}", index + 1, value).unwrap();
            }
        }
        write!(f, "\n").unwrap();
    }
}

impl ops::Sub<FeatureVec> for FeatureVec {
    type Output = FeatureVec;

    fn sub(self, rhs: FeatureVec) -> FeatureVec {
        assert!(self.arr.len() == rhs.arr.len());

        let new_arr = self
            .arr
            .iter()
            .zip(rhs.arr.iter())
            .map(|(l, r)| l - r)
            .collect();

        FeatureVec { arr: new_arr }
    }
}

pub fn featurize(state: &State) -> FeatureVec {
    FeatureVec {
        arr: state.features().iter().map(|v| *v as i8).collect(),
    }
}

pub struct Model;

impl Model {
    pub fn new() -> Self {
        Model
    }
    pub fn predict(&self, state: &State, features: &[f32; state::NUMBER_FEATURES]) -> f32 {
        let mut nn = NN::new_eval();
        nn.set_inputs(features);

        let mut result = nn.get_output(0);

        result = result.tanh();

        if state.board().side_to_move() == Color::Black {
            result = -result;
        }

        result
    }
    pub fn score(&self, state: &State, features: &[f32; state::NUMBER_FEATURES]) -> f32 {
        self.predict(state, features)
    }
}
