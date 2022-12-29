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
        writeln!(f).unwrap();
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
