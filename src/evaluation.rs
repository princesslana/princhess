use chess::*;
use features::Model;
use mcts::Evaluator;
use search::{GooseMCTS, SCALE};
use shakmaty_syzygy::Wdl;
use state::{Outcome, Player, State};
use tablebase::probe_tablebase_wdl;

const MATE_FACTOR: f32 = 1.1;
const MATE_SCORE: i64 = (MATE_FACTOR * SCALE) as i64;

pub struct GooseEval {
    model: Model,
}

impl GooseEval {
    pub fn new(model: Model) -> Self {
        Self { model }
    }
}

impl Evaluator<GooseMCTS> for GooseEval {
    fn evaluate_new_state(&self, state: &State) -> (i64, bool) {
        let mut tb_hit = false;
        let state_evaluation = match state.outcome() {
            Outcome::Draw => 0,
            Outcome::WhiteWin => MATE_SCORE,
            Outcome::BlackWin => -MATE_SCORE,
            Outcome::Ongoing => {
                if let Some(wdl) = probe_tablebase_wdl(state.shakmaty_board()) {
                    tb_hit = true;
                    let win_score = SCALE as i64;
                    match (wdl, state.board().side_to_move()) {
                        (Wdl::Win, Color::White) => win_score,
                        (Wdl::Loss, Color::White) => -win_score,
                        (Wdl::Win, Color::Black) => -win_score,
                        (Wdl::Loss, Color::Black) => win_score,
                        _ => 0,
                    }
                } else {
                    (self.model.score(state) * SCALE as f32) as i64
                }
            }
        };
        (state_evaluation, tb_hit)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, player: &Player) -> i64 {
        match *player {
            Color::White => *evaln,
            Color::Black => -*evaln,
        }
    }
}
