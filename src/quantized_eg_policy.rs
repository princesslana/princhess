use std::path::Path;

use arrayvec::ArrayVec;
use bytemuck::{self, allocation, Pod, Zeroable};

use crate::chess::Piece;
use crate::mem::Align16;

use crate::nets::{self, Accumulator, MoveIndex};
use crate::quantized_policy::QuantizedLinearNetwork;
use crate::state::{self, State};

pub const INPUT_SIZE: usize = state::POLICY_NUMBER_FEATURES;
pub const ATTENTION_SIZE: usize = 16;
pub const CTX_SIZE: usize = 2 * ATTENTION_SIZE;

pub const QA: i32 = 256;
pub const QAA: i32 = QA * QA;

pub type RawLinearWeights = Align16<[[i16; ATTENTION_SIZE]; INPUT_SIZE]>;
pub type RawLinearBias = Align16<[i16; ATTENTION_SIZE]>;

pub type RawCtxWeights = Align16<[[i16; CTX_SIZE]; INPUT_SIZE]>;
pub type RawCtxBias = Align16<[i16; CTX_SIZE]>;

pub type RawSquareWeights = [RawLinearWeights; Square::COUNT];
pub type RawSquareBias = [RawLinearBias; Square::COUNT];

use crate::chess::Square;
pub type QuantizedSquareSubnets =
    QuantizedLinearNetwork<{ Square::COUNT }, INPUT_SIZE, ATTENTION_SIZE>;

type FeatureVector = ArrayVec<usize, 32>;

#[repr(C)]
#[derive(Copy, Clone, Zeroable)]
pub struct QuantizedCtxNetwork {
    weights: [Align16<Accumulator<i16, CTX_SIZE>>; INPUT_SIZE],
    bias: Align16<Accumulator<i16, CTX_SIZE>>,
}

unsafe impl Pod for QuantizedCtxNetwork {}

impl QuantizedCtxNetwork {
    #[must_use]
    pub fn from_raw(weights: &RawCtxWeights, bias: &RawCtxBias) -> Box<Self> {
        let mut result: Box<Self> = allocation::zeroed_box();
        result.weights = *bytemuck::must_cast_ref(weights);
        result.bias = *bytemuck::must_cast_ref(bias);
        result
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuantizedSeeSplitSubnets {
    pub base: QuantizedSquareSubnets,
    pub good_see: QuantizedSquareSubnets,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct QuantizedEgPolicyNetwork {
    pub ctx: QuantizedCtxNetwork,
    pub pawn: QuantizedSeeSplitSubnets,
    pub knight: QuantizedSeeSplitSubnets,
    pub bishop: QuantizedSeeSplitSubnets,
    pub rook: QuantizedSeeSplitSubnets,
    pub queen: QuantizedSeeSplitSubnets,
    pub king: QuantizedSquareSubnets,
}

impl QuantizedEgPolicyNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        allocation::zeroed_box()
    }

    pub fn save_to_bin(&self, dir: &Path, name: &str) {
        nets::save_to_bin(dir, name, self);
    }

    fn piece_base_subnets(&self, piece: Piece) -> &QuantizedSquareSubnets {
        match piece {
            Piece::PAWN => &self.pawn.base,
            Piece::KNIGHT => &self.knight.base,
            Piece::BISHOP => &self.bishop.base,
            Piece::ROOK => &self.rook.base,
            Piece::QUEEN => &self.queen.base,
            Piece::KING => &self.king,
            _ => unreachable!(),
        }
    }

    fn piece_to_subnets(&self, piece: Piece, good_see: bool) -> &QuantizedSquareSubnets {
        match (piece, good_see) {
            (_, false) | (Piece::KING, _) => self.piece_base_subnets(piece),
            (Piece::PAWN, true) => &self.pawn.good_see,
            (Piece::KNIGHT, true) => &self.knight.good_see,
            (Piece::BISHOP, true) => &self.bishop.good_see,
            (Piece::ROOK, true) => &self.rook.good_see,
            (Piece::QUEEN, true) => &self.queen.good_see,
            _ => unreachable!(),
        }
    }

    pub fn get_all<I: Iterator<Item = MoveIndex>>(
        &self,
        state: &State,
        move_idxes: I,
        out: &mut [f32],
    ) {
        let mut features = FeatureVector::new();
        let mut ctx = *self.ctx.bias;

        state.policy_features_map(|f| {
            features.push(f);
            unsafe { ctx.set(self.ctx.weights.get_unchecked(f)); }
        });

        let [ctx_to, ctx_from]: &[Accumulator<i16, ATTENTION_SIZE>; 2] =
            bytemuck::cast_ref(&ctx);

        for (i, move_idx) in move_idxes.enumerate() {
            let from_sq = move_idx.from_sq();
            let to_sq = move_idx.to_sq_for_piece_subnet();

            let from_piece = self.piece_base_subnets(move_idx.piece());
            let to_piece = self.piece_to_subnets(move_idx.piece(), move_idx.good_see());

            let mut from_piece_sq = from_piece.get_bias(from_sq.index());
            let mut to_piece_sq = to_piece.get_bias(to_sq.index());

            for f in &features {
                from_piece.set(from_sq.index(), *f, &mut from_piece_sq);
                to_piece.set(to_sq.index(), *f, &mut to_piece_sq);
            }

            out[i] = ctx_to.hardtanh_dot_relu::<QA>(&to_piece_sq)
                + ctx_from.hardtanh_dot_relu::<QA>(&from_piece_sq);
        }
    }
}
