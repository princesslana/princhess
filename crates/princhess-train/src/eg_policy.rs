use std::fmt::{self, Display};
use std::ops::{AddAssign, DivAssign};
use std::ptr;

use arrayvec::ArrayVec;
use bytemuck::{allocation, Zeroable};
use princhess::chess::{Piece, Square};
use princhess::nets::MoveIndex;
use princhess::quantized_eg_policy::{
    QuantizedCtxNetwork, QuantizedEgPolicyNetwork, QuantizedSquareSubnets, RawCtxBias,
    RawCtxWeights, RawSquareBias, RawSquareWeights, ATTENTION_SIZE, CTX_SIZE, INPUT_SIZE, QA,
};
use princhess::state::State;

use crate::data::TrainingPosition;
use crate::nets;
use crate::neural::{
    AdamWOptimizer, FeedForwardNetwork, HardTanh, LRScheduler, LinearNetwork, OutputLayer,
    SparseConnected, SparseConnectedLayers, SparseVector, Vector,
};

type EgCtxNetwork = SparseConnected<HardTanh, INPUT_SIZE, CTX_SIZE>;
type EgLinearNetwork = LinearNetwork<INPUT_SIZE, ATTENTION_SIZE>;

#[derive(Zeroable)]
struct SquareSubnets([EgLinearNetwork; Square::COUNT]);

impl std::ops::Deref for SquareSubnets {
    type Target = [EgLinearNetwork; Square::COUNT];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for SquareSubnets {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AddAssign<&Self> for SquareSubnets {
    fn add_assign(&mut self, rhs: &Self) {
        for (l, r) in self.0.iter_mut().zip(&rhs.0) {
            *l += r;
        }
    }
}

impl DivAssign<f32> for SquareSubnets {
    fn div_assign(&mut self, rhs: f32) {
        for s in &mut self.0 {
            *s /= rhs;
        }
    }
}

impl SquareSubnets {
    fn l1_norm(&self) -> f32 {
        self.0.iter().map(EgLinearNetwork::l1_norm).sum()
    }

    fn adamw<S: LRScheduler>(
        &mut self,
        g: &Self,
        m: &mut Self,
        v: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        for i in 0..Square::COUNT {
            self.0[i].adamw(&g.0[i], &mut m.0[i], &mut v.0[i], optimizer);
        }
    }

    fn randomize(&mut self) {
        for s in &mut self.0 {
            s.randomize();
        }
    }
}

#[must_use]
pub fn is_training_position(state: &State) -> bool {
    let board = state.board();
    let major_pieces_count =
        (board.queens() | board.rooks() | board.bishops() | board.knights()).count();
    major_pieces_count <= 8
}

#[derive(Zeroable)]
struct SeeSplitSubnets {
    base: SquareSubnets,
    good_see: SquareSubnets,
}

impl AddAssign<&Self> for SeeSplitSubnets {
    fn add_assign(&mut self, rhs: &Self) {
        self.base += &rhs.base;
        self.good_see += &rhs.good_see;
    }
}

impl DivAssign<f32> for SeeSplitSubnets {
    fn div_assign(&mut self, rhs: f32) {
        self.base /= rhs;
        self.good_see /= rhs;
    }
}

impl SeeSplitSubnets {
    fn l1_norm(&self) -> f32 {
        self.base.l1_norm() + self.good_see.l1_norm()
    }

    fn adamw<S: LRScheduler>(
        &mut self,
        g: &Self,
        m: &mut Self,
        v: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        self.base
            .adamw(&g.base, &mut m.base, &mut v.base, optimizer);
        self.good_see
            .adamw(&g.good_see, &mut m.good_see, &mut v.good_see, optimizer);
    }

    fn randomize(&mut self) {
        self.base.randomize();
        self.good_see.randomize();
    }
}

struct EgPolicyMoveLayers {
    from_piece_sq: SparseConnectedLayers<ATTENTION_SIZE>,
    to_piece_sq: SparseConnectedLayers<ATTENTION_SIZE>,
}

pub struct EgPolicyForwardCache {
    ctx_layers: SparseConnectedLayers<CTX_SIZE>,
    ctx_to: Vector<ATTENTION_SIZE>,
    ctx_from: Vector<ATTENTION_SIZE>,
    move_layers: ArrayVec<EgPolicyMoveLayers, { TrainingPosition::MAX_MOVES }>,
}

#[allow(clippy::module_name_repetitions)]
#[derive(Zeroable)]
pub struct EgPolicyNetwork {
    ctx: EgCtxNetwork,
    pawn: SeeSplitSubnets,
    knight: SeeSplitSubnets,
    bishop: SeeSplitSubnets,
    rook: SeeSplitSubnets,
    queen: SeeSplitSubnets,
    king: SquareSubnets,
}

impl Display for EgPolicyNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "hardtanh(ctx): [{INPUT_SIZE}->{CTX_SIZE}] * relu({{P/N/B/R/Q: SeeSplit([{}; {}]), K: [{}; {}]}}), to+from",
            self.pawn.base[0],
            Square::COUNT,
            self.king[0],
            Square::COUNT,
        )
    }
}

impl AddAssign<&Self> for EgPolicyNetwork {
    fn add_assign(&mut self, rhs: &Self) {
        self.ctx += &rhs.ctx;
        self.pawn += &rhs.pawn;
        self.knight += &rhs.knight;
        self.bishop += &rhs.bishop;
        self.rook += &rhs.rook;
        self.queen += &rhs.queen;
        self.king += &rhs.king;
    }
}

impl DivAssign<f32> for EgPolicyNetwork {
    fn div_assign(&mut self, rhs: f32) {
        self.ctx /= rhs;
        self.pawn /= rhs;
        self.knight /= rhs;
        self.bishop /= rhs;
        self.rook /= rhs;
        self.queen /= rhs;
        self.king /= rhs;
    }
}

impl EgPolicyNetwork {
    #[must_use]
    pub fn zeroed() -> Box<Self> {
        allocation::zeroed_box()
    }

    pub fn zero_out(&mut self) {
        // SAFETY: EgPolicyNetwork: Zeroable guarantees all-zeros is a valid bit pattern
        unsafe { ptr::write_bytes(ptr::from_mut::<Self>(self), 0, 1) }
    }

    #[must_use]
    pub fn l1_norm(&self) -> f32 {
        self.ctx.l1_norm()
            + self.pawn.l1_norm()
            + self.knight.l1_norm()
            + self.bishop.l1_norm()
            + self.rook.l1_norm()
            + self.queen.l1_norm()
            + self.king.l1_norm()
    }

    #[must_use]
    pub fn random() -> Box<Self> {
        let mut rng = princhess::math::Rng::default();
        let mut network = Self::zeroed();

        network.ctx = *EgCtxNetwork::randomized(&mut rng);
        network.pawn.randomize();
        network.knight.randomize();
        network.bishop.randomize();
        network.rook.randomize();
        network.queen.randomize();
        network.king.randomize();

        network
    }

    fn piece_base_subnets(&self, piece: Piece) -> &SquareSubnets {
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

    fn piece_base_subnets_mut(&mut self, piece: Piece) -> &mut SquareSubnets {
        match piece {
            Piece::PAWN => &mut self.pawn.base,
            Piece::KNIGHT => &mut self.knight.base,
            Piece::BISHOP => &mut self.bishop.base,
            Piece::ROOK => &mut self.rook.base,
            Piece::QUEEN => &mut self.queen.base,
            Piece::KING => &mut self.king,
            _ => unreachable!(),
        }
    }

    fn piece_to_subnets(&self, move_idx: MoveIndex) -> &SquareSubnets {
        match (move_idx.piece(), move_idx.good_see()) {
            (_, false) | (Piece::KING, _) => self.piece_base_subnets(move_idx.piece()),
            (Piece::PAWN, true) => &self.pawn.good_see,
            (Piece::KNIGHT, true) => &self.knight.good_see,
            (Piece::BISHOP, true) => &self.bishop.good_see,
            (Piece::ROOK, true) => &self.rook.good_see,
            (Piece::QUEEN, true) => &self.queen.good_see,
            _ => unreachable!(),
        }
    }

    fn piece_to_subnets_mut(&mut self, move_idx: MoveIndex) -> &mut SquareSubnets {
        match (move_idx.piece(), move_idx.good_see()) {
            (_, false) | (Piece::KING, _) => self.piece_base_subnets_mut(move_idx.piece()),
            (Piece::PAWN, true) => &mut self.pawn.good_see,
            (Piece::KNIGHT, true) => &mut self.knight.good_see,
            (Piece::BISHOP, true) => &mut self.bishop.good_see,
            (Piece::ROOK, true) => &mut self.rook.good_see,
            (Piece::QUEEN, true) => &mut self.queen.good_see,
            _ => unreachable!(),
        }
    }

    pub fn get_all_with_layers(
        &self,
        features: &SparseVector,
        move_idxes: &[MoveIndex],
        out: &mut [f32],
    ) -> EgPolicyForwardCache {
        let ctx_layers = self.ctx.out_with_layers(features);
        let ctx_out = ctx_layers.output_layer();
        let ctx_to: Vector<ATTENTION_SIZE> = ctx_out.slice(0);
        let ctx_from: Vector<ATTENTION_SIZE> = ctx_out.slice(ATTENTION_SIZE);

        let mut move_layers = ArrayVec::new();

        for (i, &move_idx) in move_idxes.iter().enumerate() {
            let from_sq = move_idx.from_sq();
            let to_piece_sq = move_idx.to_sq_for_piece_subnet();

            let from_piece = self.piece_base_subnets(move_idx.piece());
            let to_piece = self.piece_to_subnets(move_idx);

            let from_piece_sq = from_piece[from_sq].output.out_with_layers(features);
            let to_piece_sq = to_piece[to_piece_sq].output.out_with_layers(features);

            out[i] = ctx_to.dot(&to_piece_sq.output_layer())
                + ctx_from.dot(&from_piece_sq.output_layer());

            move_layers.push(EgPolicyMoveLayers {
                from_piece_sq,
                to_piece_sq,
            });
        }

        EgPolicyForwardCache {
            ctx_layers,
            ctx_to,
            ctx_from,
            move_layers,
        }
    }

    pub fn adamw<S: LRScheduler>(
        &mut self,
        g: &Self,
        m: &mut Self,
        v: &mut Self,
        optimizer: &AdamWOptimizer<S>,
    ) {
        self.ctx.adamw(&g.ctx, &mut m.ctx, &mut v.ctx, optimizer);
        self.pawn
            .adamw(&g.pawn, &mut m.pawn, &mut v.pawn, optimizer);
        self.knight
            .adamw(&g.knight, &mut m.knight, &mut v.knight, optimizer);
        self.bishop
            .adamw(&g.bishop, &mut m.bishop, &mut v.bishop, optimizer);
        self.rook
            .adamw(&g.rook, &mut m.rook, &mut v.rook, optimizer);
        self.queen
            .adamw(&g.queen, &mut m.queen, &mut v.queen, optimizer);
        self.king
            .adamw(&g.king, &mut m.king, &mut v.king, optimizer);
    }

    pub fn backprop_position(
        &self,
        features: &SparseVector,
        g: &mut Self,
        move_idxes: &[MoveIndex],
        errors: &[f32],
        cache: &EgPolicyForwardCache,
    ) {
        let ctx_to = cache.ctx_to;
        let ctx_from = cache.ctx_from;

        let mut ctx_err = Vector::<CTX_SIZE>::zeroed();

        for ((&move_idx, &error), layers) in move_idxes
            .iter()
            .zip(errors.iter())
            .zip(cache.move_layers.iter())
        {
            let from_sq = move_idx.from_sq();
            let to_piece_sq = move_idx.to_sq_for_piece_subnet();

            let from_piece = self.piece_base_subnets(move_idx.piece());
            let to_piece = self.piece_to_subnets(move_idx);

            ctx_err.madd_slice(&layers.to_piece_sq.output_layer(), error, 0);
            ctx_err.madd_slice(&layers.from_piece_sq.output_layer(), error, ATTENTION_SIZE);

            from_piece[from_sq].backprop(
                features,
                &mut g.piece_base_subnets_mut(move_idx.piece())[from_sq],
                error * ctx_from,
                &layers.from_piece_sq,
            );

            to_piece[to_piece_sq].backprop(
                features,
                &mut g.piece_to_subnets_mut(move_idx)[to_piece_sq],
                error * ctx_to,
                &layers.to_piece_sq,
            );
        }

        self.ctx
            .backprop(features, &mut g.ctx, ctx_err, &cache.ctx_layers);
    }

    #[must_use]
    pub fn to_boxed_and_quantized(&self) -> Box<QuantizedEgPolicyNetwork> {
        let mut result: Box<QuantizedEgPolicyNetwork> = allocation::zeroed_box();

        result.ctx = *quantize_ctx(&self.ctx);
        result.pawn.base = *quantize_subnets(&self.pawn.base);
        result.pawn.good_see = *quantize_subnets(&self.pawn.good_see);
        result.knight.base = *quantize_subnets(&self.knight.base);
        result.knight.good_see = *quantize_subnets(&self.knight.good_see);
        result.bishop.base = *quantize_subnets(&self.bishop.base);
        result.bishop.good_see = *quantize_subnets(&self.bishop.good_see);
        result.rook.base = *quantize_subnets(&self.rook.base);
        result.rook.good_see = *quantize_subnets(&self.rook.good_see);
        result.queen.base = *quantize_subnets(&self.queen.base);
        result.queen.good_see = *quantize_subnets(&self.queen.good_see);
        result.king = *quantize_subnets(&self.king);

        result
    }
}

fn quantize_ctx(ctx: &EgCtxNetwork) -> Box<QuantizedCtxNetwork> {
    let mut weights: Box<RawCtxWeights> = allocation::zeroed_box();
    let mut bias: Box<RawCtxBias> = allocation::zeroed_box();

    for (row_idx, weights_row) in weights.iter_mut().enumerate() {
        let row = ctx.weights_row(row_idx);
        for (weight_idx, w) in weights_row.iter_mut().enumerate() {
            *w = nets::q_i16(row[weight_idx], QA);
        }
    }
    for (weight_idx, b) in bias.iter_mut().enumerate() {
        *b = nets::q_i16(ctx.bias()[weight_idx], QA);
    }

    QuantizedCtxNetwork::from_raw(&weights, &bias)
}

fn quantize_subnets(subnets: &SquareSubnets) -> Box<QuantizedSquareSubnets> {
    let mut weights: Box<RawSquareWeights> = allocation::zeroed_box();
    let mut bias: Box<RawSquareBias> = allocation::zeroed_box();

    for (subnet, (raw_w, raw_b)) in subnets.iter().zip(weights.iter_mut().zip(bias.iter_mut())) {
        for (row_idx, weights_row) in raw_w.iter_mut().enumerate() {
            let row = subnet.output.weights_row(row_idx);
            for (weight_idx, w) in weights_row.iter_mut().enumerate() {
                *w = nets::q_i16(row[weight_idx], QA);
            }
        }
        for (weight_idx, b) in raw_b.iter_mut().enumerate() {
            *b = nets::q_i16(subnet.output.bias()[weight_idx], QA);
        }
    }

    QuantizedSquareSubnets::boxed_from_slices(&weights, &bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_does_not_crash() {
        let policy_net = EgPolicyNetwork::random();
        let _quantized_policy_net = policy_net.to_boxed_and_quantized();
    }
}
