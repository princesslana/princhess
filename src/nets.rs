use std::fs;
use std::io::Write;
use std::ops::AddAssign;
use std::path::Path;

use bytemuck::{self, Pod, Zeroable};

use crate::chess::{Piece, Rank, Square};

#[derive(Clone, Copy, Debug, Zeroable)]
#[repr(C)]
pub struct Accumulator<T, const H: usize> {
    pub vals: [T; H],
}

unsafe impl<T: Copy + Zeroable + 'static, const H: usize> Pod for Accumulator<T, H> {}

impl<T: AddAssign, const H: usize> Accumulator<T, H> {
    pub fn set<U: Copy>(&mut self, weights: &Accumulator<U, H>)
    where
        T: From<U>,
    {
        for (i, d) in self.vals.iter_mut().zip(&weights.vals) {
            *i += T::from(*d);
        }
    }
}

impl<const H: usize> Accumulator<i16, H> {
    #[must_use]
    pub fn dot_relu<const Q: i32>(&self, rhs: &Accumulator<i16, H>) -> f32 {
        let mut result: i32 = 0;

        for (a, b) in self.vals.iter().zip(&rhs.vals) {
            result += relu(*a) * relu(*b);
        }

        result as f32 / Q as f32
    }
}

pub fn relu<F>(x: F) -> i32
where
    i32: From<F>,
{
    i32::from(x).max(0)
}

#[must_use]
pub fn screlu(x: i16, q: i32) -> i32 {
    let clamped = i32::from(x).clamp(0, q);
    clamped * clamped
}

/// Saves neural network data to a binary file.
///
/// # Panics
///
/// Panics if the file cannot be created or written to.
pub fn save_to_bin<T: Pod>(dir: &Path, file_name: &str, data: &T) {
    let mut file = fs::File::create(dir.join(file_name)).expect("Failed to create file");

    let slice = bytemuck::bytes_of(data);

    file.write_all(slice).unwrap();
}

#[must_use]
#[derive(Debug, Copy, Clone)]
pub struct MoveIndex {
    piece: Piece,
    promotion: Piece,
    from_sq: Square,
    to_sq: Square,
    good_see: bool,
}

impl MoveIndex {
    pub const SQ_COUNT: usize = Square::COUNT;
    pub const FROM_PIECE_SQ_COUNT: usize = Piece::COUNT * Square::COUNT;
    pub const TO_PIECE_SQ_COUNT: usize = (5 + Piece::COUNT) * Square::COUNT;

    pub fn new(piece: Piece, promotion: Piece, from: Square, to: Square, good_see: bool) -> Self {
        Self {
            piece,
            promotion,
            from_sq: from,
            to_sq: to,
            good_see,
        }
    }

    pub fn from_sq(self) -> Square {
        self.from_sq
    }

    pub fn to_sq(self) -> Square {
        self.to_sq
    }

    #[must_use]
    pub fn from_piece_sq_index(self) -> usize {
        self.piece.index() * Square::COUNT + self.from_sq.index()
    }

    #[must_use]
    pub fn to_piece_sq_index(self) -> usize {
        let bucket = match self.piece {
            Piece::KING => self.piece.index(),
            p => [0, 6][usize::from(self.good_see)] + p.index(),
        };
        let to_sq = match self.promotion {
            Piece::KNIGHT => self.to_sq.with_rank(Rank::_1),
            Piece::BISHOP | Piece::ROOK => self.to_sq.with_rank(Rank::_2),
            _ => self.to_sq,
        };
        bucket * Square::COUNT + to_sq.index()
    }
}
