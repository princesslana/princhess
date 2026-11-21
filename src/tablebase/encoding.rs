//! Turns a placement of a table's pieces into its index in the table.
//!
//! An index is a mixed-radix number with one digit per group: the leading group (two kings,
//! three unique pieces, or the leading side's pawns), then the other side's pawns, then each run
//! of identical pieces. Symmetries are folded away first, so each digit counts only canonical
//! placements.

use std::cmp::Reverse;
use std::ops::Range;

use arrayvec::ArrayVec;

use crate::chess::{Color, File, Piece, Rank, Square};
use crate::tablebase::reader::{LoadError, LoadResult};

pub const MAX_PIECES: usize = 7;

/// Squares in a table's piece order.
pub type Squares = ArrayVec<Square, MAX_PIECES>;

/// Squares a pawn can stand on, ranks 2–7.
const PAWN_SQUARES: usize = 48;

/// Squares of the a1–d1–d4 triangle off the a1–h8 diagonal, and on it.
const TRIANGLE_OFF_DIAGONAL: usize = 6;
const TRIANGLE_SQUARES: usize = 10;

// Three unique pieces are numbered in four ranges, by the first piece off the a1–h8 diagonal:
// the first (anywhere off the diagonal in the triangle), the second (below the diagonal, 28
// squares), the third, or none.
const FIRST_OFF_DIAGONAL: u64 = 0;
const SECOND_OFF_DIAGONAL: u64 = FIRST_OFF_DIAGONAL + 6 * 63 * 62;
const THIRD_OFF_DIAGONAL: u64 = SECOND_OFF_DIAGONAL + 4 * 28 * 62;
const NONE_OFF_DIAGONAL: u64 = THIRD_OFF_DIAGONAL + 4 * 7 * 28;
const TRIPLE_PLACEMENTS: u64 = NONE_OFF_DIAGONAL + 4 * 7 * 6;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Leading {
    /// No unique piece besides the kings, so the kings lead.
    Kings,
    /// The first three unique pieces lead.
    Triple,
    /// The leading side's pawns.
    Pawns(usize),
}

impl Leading {
    fn len(self) -> usize {
        match self {
            Self::Kings => 2,
            Self::Triple => 3,
            Self::Pawns(n) => n,
        }
    }
}

/// What a table's material says about its encodings. White and black are the table's first and
/// second side.
#[derive(Clone, Copy)]
pub struct Layout {
    pub leading: Leading,
    pub other_pawns: usize,
    /// Piece counts by colour and piece, kings included.
    pub counts: [[usize; Piece::COUNT]; 2],
}

impl Layout {
    pub fn pieces(&self) -> usize {
        self.counts.iter().flatten().sum()
    }

    pub fn has_pawns(&self) -> bool {
        matches!(self.leading, Leading::Pawns(_))
    }

    /// Pawn tables have a sub-table per leading pawn file, a–d.
    pub fn subtables(&self) -> usize {
        if self.has_pawns() {
            4
        } else {
            1
        }
    }

    /// Bytes giving the digit positions of the leading group and the other side's pawns.
    fn order_bytes(&self) -> usize {
        1 + usize::from(self.other_pawns > 0)
    }

    /// A sub-table's descriptor: the order bytes, then a byte per piece.
    pub fn descriptor_len(&self) -> usize {
        self.order_bytes() + self.pieces()
    }

    fn count(&self, (color, piece): (Color, Piece)) -> usize {
        self.counts[usize::from(u8::from(color))][piece.index()]
    }
}

/// Pieces after the leading group whose squares form one digit: a combination of their squares
/// among those the earlier groups left free.
struct Group {
    squares: Range<usize>,
    stride: u64,
}

impl Group {
    /// `offset`: squares at the bottom of the board the group can't use (rank 1 for pawns).
    fn digit(&self, squares: &mut [Square], offset: usize) -> u64 {
        let (placed, rest) = squares.split_at_mut(self.squares.start);
        let group = &mut rest[..self.squares.len()];
        group.sort_unstable_by_key(|sq| sq.index());

        let digit: u64 = group
            .iter()
            .enumerate()
            .map(|(i, &sq)| {
                let skips = placed.iter().filter(|s| s.index() < sq.index()).count();
                binomial(sq.index() - skips - offset, i + 1)
            })
            .sum();

        digit * self.stride
    }
}

/// One sub-table's piece order and index radix.
pub struct Encoding {
    /// The pieces in the table's order, by table colour.
    pub pieces: ArrayVec<(Color, Piece), MAX_PIECES>,
    leading: Leading,
    leading_stride: u64,
    /// The other side's pawns, when both sides have pawns.
    other_pawns: Option<Group>,
    /// Each run of identical pieces after those.
    rest: ArrayVec<Group, MAX_PIECES>,
}

impl Encoding {
    /// Reads the encoding for a side to move from a sub-table's descriptor, whose bytes hold a
    /// nibble per side to move. `file` is the leading pawn's folded file for pawn tables.
    /// Returns the encoding and the number of positions it indexes.
    pub fn read(
        descriptor: &[u8],
        side: Color,
        layout: &Layout,
        file: usize,
    ) -> LoadResult<(Self, u64)> {
        let shift = 4 * u32::from(u8::from(side));
        let nibble = |b: u8| (b >> shift) & 0x0f;
        let (order_bytes, codes) = descriptor.split_at(layout.order_bytes());

        let pieces = codes
            .iter()
            .map(|&b| table_piece(nibble(b)))
            .collect::<Option<ArrayVec<_, MAX_PIECES>>>()
            .ok_or(LoadError::Corrupt("piece code"))?;
        check_pieces(&pieces, layout)?;

        let fixed = layout.leading.len() + layout.other_pawns;
        let mut runs = pieces[fixed..]
            .chunk_by(|a, b| a == b)
            .scan(fixed, |start, run| {
                let squares = *start..*start + run.len();
                *start = squares.end;
                Some(squares)
            });

        // The order bytes give the digit positions, least significant first, of the leading
        // group and the other side's pawns. The remaining groups fill the others in turn.
        let leading_digit = usize::from(nibble(order_bytes[0]));
        let pawns_digit = order_bytes.get(1).map(|&b| usize::from(nibble(b)));
        let digits = 1 + usize::from(layout.other_pawns > 0) + runs.clone().count();

        let mut free = 64 - fixed;
        let mut radix = 1u64;
        let mut leading_stride = None;
        let mut other_pawns = None;
        let mut rest = ArrayVec::new();

        for digit in 0..digits {
            let stride = radix;
            let placements = if digit == leading_digit {
                leading_stride = Some(stride);
                match layout.leading {
                    Leading::Kings => KING_PAIRS,
                    Leading::Triple => TRIPLE_PLACEMENTS,
                    Leading::Pawns(n) => LEAD_PAWN_PLACEMENTS[n - 1][file],
                }
            } else if Some(digit) == pawns_digit {
                let start = layout.leading.len();
                let squares = start..start + layout.other_pawns;
                other_pawns = Some(Group { squares, stride });
                binomial(PAWN_SQUARES - start, layout.other_pawns)
            } else {
                let squares = runs.next().ok_or(LoadError::Corrupt("piece order"))?;
                let placements = binomial(free, squares.len());
                free -= squares.len();
                rest.push(Group { squares, stride });
                placements
            };
            radix *= placements;
        }

        let Some(leading_stride) = leading_stride else {
            return Err(LoadError::Corrupt("piece order"));
        };
        if other_pawns.is_some() != (layout.other_pawns > 0) {
            return Err(LoadError::Corrupt("piece order"));
        }

        let encoding = Self {
            pieces,
            leading: layout.leading,
            leading_stride,
            other_pawns,
            rest,
        };

        Ok((encoding, radix))
    }

    /// `squares` holds the squares in piece order.
    pub fn index(&self, mut squares: Squares) -> Option<u64> {
        let p = &mut squares[..];

        if let Leading::Pawns(n) = self.leading {
            let first = (0..n).min_by_key(|&i| lead_order(p[i]))?;
            p.swap(0, first);
        }

        // Fold the board so the first piece (the leading pawn for pawn tables) is on files a–d,
        // and for pawnless tables also on ranks 1–4.
        let mut fold = 0;
        if p[0].file() > File::D {
            fold |= Square::FLIP_FILE_MASK;
        }
        if !self.has_pawns() && p[0].rank() > Rank::R4 {
            fold |= Square::FLIP_RANK_MASK;
        }
        for sq in p.iter_mut() {
            *sq = *sq ^ fold;
        }

        let leading = match self.leading {
            Leading::Kings | Leading::Triple => {
                // Pawnless tables also fold the a1–h8 diagonal, so the leading group's first
                // piece off the diagonal is below it.
                let lead = &p[..self.leading.len()];
                if lead
                    .iter()
                    .find(|&&sq| !on_diagonal(sq))
                    .is_some_and(|&sq| above_diagonal(sq))
                {
                    for sq in p.iter_mut() {
                        *sq = transpose(*sq);
                    }
                }

                if self.leading == Leading::Kings {
                    king_pair_index(p[0], p[1])?
                } else {
                    triple_index(p[0], p[1], p[2])
                }
            }
            Leading::Pawns(n) => {
                p[1..n].sort_unstable_by_key(|&sq| Reverse(pawn_order(sq)));
                let others: u64 = (1..n).map(|i| binomial(pawn_order(p[i]), n - i)).sum();
                LEAD_PAWN_OFFSET[n - 1][lead_order(p[0])] + others
            }
        };

        let mut idx = leading * self.leading_stride;
        if let Some(pawns) = &self.other_pawns {
            idx += pawns.digit(p, 8);
        }
        for group in &self.rest {
            idx += group.digit(p, 0);
        }

        Some(idx)
    }

    fn has_pawns(&self) -> bool {
        matches!(self.leading, Leading::Pawns(_))
    }
}

/// The leading pawn's folded file (a = h = 0, ..., d = e = 3) picks a pawn table's sub-table.
/// The leading pawn is the one that comes first in `lead_order`.
pub fn leading_file(pawns: impl Iterator<Item = Square>) -> Option<usize> {
    pawns.min_by_key(|&sq| lead_order(sq)).map(folded_file)
}

/// A piece code: bits 0–2 the type (pawn = 1 to king = 6), bit 3 the table's second side.
fn table_piece(code: u8) -> Option<(Color, Piece)> {
    let piece = match code & 0x07 {
        t @ 1..=6 => Piece::from(t - 1),
        _ => return None,
    };

    Some((Color::from(code & 0x08), piece))
}

/// The pieces must be the table's material, grouped as its layout says, so every index computed
/// from a legal board stays in range.
fn check_pieces(pieces: &[(Color, Piece)], layout: &Layout) -> LoadResult<()> {
    let corrupt = Err(LoadError::Corrupt("piece order"));
    let runs: ArrayVec<_, MAX_PIECES> = pieces.chunk_by(|a, b| a == b).collect();

    // Each piece in one run, of all its copies
    let whole_runs = runs.iter().all(|run| run.len() == layout.count(run[0]));
    let distinct = runs
        .iter()
        .enumerate()
        .all(|(i, run)| runs[..i].iter().all(|other| other[0] != run[0]));
    if pieces.len() != layout.pieces() || !whole_runs || !distinct {
        return corrupt;
    }

    let (lead, rest) = pieces.split_at(layout.leading.len());
    let leading_ok = match layout.leading {
        Leading::Kings => lead.iter().all(|&(_, piece)| piece == Piece::KING),
        Leading::Triple => lead.iter().all(|&p| layout.count(p) == 1),
        Leading::Pawns(_) => {
            let other = &rest[..layout.other_pawns];
            lead.iter().all(|&(_, piece)| piece == Piece::PAWN)
                && other
                    .iter()
                    .all(|&(c, piece)| piece == Piece::PAWN && c != lead[0].0)
        }
    };

    if leading_ok {
        Ok(())
    } else {
        corrupt
    }
}

fn folded_file(sq: Square) -> usize {
    let f = sq.file().index();
    f.min(7 - f)
}

fn on_diagonal(sq: Square) -> bool {
    sq.rank().index() == sq.file().index()
}

fn above_diagonal(sq: Square) -> bool {
    sq.rank().index() > sq.file().index()
}

fn transpose(sq: Square) -> Square {
    let i = sq.index();
    Square::from(((i >> 3) | (i << 3)) & 63)
}

/// A leading pawn's place among the 24 squares it can stand on: by folded file, then rank.
fn lead_order(sq: Square) -> usize {
    6 * folded_file(sq) + sq.rank().index() - 1
}

/// Numbers pawn squares from 47 (a2) down to 0 (e7): by folded file, then rank, each square
/// followed by its mirror (h2 after a2). The leading side's other pawns always number below the
/// leading pawn, so they are numbered as a combination of the squares below it.
fn pawn_order(sq: Square) -> usize {
    pawn_order_index(sq.index())
}

fn king_pair_index(k0: Square, k1: Square) -> Option<u64> {
    match KING_PAIR_INDEX[TRIANGLE[k0]][k1] {
        u16::MAX => None,
        idx => Some(u64::from(idx)),
    }
}

/// Three unique pieces, folded: `p0` in the triangle, and the first off-diagonal piece below
/// the diagonal. Each case numbers the placements left once the earlier pieces are placed.
fn triple_index(p0: Square, p1: Square, p2: Square) -> u64 {
    let s1 = u64::from(p1.index() > p0.index());
    let s2 = u64::from(p2.index() > p0.index()) + u64::from(p2.index() > p1.index());
    let square = |s: Square| s.index() as u64;
    let diagonal_pos = |s: Square| s.file().index() as u64;
    let lower = |s: Square| u64::from(LOWER[s]);

    if !on_diagonal(p0) {
        FIRST_OFF_DIAGONAL
            + TRIANGLE[p0] as u64 * 63 * 62
            + (square(p1) - s1) * 62
            + (square(p2) - s2)
    } else if !on_diagonal(p1) {
        SECOND_OFF_DIAGONAL + diagonal_pos(p0) * 28 * 62 + lower(p1) * 62 + (square(p2) - s2)
    } else if !on_diagonal(p2) {
        THIRD_OFF_DIAGONAL + diagonal_pos(p0) * 7 * 28 + (diagonal_pos(p1) - s1) * 28 + lower(p2)
    } else {
        NONE_OFF_DIAGONAL
            + diagonal_pos(p0) * 7 * 6
            + (diagonal_pos(p1) - s1) * 6
            + (diagonal_pos(p2) - s2)
    }
}

pub fn binomial(n: usize, k: usize) -> u64 {
    BINOMIAL[k][n]
}

// Built at compile time on square indices (a1 = 0), and indexed by `Square` at runtime.

const fn pawn_order_index(sq: usize) -> usize {
    let (f, r) = (sq & 7, sq >> 3);
    let folded = if f < 4 { f } else { 7 - f };
    PAWN_SQUARES - 1 - (12 * folded + 2 * (r - 1) + (f >= 4) as usize)
}

const BINOMIAL: [[u64; 64]; MAX_PIECES] = {
    let mut table = [[0; 64]; MAX_PIECES];
    let mut n = 0;
    while n < 64 {
        table[0][n] = 1;
        let mut k = 1;
        while k < MAX_PIECES && k <= n {
            table[k][n] = table[k - 1][n - 1] + if k < n { table[k][n - 1] } else { 0 };
            k += 1;
        }
        n += 1;
    }
    table
};

/// Placements of `n` leading pawns whose leading pawn comes earlier on the same file, indexed by
/// `n − 1` and the leading pawn's `lead_order`. A placement's index is this plus its number
/// among those with the same leading pawn.
const LEAD_PAWN_OFFSET: [[u64; 24]; MAX_PIECES - 1] = lead_pawn_tables().0;
/// Placements of `n` leading pawns per leading pawn file: the leading group's size.
const LEAD_PAWN_PLACEMENTS: [[u64; 4]; MAX_PIECES - 1] = lead_pawn_tables().1;

const fn lead_pawn_tables() -> ([[u64; 24]; MAX_PIECES - 1], [[u64; 4]; MAX_PIECES - 1]) {
    let mut offset = [[0; 24]; MAX_PIECES - 1];
    let mut placements = [[0; 4]; MAX_PIECES - 1];
    let mut others = 0;
    while others < MAX_PIECES - 1 {
        let mut f = 0;
        while f < 4 {
            let mut sum = 0;
            let mut r = 0;
            while r < 6 {
                offset[others][6 * f + r] = sum;
                sum += BINOMIAL[others][pawn_order_index((r + 1) * 8 + f)];
                r += 1;
            }
            placements[others][f] = sum;
            f += 1;
        }
        others += 1;
    }
    (offset, placements)
}

/// Numbers the triangle: b1, c1, d1, c2, d2, d3 are 0–5, the diagonal a1, b2, c3, d4 is 6–9.
const TRIANGLE: [usize; 64] = {
    let mut table = [usize::MAX; 64];
    let mut next = 0;
    let mut sq = 0;
    while sq < 64 {
        let (f, r) = (sq & 7, sq >> 3);
        if f < 4 && r < f {
            table[sq] = next;
            next += 1;
        } else if f < 4 && r == f {
            table[sq] = TRIANGLE_OFF_DIAGONAL + f;
        }
        sq += 1;
    }
    table
};

/// Numbers the 28 squares below the a1–h8 diagonal.
const LOWER: [u8; 64] = {
    let mut table = [u8::MAX; 64];
    let mut next = 0;
    let mut sq = 0;
    while sq < 64 {
        if (sq >> 3) < (sq & 7) {
            table[sq] = next;
            next += 1;
        }
        sq += 1;
    }
    table
};

const KING_PAIR_INDEX: [[u16; 64]; TRIANGLE_SQUARES] = king_pair_tables().0;
/// Canonical king pairs, the leading group of a table whose only unique pieces are the kings.
const KING_PAIRS: u64 = king_pair_tables().1;

/// Numbers the canonical king pairs by the triangle square of the first king, then the second
/// king's square: first kings off the diagonal, then on it with the second king below, then
/// both on the diagonal. Kings on equal or adjacent squares are `u16::MAX`.
const fn king_pair_tables() -> ([[u16; 64]; TRIANGLE_SQUARES], u64) {
    let mut table = [[u16::MAX; 64]; TRIANGLE_SQUARES];
    let mut next = 0;
    let mut pass = 0;
    while pass < 3 {
        let mut t = if pass == 0 { 0 } else { TRIANGLE_OFF_DIAGONAL };
        let end = if pass == 0 {
            TRIANGLE_OFF_DIAGONAL
        } else {
            TRIANGLE_SQUARES
        };
        while t < end {
            let k0 = triangle_square(t);
            let mut k1 = 0;
            while k1 < 64 {
                let (f, r) = (k1 & 7, k1 >> 3);
                let wanted = match pass {
                    0 => true,
                    1 => r < f,
                    _ => r == f,
                };
                if wanted && !same_or_adjacent(k0, k1) {
                    table[t][k1] = next;
                    next += 1;
                }
                k1 += 1;
            }
            t += 1;
        }
        pass += 1;
    }
    (table, next as u64)
}

const fn triangle_square(t: usize) -> usize {
    let mut sq = 0;
    while TRIANGLE[sq] != t {
        sq += 1;
    }
    sq
}

const fn same_or_adjacent(a: usize, b: usize) -> bool {
    (a & 7).abs_diff(b & 7) <= 1 && (a >> 3).abs_diff(b >> 3) <= 1
}
