//! Finds a value in a sub-table's compressed stream.
//!
//! Values are compressed twice: runs of values become symbols of a grammar (a symbol is one
//! value, or a pair of earlier symbols expanding to the left's run then the right's), and the
//! symbol sequence is Huffman coded in fixed-size blocks.

use bytemuck::{Pod, Zeroable};

use crate::tablebase::reader::{Cursor, LoadError, LoadResult};

const CONSTANT: u8 = 0x80;
/// The 12-bit all-ones right half that marks a symbol as a value.
const LEAF: u16 = 0xfff;
/// Larger blocks would make block offsets overflow.
const MAX_BLOCK_SIZE: u8 = 24;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Header {
    flags: u8,
    /// log2 of a block's size in bytes.
    block_size: u8,
    /// log2 of the positions each index entry covers.
    idx_bits: u8,
    /// Dummy blocks after the real ones, so the last index entry can point past the end.
    extra_blocks: u8,
    real_blocks: [u8; 4],
    max_len: u8,
    min_len: u8,
}

/// The block holding the middle position of an index entry's range, and that position's offset
/// from the block's start.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndexEntry {
    block: [u8; 4],
    offset: [u8; 2],
}

impl IndexEntry {
    fn block(self) -> usize {
        u32::from_le_bytes(self.block) as usize
    }

    fn offset(self) -> i64 {
        i64::from(u16::from_le_bytes(self.offset))
    }
}

/// Two 12-bit symbol numbers, left and right.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SymbolPattern([u8; 3]);

struct Symbol {
    left: u16,
    /// `LEAF` for a value, which is then `left`; otherwise the right half of a pair.
    right: u16,
    /// Number of values the symbol expands to.
    len: u32,
}

impl Symbol {
    fn is_leaf(&self) -> bool {
        self.right == LEAF
    }
}

/// Huffman codes of one length.
struct CodeLength {
    /// The smallest code, left-aligned to 64 bits.
    base: u64,
    first_symbol: u16,
}

pub struct Compressed {
    block_size: u32,
    idx_bits: u32,
    min_len: usize,
    /// From `min_len` up.
    code_lengths: Vec<CodeLength>,
    symbols: Vec<Symbol>,
    real_blocks: usize,
    index: &'static [IndexEntry],
    /// Values per block, minus one, little-endian.
    block_sizes: &'static [[u8; 2]],
    /// The blocks, then the rest of the file: decoding reads ahead past the last block.
    data: &'static [u8],
}

pub enum Pairs {
    /// The whole sub-table is one value.
    Constant(u16),
    Compressed(Compressed),
}

/// A pairs header, read before the sections it describes. Every sub-table's sections follow all
/// the headers, grouped by kind: every index table, then every block size table, then every
/// sub-table's data.
pub enum PairsHeader {
    Constant(u16),
    Compressed(CompressedHeader),
}

pub struct CompressedHeader {
    block_size: u32,
    idx_bits: u32,
    min_len: usize,
    code_lengths: Vec<CodeLength>,
    symbols: Vec<Symbol>,
    real_blocks: usize,
    entries: usize,
    blocks: usize,
}

impl PairsHeader {
    /// Reads a pairs header for a sub-table of `positions` values. Only WDL tables store a
    /// constant sub-table's value; a constant DTZ sub-table is all zero.
    pub fn read(cursor: &mut Cursor, positions: u64, wdl: bool) -> LoadResult<Self> {
        if cursor.peek()? & CONSTANT != 0 {
            let [_, value] = cursor.take_array::<2>()?;
            return Ok(Self::Constant(if wdl { u16::from(value) } else { 0 }));
        }

        let corrupt = || LoadError::Corrupt("pairs header");
        let header: &Header = cursor.take_pod()?;
        let idx_bits = u32::from(header.idx_bits);
        let (min_len, max_len) = (usize::from(header.min_len), usize::from(header.max_len));
        let valid = (1..=32).contains(&idx_bits)
            && header.block_size <= MAX_BLOCK_SIZE
            && (1..64).contains(&min_len)
            && (min_len..64).contains(&max_len);
        if !valid {
            return Err(corrupt());
        }

        let first_symbols: Vec<u16> = cursor
            .take(2 * (max_len - min_len + 1))?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_le_bytes(b))
            .collect();

        let num_syms = usize::from(u16::from_le_bytes(cursor.take_array::<2>()?));
        let patterns = bytemuck::cast_slice(cursor.take(3 * num_syms)?);
        cursor.skip(num_syms & 1)?;

        let real_blocks = u32::from_le_bytes(header.real_blocks) as usize;

        Ok(Self::Compressed(CompressedHeader {
            block_size: u32::from(header.block_size),
            idx_bits,
            min_len,
            code_lengths: code_lengths(&first_symbols, min_len).ok_or_else(corrupt)?,
            symbols: symbols(patterns).ok_or_else(corrupt)?,
            real_blocks,
            entries: positions
                .div_ceil(1 << idx_bits)
                .try_into()
                .map_err(|_| corrupt())?,
            blocks: real_blocks + usize::from(header.extra_blocks),
        }))
    }

    pub fn read_index(&self, cursor: &mut Cursor) -> LoadResult<&'static [u8]> {
        match self {
            Self::Constant(_) => Ok(&[]),
            Self::Compressed(h) => cursor.take(6 * h.entries),
        }
    }

    pub fn read_block_sizes(&self, cursor: &mut Cursor) -> LoadResult<&'static [u8]> {
        match self {
            Self::Constant(_) => Ok(&[]),
            Self::Compressed(h) => cursor.take(2 * h.blocks),
        }
    }

    pub fn read_data(&self, cursor: &mut Cursor) -> LoadResult<&'static [u8]> {
        let data = cursor.rest();
        if let Self::Compressed(h) = self {
            let len = h
                .real_blocks
                .checked_mul(1 << h.block_size)
                .ok_or(LoadError::Corrupt("pairs header"))?;
            cursor.skip(len)?;
        }
        Ok(data)
    }

    pub fn finish(
        self,
        index: &'static [u8],
        block_sizes: &'static [u8],
        data: &'static [u8],
    ) -> Pairs {
        match self {
            Self::Constant(v) => Pairs::Constant(v),
            Self::Compressed(h) => Pairs::Compressed(Compressed {
                block_size: h.block_size,
                idx_bits: h.idx_bits,
                min_len: h.min_len,
                code_lengths: h.code_lengths,
                symbols: h.symbols,
                real_blocks: h.real_blocks,
                index: bytemuck::cast_slice(index),
                block_sizes: block_sizes.as_chunks::<2>().0,
                data,
            }),
        }
    }
}

impl Pairs {
    /// The raw value at `idx`: 12 bits, of which WDL tables use the low 8.
    pub fn value(&self, idx: u64) -> Option<u16> {
        match self {
            Self::Constant(v) => Some(*v),
            Self::Compressed(c) => c.value(idx),
        }
    }
}

impl Compressed {
    fn value(&self, idx: u64) -> Option<u16> {
        let (block, mut offset) = self.locate(idx)?;

        let mut words = self
            .data
            .get(block << self.block_size..)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&w| u32::from_be_bytes(w));

        // `code` holds the next 64 bits of the stream, `free` how many of its low bits are
        // consumed and not yet refilled.
        let mut code = (u64::from(words.next()?) << 32) | u64::from(words.next()?);
        let mut free = 0;

        // Shorter codes are numerically larger, so a code's length is the first it reaches
        let mut sym = loop {
            let (l, length) = self
                .code_lengths
                .iter()
                .enumerate()
                .find(|(_, length)| code >= length.base)?;
            let bits = self.min_len + l;
            let n = usize::try_from((code - length.base) >> (64 - bits)).ok()?;
            let symbol = self.symbols.get(usize::from(length.first_symbol) + n)?;

            if offset < symbol.len {
                break symbol;
            }

            offset -= symbol.len;
            code <<= bits;
            free += bits;
            if free >= 32 {
                free -= 32;
                code |= u64::from(words.next()?) << free;
            }
        };

        while !sym.is_leaf() {
            let left = &self.symbols[usize::from(sym.left)];
            if offset < left.len {
                sym = left;
            } else {
                offset -= left.len;
                sym = &self.symbols[usize::from(sym.right)];
            }
        }

        Some(sym.left)
    }

    /// The block holding `idx` and its offset in the block. Each index entry gives the block
    /// and offset of the middle position of its range, so find our offset from that block,
    /// then walk to the block holding it.
    fn locate(&self, idx: u64) -> Option<(usize, u32)> {
        let entry = *self
            .index
            .get(usize::try_from(idx >> self.idx_bits).ok()?)?;
        let half = 1i64 << (self.idx_bits - 1);
        let within = i64::try_from(idx & ((1 << self.idx_bits) - 1)).ok()?;

        let mut block = entry.block();
        let mut offset = within - half + entry.offset();
        let block_len = |b: usize| {
            self.block_sizes
                .get(b)
                .map(|&s| i64::from(u16::from_le_bytes(s)) + 1)
        };

        while offset < 0 {
            block = block.checked_sub(1)?;
            offset += block_len(block)?;
        }
        while offset >= block_len(block)? {
            offset -= block_len(block)?;
            block += 1;
        }

        (block < self.real_blocks).then_some((block, u32::try_from(offset).ok()?))
    }
}

/// `first_symbols` holds the first symbol of each code length from `min_len` up. Longer codes
/// are numerically smaller, so each length's smallest code is half the next longer length's
/// plus the number of codes of that length: `base[l] = (base[l + 1] + count[l + 1]) / 2`.
fn code_lengths(first_symbols: &[u16], min_len: usize) -> Option<Vec<CodeLength>> {
    let mut bases = vec![0u64; first_symbols.len()];

    for l in (0..bases.len() - 1).rev() {
        let count = first_symbols[l].checked_sub(first_symbols[l + 1])?;
        bases[l] = bases[l + 1].midpoint(u64::from(count));
    }

    let lengths = bases
        .iter()
        .zip(first_symbols)
        .enumerate()
        .map(|(l, (&base, &first_symbol))| CodeLength {
            base: base << (64 - (min_len + l)),
            first_symbol,
        })
        .collect();

    Some(lengths)
}

/// Unpacks symbols and works out the values each expands to. A pair may refer to any symbol in
/// the list, so lengths are resolved depth first.
fn symbols(patterns: &[SymbolPattern]) -> Option<Vec<Symbol>> {
    let mut symbols: Vec<Symbol> = patterns
        .iter()
        .map(|&SymbolPattern([a, b, c])| Symbol {
            left: (u16::from(b & 0x0f) << 8) | u16::from(a),
            right: (u16::from(c) << 4) | u16::from(b >> 4),
            len: 0,
        })
        .collect();

    let n = symbols.len();
    let mut stack = Vec::new();

    for s in 0..n {
        stack.push(s);
        while let Some(&top) = stack.last() {
            let sym = &symbols[top];
            if sym.len > 0 {
                stack.pop();
                continue;
            }
            if sym.is_leaf() {
                symbols[top].len = 1;
                stack.pop();
                continue;
            }

            let (left, right) = (usize::from(sym.left), usize::from(sym.right));
            // A longer chain than there are symbols is a cycle
            if left >= n || right >= n || stack.len() > n {
                return None;
            }

            match (symbols[left].len, symbols[right].len) {
                (0, _) => stack.push(left),
                (_, 0) => stack.push(right),
                (l, r) => {
                    symbols[top].len = l.checked_add(r)?;
                    stack.pop();
                }
            }
        }
    }

    Some(symbols)
}
