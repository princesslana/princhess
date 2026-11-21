//! One table file: its header, sub-tables and DTZ value maps, read from a memory map.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

use arrayvec::ArrayVec;
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;

use crate::chess::{Board, Color, Piece, Square};
use crate::tablebase::compression::{Pairs, PairsHeader};
use crate::tablebase::encoding::{self, Encoding, Layout, Squares};
use crate::tablebase::reader::{Cursor, LoadError, LoadResult, Span};
use crate::tablebase::Outcome;

/// Each sub-table's data starts on this boundary, and so does the trailer.
pub const DATA_ALIGN: usize = 64;
/// Bytes after the padded data, which nothing reads.
pub const TRAILER_LEN: usize = 16;

const WDL_MAGIC: [u8; 4] = 0x5d23_e871_u32.to_le_bytes();
const DTZ_MAGIC: [u8; 4] = 0xa50c_66d7_u32.to_le_bytes();

/// Split WDL tables store both sides to move. DTZ tables store one.
const SPLIT: u8 = 0x01;

// DTZ flags, in the flags byte of each leading pawn file's first pairs header
const DTZ_STORED_SIDE: u8 = 0x01;
const DTZ_MAPPED: u8 = 0x02;
const DTZ_WIN_IN_PLIES: u8 = 0x04;
const DTZ_LOSS_IN_PLIES: u8 = 0x08;
const DTZ_WIDE_MAPS: u8 = 0x10;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FileHeader {
    magic: [u8; 4],
    flags: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Wdl,
    Dtz,
}

pub enum DtzEntry {
    Value(i32),
    /// DTZ tables store one side to move; this position has the other.
    OtherSide,
}

/// How a board maps onto a table.
#[derive(Clone, Copy)]
pub struct View {
    /// The table's first side is the board's black.
    pub swap_colours: bool,
    /// The side to move, in table colours.
    pub stm: Color,
    /// Both sides have the same pieces, so the table stores only its first side to move.
    pub symmetric: bool,
}

struct SubTable {
    encoding: Encoding,
    pairs: Pairs,
}

/// How a DTZ sub-table's raw values become distances.
struct DtzValues {
    stored_side: Color,
    /// Values are in plies rather than moves.
    win_in_plies: bool,
    loss_in_plies: bool,
    maps: Option<DtzMaps>,
}

/// Value maps for wins, losses, cursed wins and blessed losses. Narrow maps hold bytes, wide
/// maps little-endian `u16`s.
enum DtzMaps {
    Narrow([Span; 4]),
    Wide([Span; 4]),
}

impl DtzMaps {
    fn get(&self, file: &[u8], outcome: Outcome, raw: u16) -> Option<i32> {
        let map = match outcome {
            Outcome::Win => 0,
            Outcome::Loss => 1,
            Outcome::CursedWin => 2,
            Outcome::BlessedLoss => 3,
            Outcome::Draw => return None,
        };
        let raw = usize::from(raw);

        match self {
            Self::Narrow(maps) => maps[map].of(file)?.get(raw).map(|&v| i32::from(v)),
            Self::Wide(maps) => maps[map]
                .of(file)?
                .as_chunks::<2>()
                .0
                .get(raw)
                .map(|&v| i32::from(u16::from_le_bytes(v))),
        }
    }
}

/// The sub-tables for one file of the leading pawn, or the only ones of a pawnless table.
struct FileTable {
    /// Per side to move stored.
    sides: ArrayVec<SubTable, 2>,
    dtz: Option<DtzValues>,
}

/// The parsed structures locate their sections in `map` by `Span`.
pub struct Table {
    has_pawns: bool,
    files: ArrayVec<FileTable, 4>,
    map: Mmap,
}

impl Table {
    pub fn open(path: &Path, kind: Kind, layout: &Layout) -> LoadResult<Self> {
        let file = open_for_random_access(path)?;

        // SAFETY: Table files are only read. Like any reader of a file mapping, we rely on
        // nothing truncating or rewriting the files while they are in use.
        let map = unsafe { Mmap::map(&file) }?;

        // Probes read scattered blocks, so read-ahead would only fill the page cache
        #[cfg(unix)]
        let _ = map.advise(memmap2::Advice::Random);

        let (has_pawns, files) = read(&map, kind, layout)?;

        Ok(Self {
            has_pawns,
            files,
            map,
        })
    }

    pub fn wdl(&self, b: &Board, view: View) -> Option<Outcome> {
        let file = self.leading_file(b, view)?;
        let sub = self.files[file]
            .sides
            .get(usize::from(u8::from(view.stm)))?;

        // Stored as the outcome + 2
        match self.value(b, view, sub)? & 0xff {
            0 => Some(Outcome::Loss),
            1 => Some(Outcome::BlessedLoss),
            2 => Some(Outcome::Draw),
            3 => Some(Outcome::CursedWin),
            4 => Some(Outcome::Win),
            _ => None,
        }
    }

    pub fn dtz(&self, b: &Board, view: View, outcome: Outcome) -> Option<DtzEntry> {
        let file = &self.files[self.leading_file(b, view)?];
        let values = file.dtz.as_ref()?;

        if !view.symmetric && values.stored_side != view.stm {
            return Some(DtzEntry::OtherSide);
        }

        let raw = self.value(b, view, &file.sides[0])?;
        let v = match &values.maps {
            Some(maps) => maps.get(&self.map, outcome, raw)?,
            None => i32::from(raw),
        };

        let in_plies = match outcome {
            Outcome::Win => values.win_in_plies,
            Outcome::Loss => values.loss_in_plies,
            _ => false,
        };

        Some(DtzEntry::Value(if in_plies { v } else { 2 * v }))
    }

    fn value(&self, b: &Board, view: View, sub: &SubTable) -> Option<u16> {
        let idx = sub.encoding.index(self.squares(b, view, &sub.encoding)?)?;
        sub.pairs.value(&self.map, idx)
    }

    /// Pawn tables have a sub-table per file of the leading pawn. Its piece comes first in every
    /// sub-table's order.
    fn leading_file(&self, b: &Board, view: View) -> Option<usize> {
        if !self.has_pawns {
            return Some(0);
        }

        let leading = self.files[0].sides[0].encoding.pieces[0];
        encoding::leading_file(self.squares_of(b, view, leading))
    }

    /// The board's squares in a sub-table's piece order.
    fn squares(&self, b: &Board, view: View, encoding: &Encoding) -> Option<Squares> {
        let mut squares = Squares::new();

        for group in encoding.pieces.chunk_by(|a, b| a == b) {
            let before = squares.len();
            squares.extend(self.squares_of(b, view, group[0]));
            if squares.len() - before != group.len() {
                return None;
            }
        }

        Some(squares)
    }

    /// Colours swap with the view, and for pawns so do ranks, since pawns move the other way.
    fn squares_of(
        &self,
        b: &Board,
        view: View,
        (color, piece): (Color, Piece),
    ) -> impl Iterator<Item = Square> {
        let (color, mirror) = match (view.swap_colours, self.has_pawns) {
            (false, _) => (color, 0),
            (true, false) => (!color, 0),
            (true, true) => (!color, Square::FLIP_RANK_MASK),
        };

        (b.by_color(color) & b.by_piece(piece))
            .into_iter()
            .map(move |sq| sq ^ mirror)
    }
}

fn open_for_random_access(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);

    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;

        const FILE_FLAG_RANDOM_ACCESS: u32 = 0x1000_0000;
        options.custom_flags(FILE_FLAG_RANDOM_ACCESS);
    }

    options.open(path)
}

/// The file's sections in order: piece descriptors, pairs headers, DTZ value maps, then every
/// sub-table's index table, every block size table and every data section. Returns whether the
/// table has pawns, and its file tables.
fn read(data: &[u8], kind: Kind, layout: &Layout) -> LoadResult<(bool, ArrayVec<FileTable, 4>)> {
    let mut cursor = Cursor::new(data);

    let header: &FileHeader = cursor.take_pod()?;
    let magic = match kind {
        Kind::Wdl => WDL_MAGIC,
        Kind::Dtz => DTZ_MAGIC,
    };
    if header.magic != magic {
        return Err(LoadError::BadMagic);
    }

    let sides: &[Color] = if kind == Kind::Wdl && header.flags & SPLIT != 0 {
        &[Color::WHITE, Color::BLACK]
    } else {
        &[Color::WHITE]
    };

    let encodings = (0..layout.subtables())
        .map(|file| {
            let descriptor = cursor.take(layout.descriptor_len())?;
            sides
                .iter()
                .map(|&side| Encoding::read(descriptor, side, layout, file))
                .collect::<LoadResult<ArrayVec<_, 2>>>()
        })
        .collect::<LoadResult<ArrayVec<_, 4>>>()?;
    cursor.align(2)?;

    let mut dtz_flags = ArrayVec::<u8, 4>::new();
    let mut headers = Vec::new();
    for file_encodings in &encodings {
        dtz_flags.push(cursor.peek()?);
        for (_, positions) in file_encodings {
            headers.push(PairsHeader::read(
                &mut cursor,
                *positions,
                kind == Kind::Wdl,
            )?);
        }
    }

    let dtz = match kind {
        Kind::Wdl => Vec::new(),
        Kind::Dtz => {
            let dtz = read_dtz_values(&mut cursor, &dtz_flags)?;
            cursor.align(2)?;
            dtz
        }
    };

    let index = headers
        .iter()
        .map(|h| h.read_index(&mut cursor))
        .collect::<LoadResult<Vec<_>>>()?;
    let block_sizes = headers
        .iter()
        .map(|h| h.read_block_sizes(&mut cursor))
        .collect::<LoadResult<Vec<_>>>()?;
    let data_sections = headers
        .iter()
        .map(|h| {
            cursor.align(DATA_ALIGN)?;
            h.read_data(&mut cursor)
        })
        .collect::<LoadResult<Vec<_>>>()?;

    cursor.align(DATA_ALIGN)?;
    if cursor.pos() + TRAILER_LEN != data.len() {
        return Err(LoadError::Corrupt("layout"));
    }

    let mut pairs = headers
        .into_iter()
        .zip(index)
        .zip(block_sizes)
        .zip(data_sections)
        .map(|(((header, index), sizes), data)| header.finish(index, sizes, data));
    let mut dtz = dtz.into_iter();

    let files = encodings
        .into_iter()
        .map(|file_encodings| FileTable {
            sides: file_encodings
                .into_iter()
                .zip(&mut pairs)
                .map(|((encoding, _), pairs)| SubTable { encoding, pairs })
                .collect(),
            dtz: dtz.next(),
        })
        .collect();

    Ok((layout.has_pawns(), files))
}

fn read_dtz_values(cursor: &mut Cursor, flags: &[u8]) -> LoadResult<Vec<DtzValues>> {
    flags
        .iter()
        .map(|&flags| {
            let maps = match (flags & DTZ_MAPPED != 0, flags & DTZ_WIDE_MAPS != 0) {
                (false, _) => None,
                (true, false) => Some(DtzMaps::Narrow(read_maps(cursor, |c| {
                    let [len] = c.take_array::<1>()?;
                    c.take_span(usize::from(len))
                })?)),
                (true, true) => {
                    cursor.align(2)?;
                    Some(DtzMaps::Wide(read_maps(cursor, |c| {
                        let len = u16::from_le_bytes(c.take_array::<2>()?);
                        c.take_span(2 * usize::from(len))
                    })?))
                }
            };

            Ok(DtzValues {
                stored_side: Color::from(flags & DTZ_STORED_SIDE),
                win_in_plies: flags & DTZ_WIN_IN_PLIES != 0,
                loss_in_plies: flags & DTZ_LOSS_IN_PLIES != 0,
                maps,
            })
        })
        .collect()
}

fn read_maps(
    cursor: &mut Cursor,
    mut read: impl FnMut(&mut Cursor) -> LoadResult<Span>,
) -> LoadResult<[Span; 4]> {
    let mut maps = [Span::EMPTY; 4];
    for map in &mut maps {
        *map = read(cursor)?;
    }
    Ok(maps)
}
