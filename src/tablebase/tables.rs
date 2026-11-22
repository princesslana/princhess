//! The tablebase set: which tables exist, found by material, each opened on its first probe.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use nohash_hasher::{BuildNoHashHasher, IsEnabled};

use crate::chess::{Board, Color, Piece};
use crate::tablebase::encoding::{Layout, Leading, MAX_PIECES};
use crate::tablebase::table::{Kind, Table, View, DATA_ALIGN, TRAILER_LEN};
use crate::tablebase::Outcome;

pub use crate::tablebase::table::DtzEntry;

/// Pieces other than kings, in the order of a table name's letters.
const PIECE_LETTERS: [(char, Piece); 5] = [
    ('Q', Piece::QUEEN),
    ('R', Piece::ROOK),
    ('B', Piece::BISHOP),
    ('N', Piece::KNIGHT),
    ('P', Piece::PAWN),
];

/// Piece counts other than kings, a nibble each: white pawn to queen, then black.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Material(u64);

impl Material {
    /// Five piece types of four bits each.
    const SIDE_BITS: u32 = 20;

    fn of(b: &Board) -> Self {
        let mut material = Self(0);
        for color in [Color::WHITE, Color::BLACK] {
            for (_, piece) in PIECE_LETTERS {
                let count = (b.by_color(color) & b.by_piece(piece)).count();
                material.add(color, piece, count as u64);
            }
        }
        material
    }

    fn add(&mut self, color: Color, piece: Piece, count: u64) {
        self.0 += count << Self::shift(color, piece);
    }

    fn count(self, color: Color, piece: Piece) -> usize {
        ((self.0 >> Self::shift(color, piece)) & 0x0f) as usize
    }

    fn shift(color: Color, piece: Piece) -> u32 {
        u32::from(u8::from(color)) * Self::SIDE_BITS + 4 * u32::from(u8::from(piece))
    }

    fn swapped(self) -> Self {
        let white = self.0 & ((1 << Self::SIDE_BITS) - 1);
        Self((self.0 >> Self::SIDE_BITS) | (white << Self::SIDE_BITS))
    }

    fn is_bare_kings(self) -> bool {
        self.0 == 0
    }

    /// From a table name such as `KRPvKR`, its first side as white.
    fn from_name(name: &str) -> Option<Self> {
        // Longer names could overflow a count into the next piece's before the size check
        if name.chars().filter(|&c| c != 'v').count() > MAX_PIECES {
            return None;
        }

        let (white, black) = name.split_once('v')?;
        let mut material = Self(0);

        for (color, side) in [(Color::WHITE, white), (Color::BLACK, black)] {
            let pieces = side.strip_prefix('K')?;
            for c in pieces.chars() {
                let (_, piece) = PIECE_LETTERS.iter().find(|(letter, _)| *letter == c)?;
                material.add(color, *piece, 1);
            }
        }

        let pieces = material.layout().pieces();
        (3..=MAX_PIECES).contains(&pieces).then_some(material)
    }

    /// `self` must have the table's first side as white.
    fn layout(self) -> Layout {
        let mut counts = [[0; Piece::COUNT]; 2];
        for (color, side) in [Color::WHITE, Color::BLACK].into_iter().zip(&mut counts) {
            side[Piece::KING.index()] = 1;
            for (_, piece) in PIECE_LETTERS {
                side[piece.index()] = self.count(color, piece);
            }
        }

        let pawns = |c| self.count(c, Piece::PAWN);
        let (white, black) = (pawns(Color::WHITE), pawns(Color::BLACK));

        let (leading, other_pawns) = if white + black == 0 {
            let unique = counts.iter().flatten().filter(|&&n| n == 1).count();
            // Two of the unique pieces are the kings
            let leading = if unique == 2 {
                Leading::Kings
            } else {
                Leading::Triple
            };
            (leading, 0)
        } else if black > 0 && (white == 0 || white > black) {
            // The side with fewer pawns, but at least one, leads; white on a tie
            (Leading::Pawns(black), white)
        } else {
            (Leading::Pawns(white), black)
        };

        Layout {
            leading,
            other_pawns,
            counts,
        }
    }
}

impl Hash for Material {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // The hash map picks buckets by the low bits, and a product's low bits depend only on
        // the key's low bits, so fold the high half back in
        let h = self.0.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        state.write_u64(h ^ (h >> 32));
    }
}

impl IsEnabled for Material {}

/// A table file, opened on first use. A file that fails to open stays unavailable. Boxed, since
/// most tables in a large set are never opened.
struct LazyTable {
    path: PathBuf,
    kind: Kind,
    layout: Layout,
    table: OnceLock<Option<Box<Table>>>,
}

impl LazyTable {
    fn new(path: PathBuf, kind: Kind, layout: Layout) -> Self {
        Self {
            path,
            kind,
            layout,
            table: OnceLock::new(),
        }
    }

    fn table(&self) -> Option<&Table> {
        self.table
            .get_or_init(|| match Table::open(&self.path, self.kind, &self.layout) {
                Ok(table) => Some(Box::new(table)),
                Err(e) => {
                    println!(
                        "info string Could not read tablebase file {}: {e}",
                        self.path.display()
                    );
                    None
                }
            })
            .as_deref()
    }
}

struct Entry {
    /// The material with the table's first side as white.
    material: Material,
    symmetric: bool,
    wdl: LazyTable,
    dtz: Option<LazyTable>,
}

impl Entry {
    fn view(&self, material: Material, b: &Board) -> View {
        let stm = b.side_to_move();

        if self.symmetric {
            // The side to move is always viewed as the table's first side
            View {
                swap_colours: stm == Color::BLACK,
                stm: Color::WHITE,
                symmetric: true,
            }
        } else {
            let swap_colours = material != self.material;
            View {
                swap_colours,
                stm: if swap_colours { !stm } else { stm },
                symmetric: false,
            }
        }
    }
}

/// A set of tables, each opened on its first probe. Dropping the set unmaps them, so the engines
/// probing it share it, and it outlives every search through them.
#[derive(Default)]
pub struct Tablebases {
    entries: Vec<Entry>,
    /// Entry index by material, both as stored and colour swapped.
    by_material: HashMap<Material, usize, BuildNoHashHasher<Material>>,
    max_pieces: usize,
}

impl Tablebases {
    /// The tables found in a path list (`:`-separated, `;` on Windows), or `None` for an empty
    /// path. WDL files define the tables; a table's DTZ file is optional. A table found in more
    /// than one directory is taken from the first.
    #[must_use]
    pub fn load(paths: &str) -> Option<Self> {
        if matches!(paths, "" | "<empty>") {
            return None;
        }

        let mut set = Self::default();
        let dirs: Vec<PathBuf> = env::split_paths(paths).collect();

        for dir in &dirs {
            let Ok(files) = fs::read_dir(dir) else {
                println!(
                    "info string Could not read tablebase directory {}",
                    dir.display()
                );
                continue;
            };

            for path in files.filter_map(|f| Some(f.ok()?.path())) {
                set.add(&path, &dirs);
            }
        }

        Some(set)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Pieces in the largest table, 0 when there are none.
    pub fn max_pieces(&self) -> usize {
        self.max_pieces
    }

    pub fn probe_wdl_table(&self, b: &Board) -> Option<Outcome> {
        let material = Material::of(b);

        // K v K has no table
        if material.is_bare_kings() {
            return Some(Outcome::Draw);
        }

        let entry = self.get(material)?;
        entry.wdl.table()?.wdl(b, entry.view(material, b))
    }

    pub fn probe_dtz_table(&self, b: &Board, outcome: Outcome) -> Option<DtzEntry> {
        let material = Material::of(b);
        let entry = self.get(material)?;

        entry
            .dtz
            .as_ref()?
            .table()?
            .dtz(b, entry.view(material, b), outcome)
    }

    /// Adds the table of a WDL file, unless it's already present.
    fn add(&mut self, wdl: &Path, dirs: &[PathBuf]) -> Option<()> {
        if wdl.extension()? != "rtbw" {
            return None;
        }

        let name = wdl.file_stem()?.to_str()?;
        let material = Material::from_name(name)?;

        if self.by_material.contains_key(&material) || !is_complete(wdl) {
            return None;
        }

        let dtz = dirs
            .iter()
            .map(|dir| dir.join(format!("{name}.rtbz")))
            .find(|path| is_complete(path));

        let layout = material.layout();
        let index = self.entries.len();

        self.by_material.insert(material, index);
        self.by_material.insert(material.swapped(), index);
        self.max_pieces = self.max_pieces.max(layout.pieces());

        self.entries.push(Entry {
            material,
            symmetric: material == material.swapped(),
            wdl: LazyTable::new(wdl.to_path_buf(), Kind::Wdl, layout),
            dtz: dtz.map(|path| LazyTable::new(path, Kind::Dtz, layout)),
        });

        Some(())
    }

    fn get(&self, material: Material) -> Option<&Entry> {
        self.by_material.get(&material).map(|&i| &self.entries[i])
    }
}

/// Table files are padded and then end in a trailer, so other sizes are truncated.
fn is_complete(path: &Path) -> bool {
    match fs::metadata(path) {
        Ok(meta) if meta.len() % DATA_ALIGN as u64 == TRAILER_LEN as u64 => true,
        Ok(_) => {
            println!("info string Incomplete tablebase file {}", path.display());
            false
        }
        Err(_) => false,
    }
}
