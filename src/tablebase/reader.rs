//! A bounds-checked cursor over a mapped table file.

use std::fmt::{self, Display, Formatter};
use std::io;
use std::mem;

use bytemuck::Pod;

pub enum LoadError {
    Io(io::Error),
    BadMagic,
    Truncated,
    /// The file's contents contradict its name or the format.
    Corrupt(&'static str),
}

impl Display for LoadError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::BadMagic => write!(f, "not a table of this type"),
            Self::Truncated => write!(f, "truncated"),
            Self::Corrupt(what) => write!(f, "corrupt {what}"),
        }
    }
}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

pub type LoadResult<T> = Result<T, LoadError>;

pub struct Cursor {
    data: &'static [u8],
    pos: usize,
}

impl Cursor {
    pub fn new(data: &'static [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn peek(&self) -> LoadResult<u8> {
        self.data.get(self.pos).copied().ok_or(LoadError::Truncated)
    }

    pub fn take(&mut self, len: usize) -> LoadResult<&'static [u8]> {
        let (taken, _) = self
            .rest()
            .split_at_checked(len)
            .ok_or(LoadError::Truncated)?;
        self.pos += len;
        Ok(taken)
    }

    pub fn take_array<const N: usize>(&mut self) -> LoadResult<[u8; N]> {
        let (&taken, _) = self
            .rest()
            .split_first_chunk::<N>()
            .ok_or(LoadError::Truncated)?;
        self.pos += N;
        Ok(taken)
    }

    pub fn take_pod<T: Pod>(&mut self) -> LoadResult<&'static T> {
        bytemuck::try_from_bytes(self.take(mem::size_of::<T>())?)
            .map_err(|_| LoadError::Corrupt("header"))
    }

    /// Everything from here to the end of the file.
    pub fn rest(&self) -> &'static [u8] {
        self.data.get(self.pos..).unwrap_or_default()
    }

    pub fn skip(&mut self, len: usize) -> LoadResult<()> {
        self.take(len).map(|_| ())
    }

    pub fn align(&mut self, to: usize) -> LoadResult<()> {
        let aligned = self
            .pos
            .checked_next_multiple_of(to)
            .ok_or(LoadError::Truncated)?;
        self.skip(aligned - self.pos)
    }
}
