use std::ops::{Index, IndexMut, Not};

#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color(bool);

impl Color {
    pub const WHITE: Color = Color(false);
    pub const BLACK: Color = Color(true);

    pub const COUNT: usize = 2;

    pub fn fold<T>(self, white: T, black: T) -> T {
        if self.0 {
            black
        } else {
            white
        }
    }
}

impl From<Color> for u8 {
    fn from(c: Color) -> Self {
        u8::from(c.0)
    }
}

impl From<bool> for Color {
    fn from(b: bool) -> Self {
        Self(b)
    }
}

impl From<u8> for Color {
    fn from(b: u8) -> Self {
        Self(b != 0)
    }
}

impl Not for Color {
    type Output = Color;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl<T> Index<Color> for [T; 2] {
    type Output = T;

    fn index(&self, index: Color) -> &Self::Output {
        &self[usize::from(index.0)]
    }
}

impl<T> IndexMut<Color> for [T; 2] {
    fn index_mut(&mut self, index: Color) -> &mut Self::Output {
        &mut self[usize::from(index.0)]
    }
}
