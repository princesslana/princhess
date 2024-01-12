use std::ops::Not;

#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color(bool);

impl Color {
    pub const WHITE: Color = Color(false);
    pub const BLACK: Color = Color(true);

    pub const COUNT: usize = 2;

    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

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
