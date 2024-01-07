use std::ops::Not;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color(bool);

impl Color {
    pub const WHITE: Color = Color(false);
    pub const BLACK: Color = Color(true);

    pub const COUNT: usize = 2;

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

impl From<bool> for Color {
    fn from(b: bool) -> Self {
        Self(b)
    }
}

impl Not for Color {
    type Output = Color;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}
