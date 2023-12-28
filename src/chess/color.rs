use std::ops::Not;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color(bool);

impl Color {
    pub const WHITE: Color = Color(false);
    pub const BLACK: Color = Color(true);

    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    pub fn fold<T>(self, white: T, black: T) -> T {
        if self.0 {
            black
        } else {
            white
        }
    }
}

impl From<shakmaty::Color> for Color {
    fn from(color: shakmaty::Color) -> Self {
        match color {
            shakmaty::Color::Black => Color::BLACK,
            shakmaty::Color::White => Color::WHITE,
        }
    }
}

impl From<Color> for shakmaty::Color {
    fn from(color: Color) -> Self {
        match color {
            Color::BLACK => shakmaty::Color::Black,
            Color::WHITE => shakmaty::Color::White,
        }
    }
}

impl Not for Color {
    type Output = Color;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}
