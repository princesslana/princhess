use wide::f32x8;

pub fn f32x8_from_slice_with_padding(src: &[f32], pad: f32) -> f32x8 {
    match src.len() {
        8 => f32x8::from([
            src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
        ]),
        7 => f32x8::from([src[0], src[1], src[2], src[3], src[4], src[5], src[6], pad]),
        6 => f32x8::from([src[0], src[1], src[2], src[3], src[4], src[5], pad, pad]),
        5 => f32x8::from([src[0], src[1], src[2], src[3], src[4], pad, pad, pad]),
        4 => f32x8::from([src[0], src[1], src[2], src[3], pad, pad, pad, pad]),
        3 => f32x8::from([src[0], src[1], src[2], pad, pad, pad, pad, pad]),
        2 => f32x8::from([src[0], src[1], pad, pad, pad, pad, pad, pad]),
        1 => f32x8::from([src[0], pad, pad, pad, pad, pad, pad, pad]),
        0 => f32x8::splat(pad),
        _ => unreachable!(),
    }
}
