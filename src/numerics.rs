use std::cmp;

// Based upon https://github.com/rust-ndarray/ndarray/blob/0.15.4/src/numeric_util.rs#L61-L110
// Assumes xs.len() == ys.len() and len is a multiple of 8
pub fn unrolled_dot(xs: &[f32], ys: &[f32]) -> f32 {
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];
    let mut sum = 0.;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0., 0., 0., 0., 0., 0., 0., 0.);
    while xs.len() >= 8 {
        p0 = p0 + xs[0] * ys[0];
        p1 = p1 + xs[1] * ys[1];
        p2 = p2 + xs[2] * ys[2];
        p3 = p3 + xs[3] * ys[3];
        p4 = p4 + xs[4] * ys[4];
        p5 = p5 + xs[5] * ys[5];
        p6 = p6 + xs[6] * ys[6];
        p7 = p7 + xs[7] * ys[7];

        xs = &xs[8..];
        ys = &ys[8..];
    }
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);

    sum
}
