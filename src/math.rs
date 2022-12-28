pub fn softmax(arr: &mut [f32]) {
    let max = max(arr);
    let mut s = 0.;

    for x in arr.iter_mut() {
        *x = fastapprox::faster::exp(*x - max);
        s += *x;
    }
    for x in arr.iter_mut() {
        *x /= s;
    }
}

fn max(arr: &[f32]) -> f32 {
    let mut max = std::f32::NEG_INFINITY;
    for x in arr.iter() {
        max = max.max(*x);
    }
    max
}
