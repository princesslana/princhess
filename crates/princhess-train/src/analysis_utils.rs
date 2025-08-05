pub fn king_bucket_name(bucket: usize) -> &'static str {
    match bucket {
        0 => "Corner",
        1 => "Center",
        2 => "Other",
        _ => "Unknown",
    }
}

pub fn threat_bucket_name(bucket: usize) -> &'static str {
    match bucket {
        0 => "Safe",
        1 => "Defended",
        2 => "Threatened",
        3 => "Both",
        _ => "Unknown",
    }
}

pub fn piece_name(piece_idx: usize) -> &'static str {
    match piece_idx {
        0 => "pawn",
        1 => "knight",
        2 => "bishop",
        3 => "rook",
        4 => "queen",
        5 => "king",
        _ => "unknown",
    }
}
