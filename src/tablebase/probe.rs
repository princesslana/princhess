use std::cmp::Ordering;
use std::ops::Neg;

use crate::chess::{Board, Move, Piece};
use crate::engine::KNOWN_WIN;
use crate::tablebase::fathom::{self, DtzEntry};
use crate::tablebase::{Outcome, RootProbe};

/// Root rank of a win (or, negated, a loss) that holds whatever happens to the 50-move counter.
const CERTAIN_RANK: i32 = 1000;
/// Ranks from here up to `CERTAIN_RANK` still win under the 50-move rule, with less to spare.
const WINNING_RANK: i32 = CERTAIN_RANK - 100;

struct WdlProbe {
    outcome: Outcome,
    /// The outcome comes from a capture (or en passant) rather than the table, so the DTZ
    /// table entry for this position is not needed.
    zeroing_best: bool,
}

impl Outcome {
    /// DTZ of a position whose best move zeroes the 50-move counter. Cursed results sit past
    /// the 50-move limit.
    fn zeroing_dtz(self) -> i32 {
        match self {
            Self::Win => 1,
            Self::CursedWin => 101,
            Self::Draw => 0,
            Self::BlessedLoss => -101,
            Self::Loss => -1,
        }
    }
}

/// Resolves captures (a table entry is only reliable where no capture is at least as good)
/// and en passant (the tables assume no en passant rights).
pub fn probe_wdl(b: &Board) -> Option<Outcome> {
    probe_wdl_full(b).map(|p| p.outcome)
}

/// The root moves that keep the best result, and its score. Wins that hold under the 50-move
/// rule all rank equally unless the counter (or a repetition) makes them urgent, in which case
/// the fastest zeroing move ranks highest.
pub fn probe_root(b: &Board, has_repeated: bool) -> Option<RootProbe> {
    let cnt50 = i32::from(b.halfmove_clock());

    let ranked = b
        .legal_moves()
        .into_iter()
        .map(|mv| {
            let child = after(b, mv);

            // Counting one ply for the move itself.
            let dtz = if child.halfmove_clock() == 0 {
                (-probe_wdl(&child)?).zeroing_dtz()
            } else {
                let dtz = -probe_dtz(&child)?;
                dtz + dtz.signum()
            };

            let dtz = if dtz == 2 && is_mate(&child) { 1 } else { dtz };

            Some((mv, root_rank(dtz, cnt50, has_repeated)))
        })
        .collect::<Option<Vec<_>>>()?;

    let best = ranked.iter().map(|&(_, rank)| rank).max()?;

    let score = if best.abs() >= WINNING_RANK {
        i64::from(best) * KNOWN_WIN / i64::from(CERTAIN_RANK)
    } else {
        0
    };

    Some(RootProbe {
        moves: ranked
            .into_iter()
            .filter(|&(_, rank)| rank == best)
            .map(|(mv, _)| mv)
            .collect(),
        score,
    })
}

/// Wins inside the 50-move limit rank `CERTAIN_RANK`, or less by the plies they need once the
/// counter is high or the root has repeated. Losses rank `-CERTAIN_RANK` unless a 50-move draw
/// is in sight, and then the longest resistance ranks highest.
fn root_rank(dtz: i32, cnt50: i32, has_repeated: bool) -> i32 {
    match dtz.cmp(&0) {
        Ordering::Greater if dtz + cnt50 <= 99 && !has_repeated => CERTAIN_RANK,
        Ordering::Greater => CERTAIN_RANK - (dtz + cnt50),
        Ordering::Less if -dtz * 2 + cnt50 < 100 => -CERTAIN_RANK,
        Ordering::Less => -CERTAIN_RANK + (-dtz + cnt50),
        Ordering::Equal => 0,
    }
}

fn after(b: &Board, mv: Move) -> Board {
    let mut child = *b;
    child.make_move(mv);
    child
}

fn is_capture(b: &Board, mv: Move) -> bool {
    mv.is_enpassant() || b.by_color(!b.side_to_move()).contains(mv.to())
}

fn is_mate(b: &Board) -> bool {
    b.is_check() && !b.is_legal_move()
}

/// Captures-only alpha-beta with the table outcome as stand-pat. Terminates as each capture
/// drops to a smaller table. Never sees en passant: every position here follows a capture.
fn probe_ab(b: &Board, mut alpha: Outcome, beta: Outcome) -> Option<Outcome> {
    for mv in b.legal_moves() {
        if !is_capture(b, mv) {
            continue;
        }

        let v = -probe_ab(&after(b, mv), -beta, -alpha)?;

        if v > alpha {
            if v >= beta {
                return Some(v);
            }
            alpha = v;
        }
    }

    Some(alpha.max(fathom::probe_wdl_table(b)?))
}

fn probe_wdl_full(b: &Board) -> Option<WdlProbe> {
    let moves = b.legal_moves();

    // The best capture other than en passant, and the best en passant capture if better still.
    let mut best_capture: Option<Outcome> = None;
    let mut best_ep: Option<Outcome> = None;

    for &mv in &moves {
        if !is_capture(b, mv) {
            continue;
        }

        let beta = best_capture.map_or(Outcome::Win, Neg::neg);
        let v = -probe_ab(&after(b, mv), Outcome::Loss, beta)?;

        if Some(v) > best_capture {
            if v == Outcome::Win {
                return Some(WdlProbe {
                    outcome: v,
                    zeroing_best: true,
                });
            }

            if !mv.is_enpassant() {
                best_capture = Some(v);
            } else if Some(v) > best_ep {
                best_ep = Some(v);
            }
        }
    }

    let table = fathom::probe_wdl_table(b)?;

    if let Some(ep) = best_ep.filter(|&ep| Some(ep) > best_capture) {
        if ep > table {
            return Some(WdlProbe {
                outcome: ep,
                zeroing_best: true,
            });
        }
        best_capture = Some(ep);
    }

    if let Some(capture) = best_capture.filter(|&capture| capture >= table) {
        return Some(WdlProbe {
            outcome: capture,
            zeroing_best: capture > Outcome::Draw,
        });
    }

    // The table scores the position without en passant rights. If en passant is the only legal
    // move, that position is stalemate (a table draw), but here the capture is forced.
    let forced_ep = best_ep.filter(|_| {
        table == Outcome::Draw && !b.is_check() && moves.iter().all(|m| m.is_enpassant())
    });

    if let Some(ep) = forced_ep {
        return Some(WdlProbe {
            outcome: ep,
            zeroing_best: true,
        });
    }

    Some(WdlProbe {
        outcome: table,
        zeroing_best: false,
    })
}

/// Distance to zeroing the 50-move counter, in plies from the side to move: positive for a
/// win, negative for a loss, 0 for a draw and -1 when mated. Values beyond ±100 are cursed
/// wins and blessed losses. Can be one ply short (a win in n + 1 reported as n), except in
/// tables with positions exactly at the 50-move limit, so a win is only certain while
/// dtz + counter stays below 100.
fn probe_dtz(b: &Board) -> Option<i32> {
    let WdlProbe {
        outcome,
        zeroing_best,
    } = probe_wdl_full(b)?;

    if outcome == Outcome::Draw {
        return Some(0);
    }

    if zeroing_best {
        return Some(outcome.zeroing_dtz());
    }

    let moves = b.legal_moves();
    let winning = outcome > Outcome::Draw;

    if winning {
        for &mv in &moves {
            if b.piece_at(mv.from()) != Piece::PAWN || is_capture(b, mv) {
                continue;
            }

            if -probe_wdl(&after(b, mv))? == outcome {
                return Some(outcome.zeroing_dtz());
            }
        }
    }

    // The best move is not en passant, so the table's outcome (which ignores en passant) is
    // this position's, and its DTZ entry applies.
    if let DtzEntry::Value(dtz) = fathom::probe_dtz_table(b, outcome)? {
        return Some(outcome.zeroing_dtz() + if winning { dtz } else { -dtz });
    }

    // The table stores only the other side to move, so take the best non-zeroing move. Each
    // such child keeps the material and pawns with the other side to move, so its own entry
    // is stored and this recursion is one level deep. A losing side's zeroing moves are
    // already counted by the starting value.
    let mut best = (!winning).then(|| outcome.zeroing_dtz());

    for &mv in &moves {
        if is_capture(b, mv) || b.piece_at(mv.from()) == Piece::PAWN {
            continue;
        }

        let child = after(b, mv);
        let v = -probe_dtz(&child)?;

        if v == 1 && is_mate(&child) {
            best = Some(1);
        } else if winning {
            if v > 0 && best.is_none_or(|best| v + 1 < best) {
                best = Some(v + 1);
            }
        } else {
            best = best.map(|best| best.min(v - 1));
        }
    }

    // `None` only for a win without a winning move, which consistent tables never give.
    best
}
