use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::time::Duration;

use super::command::{GoParams, PositionSpec, SetOptionParams, UciCommand};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    InvalidPosition,
    InvalidFen,
    MissingOptionName,
    MalformedCommand,
}

impl Display for ParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPosition => write!(f, "invalid position specification"),
            Self::InvalidFen => {
                write!(f, "invalid FEN field count (must be between 4 and 6 fields)")
            }
            Self::MissingOptionName => write!(f, "missing or malformed setoption name"),
            Self::MalformedCommand => write!(f, "malformed command arguments"),
        }
    }
}

impl Error for ParseError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpan<'a> {
    pub start: usize,
    pub end: usize,
    pub text: &'a str,
}

// zero-copy tokenizer spitting out byte spans directly over the line
pub struct Tokenizer<'a> {
    source: &'a str,
    indices: std::str::CharIndices<'a>,
    current_char: Option<(usize, char)>,
}

impl<'a> Tokenizer<'a> {
    #[must_use]
    pub fn new(source: &'a str) -> Self {
        let mut indices = source.char_indices();
        let current_char = indices.next();
        Self {
            source,
            indices,
            current_char,
        }
    }
}

impl<'a> Iterator for Tokenizer<'a> {
    type Item = TokenSpan<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // skip leading whitespace
        while let Some((_, ch)) = self.current_char {
            if !ch.is_whitespace() {
                break;
            }
            self.current_char = self.indices.next();
        }

        let (start, _) = self.current_char?;

        // find token end
        while let Some((idx, ch)) = self.indices.next() {
            if ch.is_whitespace() {
                self.current_char = Some((idx, ch));
                return Some(TokenSpan {
                    start,
                    end: idx,
                    text: &self.source[start..idx],
                });
            }
        }

        // reached end of line
        self.current_char = None;
        let end = self.source.len();
        Some(TokenSpan {
            start,
            end,
            text: &self.source[start..end],
        })
    }
}

impl<'a> UciCommand<'a> {
    // top-level parser dispatching lines into typed commands
    pub fn parse(line: &'a str) -> Result<Option<Self>, ParseError> {
        let mut tokens = Tokenizer::new(line);
        let Some(first) = tokens.next() else {
            return Ok(None);
        };

        match first.text {
            "uci" => Ok(Some(Self::Uci)),
            "debug" => {
                let on = match tokens.next().map(|t| t.text) {
                    Some("on") => true,
                    Some("off") => false,
                    _ => return Err(ParseError::MalformedCommand),
                };
                Ok(Some(Self::Debug(on)))
            }
            "isready" => Ok(Some(Self::IsReady)),
            "setoption" => parse_setoption(line, &mut tokens).map(Some),
            "ucinewgame" => Ok(Some(Self::UciNewGame)),
            "position" => parse_position(line, &mut tokens).map(Some),
            "quit" => Ok(Some(Self::Quit)),
            "stop" => Ok(Some(Self::Stop)),
            "ponderhit" => Ok(Some(Self::PonderHit)),
            "go" => Ok(Some(Self::Go(parse_go(&mut tokens)))),
            "movelist" => Ok(Some(Self::MoveList(tokens.map(|t| t.text).collect()))),
            "sizelist" => Ok(Some(Self::SizeList)),
            "eval" => Ok(Some(Self::Eval)),
            "bench" => Ok(Some(Self::Bench)),
            "randomopen" => Ok(Some(Self::RandomOpen)),
            "fingerprint" => Ok(Some(Self::Fingerprint)),
            unknown => Ok(Some(Self::UnknownCommand(unknown))),
        }
    }
}

// parse name and optional value for setoption, buttons got no value field
pub fn parse_setoption<'a, I>(line: &'a str, tokens: &mut I) -> Result<UciCommand<'a>, ParseError>
where
    I: Iterator<Item = TokenSpan<'a>>,
{
    let first = tokens.next().ok_or(ParseError::MissingOptionName)?;
    if !first.text.eq_ignore_ascii_case("name") {
        return Err(ParseError::MissingOptionName);
    }

    let mut name_start = None;
    let mut name_end = 0;
    let mut value_start = None;
    let mut value_end = 0;
    let mut parsing_value = false;

    for t in tokens {
        if !parsing_value && t.text.eq_ignore_ascii_case("value") {
            parsing_value = true;
            continue;
        }

        if parsing_value {
            if value_start.is_none() {
                value_start = Some(t.start);
            }
            value_end = t.end;
        } else {
            if name_start.is_none() {
                name_start = Some(t.start);
            }
            name_end = t.end;
        }
    }

    let name_start = name_start.ok_or(ParseError::MissingOptionName)?;
    let name = &line[name_start..name_end];
    let value = if parsing_value {
        match value_start {
            Some(start) => Some(&line[start..value_end]),
            None => Some(""),
        }
    } else {
        None
    };

    Ok(UciCommand::SetOption(SetOptionParams { name, value }))
}

// grab startpos or slice raw fen without allocating :)
pub fn parse_position<'a, I>(line: &'a str, tokens: &mut I) -> Result<UciCommand<'a>, ParseError>
where
    I: Iterator<Item = TokenSpan<'a>>,
{
    let spec_type = tokens.next().ok_or(ParseError::InvalidPosition)?;

    match spec_type.text {
        "startpos" => {
            let moves = match tokens.next() {
                Some(t) if t.text == "moves" => tokens.map(|t| t.text).collect(),
                None => Vec::new(),
                Some(_) => return Err(ParseError::InvalidPosition),
            };
            Ok(UciCommand::Position {
                spec: PositionSpec::Startpos,
                moves,
            })
        }
        "fen" => {
            let mut fen_start = None;
            let mut fen_end = 0;
            let mut fen_count = 0;
            let mut has_moves = false;

            for t in tokens.by_ref() {
                if t.text == "moves" {
                    has_moves = true;
                    break;
                }
                if fen_count >= 6 {
                    return Err(ParseError::InvalidFen);
                }
                if fen_start.is_none() {
                    fen_start = Some(t.start);
                }
                fen_end = t.end;
                fen_count += 1;
            }

            // fen needs 4 to 6 fields or it's bogus
            if fen_count < 4 || fen_count > 6 {
                return Err(ParseError::InvalidFen);
            }

            let fen_start = fen_start.ok_or(ParseError::InvalidFen)?;
            let fen_str = &line[fen_start..fen_end];
            let moves = if has_moves {
                tokens.map(|t| t.text).collect()
            } else {
                Vec::new()
            };

            Ok(UciCommand::Position {
                spec: PositionSpec::Fen(fen_str),
                moves,
            })
        }
        _ => Err(ParseError::InvalidPosition),
    }
}

// check if token is a go keyword so we don't accidentally consume it as a val
fn is_go_keyword(s: &str) -> bool {
    matches!(
        s,
        "infinite"
            | "ponder"
            | "movetime"
            | "wtime"
            | "btime"
            | "winc"
            | "binc"
            | "movestogo"
            | "nodes"
            | "depth"
            | "mate"
            | "searchmoves"
    )
}

// pull out numeric value if next token is a value (not another go keyword)
fn parse_param_val<'a, T: std::str::FromStr, I>(it: &mut std::iter::Peekable<I>) -> Option<T>
where
    I: Iterator<Item = TokenSpan<'a>>,
{
    if let Some(next_t) = it.peek() {
        if is_go_keyword(next_t.text) {
            return None;
        }
    }
    it.next().and_then(|t| t.text.parse::<T>().ok())
}

// pull duration in millis if next token is numeric
fn parse_duration_ms<'a, I>(it: &mut std::iter::Peekable<I>) -> Option<Duration>
where
    I: Iterator<Item = TokenSpan<'a>>,
{
    parse_param_val::<u64, I>(it).map(Duration::from_millis)
}

// pull out all time control args and search limits
pub fn parse_go<'a, I>(tokens: &mut I) -> GoParams
where
    I: Iterator<Item = TokenSpan<'a>>,
{
    let mut it = tokens.peekable();
    let mut params = GoParams::default();

    while let Some(t) = it.next() {
        match t.text {
            "infinite" => params.infinite = true,
            "ponder" => params.ponder = true,
            "movetime" => params.movetime = parse_duration_ms(&mut it),
            "wtime" => params.wtime = parse_duration_ms(&mut it),
            "btime" => params.btime = parse_duration_ms(&mut it),
            "winc" => params.winc = parse_duration_ms(&mut it),
            "binc" => params.binc = parse_duration_ms(&mut it),
            "movestogo" => params.movestogo = parse_param_val(&mut it),
            "nodes" => params.nodes = parse_param_val(&mut it),
            "depth" => params.depth = parse_param_val(&mut it),
            "mate" => params.mate = parse_param_val(&mut it),
            "searchmoves" => {
                // skip searchmoves move tokens until next keyword
                while let Some(next_t) = it.peek() {
                    if is_go_keyword(next_t.text) {
                        break;
                    }
                    it.next();
                }
            }
            _ => (),
        }
    }

    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty() {
        assert_eq!(UciCommand::parse(""), Ok(None));
        assert_eq!(UciCommand::parse("   \t  \n"), Ok(None));
    }

    #[test]
    fn test_parse_simple_commands() {
        assert_eq!(UciCommand::parse("uci"), Ok(Some(UciCommand::Uci)));
        assert_eq!(UciCommand::parse("debug on"), Ok(Some(UciCommand::Debug(true))));
        assert_eq!(UciCommand::parse("debug off"), Ok(Some(UciCommand::Debug(false))));
        assert_eq!(UciCommand::parse("isready"), Ok(Some(UciCommand::IsReady)));
        assert_eq!(
            UciCommand::parse("ucinewgame"),
            Ok(Some(UciCommand::UciNewGame))
        );
        assert_eq!(UciCommand::parse("quit"), Ok(Some(UciCommand::Quit)));
        assert_eq!(UciCommand::parse("stop"), Ok(Some(UciCommand::Stop)));
        assert_eq!(UciCommand::parse("ponderhit"), Ok(Some(UciCommand::PonderHit)));
        assert_eq!(
            UciCommand::parse("sizelist"),
            Ok(Some(UciCommand::SizeList))
        );
        assert_eq!(UciCommand::parse("eval"), Ok(Some(UciCommand::Eval)));
        assert_eq!(UciCommand::parse("bench"), Ok(Some(UciCommand::Bench)));
        assert_eq!(
            UciCommand::parse("randomopen"),
            Ok(Some(UciCommand::RandomOpen))
        );
        assert_eq!(
            UciCommand::parse("fingerprint"),
            Ok(Some(UciCommand::Fingerprint))
        );
    }

    #[test]
    fn test_parse_setoption() {
        assert_eq!(
            UciCommand::parse("setoption name Hash value 256"),
            Ok(Some(UciCommand::SetOption(SetOptionParams {
                name: "Hash",
                value: Some("256"),
            })))
        );

        assert_eq!(
            UciCommand::parse("setoption name Style Select value Solid"),
            Ok(Some(UciCommand::SetOption(SetOptionParams {
                name: "Style Select",
                value: Some("Solid"),
            })))
        );

        assert_eq!(
            UciCommand::parse("setoption name SyzygyPath value /path/to/tablebases with spaces"),
            Ok(Some(UciCommand::SetOption(SetOptionParams {
                name: "SyzygyPath",
                value: Some("/path/to/tablebases with spaces"),
            })))
        );

        assert_eq!(
            UciCommand::parse("setoption name Clear Hash"),
            Ok(Some(UciCommand::SetOption(SetOptionParams {
                name: "Clear Hash",
                value: None,
            })))
        );

        assert_eq!(
            UciCommand::parse("setoption value 128"),
            Err(ParseError::MissingOptionName)
        );

        assert_eq!(
            UciCommand::parse("setoption name"),
            Err(ParseError::MissingOptionName)
        );

        // case insensitive keyword test
        assert_eq!(
            UciCommand::parse("setoption Name Hash Value 128"),
            Ok(Some(UciCommand::SetOption(SetOptionParams {
                name: "Hash",
                value: Some("128"),
            })))
        );

        // empty value string test
        assert_eq!(
            UciCommand::parse("setoption name Hash value"),
            Ok(Some(UciCommand::SetOption(SetOptionParams {
                name: "Hash",
                value: Some(""),
            })))
        );

        // missing name followed immediately by value
        assert_eq!(
            UciCommand::parse("setoption name value 100"),
            Err(ParseError::MissingOptionName)
        );
    }

    #[test]
    fn test_parse_position() {
        assert_eq!(
            UciCommand::parse("position startpos"),
            Ok(Some(UciCommand::Position {
                spec: PositionSpec::Startpos,
                moves: Vec::new(),
            }))
        );

        assert_eq!(
            UciCommand::parse("position startpos moves e2e4 e7e5"),
            Ok(Some(UciCommand::Position {
                spec: PositionSpec::Startpos,
                moves: vec!["e2e4", "e7e5"],
            }))
        );

        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        let fen_line = format!("position fen {fen}");
        assert_eq!(
            UciCommand::parse(&fen_line),
            Ok(Some(UciCommand::Position {
                spec: PositionSpec::Fen(fen),
                moves: Vec::new(),
            }))
        );

        let fen_moves_line = format!("position fen {fen} moves e2e4");
        assert_eq!(
            UciCommand::parse(&fen_moves_line),
            Ok(Some(UciCommand::Position {
                spec: PositionSpec::Fen(fen),
                moves: vec!["e2e4"],
            }))
        );

        let fen_4part = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
        let fen_4part_line = format!("position fen {fen_4part}");
        assert_eq!(
            UciCommand::parse(&fen_4part_line),
            Ok(Some(UciCommand::Position {
                spec: PositionSpec::Fen(fen_4part),
                moves: Vec::new(),
            }))
        );

        let fen_4part_moves_line = format!("position fen {fen_4part} moves e2e4");
        assert_eq!(
            UciCommand::parse(&fen_4part_moves_line),
            Ok(Some(UciCommand::Position {
                spec: PositionSpec::Fen(fen_4part),
                moves: vec!["e2e4"],
            }))
        );

        // malformed fen counts (< 4)
        assert_eq!(
            UciCommand::parse("position fen 8/8/8 w -"),
            Err(ParseError::InvalidFen)
        );

        assert_eq!(
            UciCommand::parse("position invalid"),
            Err(ParseError::InvalidPosition)
        );

        assert_eq!(
            UciCommand::parse("position startpos invalid e2e4"),
            Err(ParseError::InvalidPosition)
        );

        // padded whitespace test
        assert_eq!(
            UciCommand::parse("  position   startpos   moves   e2e4  "),
            Ok(Some(UciCommand::Position {
                spec: PositionSpec::Startpos,
                moves: vec!["e2e4"],
            }))
        );

        // 7 fields before moves is invalid
        assert_eq!(
            UciCommand::parse("position fen 8/8/8/8/8/8/8/8 w - - 0 1 bad_seventh moves e2e4"),
            Err(ParseError::InvalidFen)
        );

        // 7 fields without moves is invalid
        assert_eq!(
            UciCommand::parse("position fen 8/8/8/8/8/8/8/8 w - - 0 1 bad_seventh"),
            Err(ParseError::InvalidFen)
        );

        // missing fen fields before moves is invalid
        assert_eq!(
            UciCommand::parse("position fen moves e2e4"),
            Err(ParseError::InvalidFen)
        );
    }

    #[test]
    fn test_parse_go() {
        assert_eq!(
            UciCommand::parse("go wtime 300000 btime 300000 movestogo 40"),
            Ok(Some(UciCommand::Go(GoParams {
                wtime: Some(Duration::from_millis(300000)),
                btime: Some(Duration::from_millis(300000)),
                movestogo: Some(40),
                ..GoParams::default()
            })))
        );

        assert_eq!(
            UciCommand::parse("go infinite"),
            Ok(Some(UciCommand::Go(GoParams {
                infinite: true,
                ..GoParams::default()
            })))
        );

        assert_eq!(
            UciCommand::parse("go ponder"),
            Ok(Some(UciCommand::Go(GoParams {
                ponder: true,
                ..GoParams::default()
            })))
        );

        assert_eq!(
            UciCommand::parse("go movetime 10000"),
            Ok(Some(UciCommand::Go(GoParams {
                movetime: Some(Duration::from_millis(10000)),
                ..GoParams::default()
            })))
        );

        assert_eq!(
            UciCommand::parse(
                "go wtime 30000 btime 20000 winc 1000 binc 500 movestogo 40 nodes 50000 depth 15 mate 3"
            ),
            Ok(Some(UciCommand::Go(GoParams {
                infinite: false,
                movetime: None,
                wtime: Some(Duration::from_millis(30000)),
                btime: Some(Duration::from_millis(20000)),
                winc: Some(Duration::from_millis(1000)),
                binc: Some(Duration::from_millis(500)),
                movestogo: Some(40),
                nodes: Some(50000),
                depth: Some(15),
                mate: Some(3),
                ponder: false,
            })))
        );

        // missing time value followed by keyword shouldn't consume subsequent keyword
        assert_eq!(
            UciCommand::parse("go wtime btime 50000"),
            Ok(Some(UciCommand::Go(GoParams {
                wtime: None,
                btime: Some(Duration::from_millis(50000)),
                ..GoParams::default()
            })))
        );

        // negative time value is consumed as invalid value, subsequent keyword parsed
        assert_eq!(
            UciCommand::parse("go wtime -1000 btime 60000"),
            Ok(Some(UciCommand::Go(GoParams {
                wtime: None,
                btime: Some(Duration::from_millis(60000)),
                ..GoParams::default()
            })))
        );

        // missing movetime followed by flag
        assert_eq!(
            UciCommand::parse("go movetime infinite"),
            Ok(Some(UciCommand::Go(GoParams {
                movetime: None,
                infinite: true,
                ..GoParams::default()
            })))
        );

        // searchmoves skips moves until next keyword
        assert_eq!(
            UciCommand::parse("go searchmoves e2e4 e7e5 wtime 30000"),
            Ok(Some(UciCommand::Go(GoParams {
                wtime: Some(Duration::from_millis(30000)),
                ..GoParams::default()
            })))
        );
    }

    #[test]
    fn test_parse_movelist() {
        assert_eq!(
            UciCommand::parse("movelist e2e4 e7e5 g1f3"),
            Ok(Some(UciCommand::MoveList(vec!["e2e4", "e7e5", "g1f3"])))
        );
    }

    #[test]
    fn test_parse_unknown_command() {
        assert_eq!(
            UciCommand::parse("foo bar baz"),
            Ok(Some(UciCommand::UnknownCommand("foo")))
        );
    }

    #[test]
    fn test_parse_error_display() {
        assert_eq!(
            ParseError::InvalidPosition.to_string(),
            "invalid position specification"
        );
        assert_eq!(
            ParseError::InvalidFen.to_string(),
            "invalid FEN field count (must be between 4 and 6 fields)"
        );
        assert_eq!(
            ParseError::MissingOptionName.to_string(),
            "missing or malformed setoption name"
        );
        assert_eq!(
            ParseError::MalformedCommand.to_string(),
            "malformed command arguments"
        );
    }
}
