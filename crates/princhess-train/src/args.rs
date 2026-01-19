use std::env;
use std::str::FromStr;

pub struct Args {
    args: Vec<String>,
    index: usize,
}

impl Args {
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            args: env::args().skip(1).collect(),
            index: 0,
        }
    }

    /// Parse optional flag: --flag value or -f value
    /// Removes the flag and its value from args if found
    ///
    /// # Panics
    /// Panics if flag is present but has no value or value cannot be parsed
    pub fn flag<T: FromStr>(&mut self, short: &str, long: &str) -> Option<T> {
        let mut i = 0;
        while i < self.args.len() {
            if self.args[i] == short || self.args[i] == long {
                let value = self
                    .args
                    .get(i + 1)
                    .unwrap_or_else(|| panic!("{short}/{long} requires a value"))
                    .parse()
                    .unwrap_or_else(|_| panic!("Invalid value for {short}/{long}"));
                self.args.remove(i);
                self.args.remove(i);
                return Some(value);
            }
            i += 1;
        }
        None
    }

    /// Get next positional argument
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<String> {
        if self.index < self.args.len() {
            let arg = self.args[self.index].clone();
            self.index += 1;
            Some(arg)
        } else {
            None
        }
    }

    /// Expect a positional argument or panic
    ///
    /// # Panics
    /// Panics if no argument is available
    pub fn expect(&mut self, name: &str) -> String {
        self.next()
            .unwrap_or_else(|| panic!("Missing required argument: {name}"))
    }
}
