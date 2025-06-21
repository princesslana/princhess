
## Code Style

  * Always run cargo fmt after any code changes
  * Use `cargo clippy` to check for common mistakes and code quality issues
  * Follow the principles of KISS (Keep It Simple, Stupid), YAGNI (You Aren't Gonna Need It),
    and DRY (Don't Repeat Yourself) 
  * Prioritize self-documenting code. When comments are necessary, use them to
    explain why the code is as it is, not what it does. For public APIs,
    use Rustdoc comments (///) to explain usage and behavior
  * Follow Rust's standard naming conventions (e.g., snake_case for functions,
    PascalCase for types, SCREAMING_SNAKE_CASE for constants).
  * When using unsafe blocks, ensure they are minimal, well-justified, and accompanied
    by comments explaining the invariants they rely on.

## Commit Messages

  * Princhess uses emoji log for commit messages. This means commit messages should be prefixed by:
    * 📦 NEW: when you add something entirely new
    * 👌 IMP: when you improve/enhance a piece of code, like refactoring
    * 🐛 FIX: when you fix a bug
    * 📖 DOC: when you add or update documentation (including code comments)
    * 🤖 TST: when you add or update tests
  * Commit messages should be:
      * Imperative and Action-Oriented. Write commit messages as if you are giving an order,
        focusing on the action performed by the commit (e.g., 'Add', 'Fix', 'Refactor').
      * Concise. Keep commit messages short and to the point, ideally under 50 characters.
