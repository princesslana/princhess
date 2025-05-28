
## Code Style

  * Always run cargo fmt after any code changes
  * Use `cargo clippy` to check for common mistakes and code quality issues
  * Follow the principles of KISS (Keep It Simple, Stupid), YAGNI (You Aren't Gonna Need It),
    and DRY (Don't Repeat Yourself) 
  * Code Comments are:
    * Ideally avoided, in preference of self-documenting code
    * Used to explain why something is done, not what is done

## Commit Messages

  * Princhess uses emoji log for commit messages. This means commit messages should be prefixed by:
    * 📦 NEW: when you add something entirely new
    * 👌 IMP: when you improve/enhance a piece of code, like refactoring
    * 🐛 FIX: when you fix a bug
    * 📖 DOC: when you add or update documentation
    * 🤖 TST: when you add or update tests
  * Commit messages should be:
      * Imperative. Write commit messages as if you are giving an order,
        e.g. "Add new feature" instead of "Added new feature".
      * Actions. Make commit messages based on the actions you take
      * Concise. Keep commit messages short and to the point, ideally under 50 characters.
