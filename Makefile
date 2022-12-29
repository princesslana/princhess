
rule:
	cargo rustc --release -- -D warnings
	cargo clippy -- -D warnings
