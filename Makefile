

rule:
	cargo build --release
	mv target/release/princhess princhess
