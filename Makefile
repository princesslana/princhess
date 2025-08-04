# Princhess Build and Test Makefile

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build         - Build release binary"
	@echo "  native        - Build with native CPU optimizations"
	@echo "  clean         - Clean all build artifacts"
	@echo ""
	@echo "Testing:"
	@echo "  sprt-gain     - Start SPRT gain test (STC)"
	@echo "  sprt-gain-ltc - Start SPRT gain test (LTC)"
	@echo "  sprt-equal    - Start SPRT equal test (STC)"
	@echo "  elo-check     - Start ELO check test (STC)"
	@echo "  elo-check-25k - Start ELO check test (25k nodes)"
	@echo "  resume-test   - Resume previous test"
	@echo "  start-test    - Start custom test (requires TYPE, TC, ENGINE1, ENGINE2)"
	@echo ""
	@echo "Tuning:"
	@echo "  tune          - Run parameter tuning"
	@echo ""
	@echo "Examples:"
	@echo "  make sprt-gain"
	@echo "  make start-test TYPE=sprt_gain TC=ltc ENGINE1=princhess ENGINE2=princhess-main"
	@echo "  make resume-test"

# Build targets
.PHONY: build
build:
	cargo build --release

.PHONY: native
native:
	cargo rustc --release --bin princhess -- -C target-cpu=native

.PHONY: clean
clean:
	cargo clean
	rm -rf target/fastchess target/tuning

# Test targets
.PHONY: sprt-gain
sprt-gain:
	@scripts/start-test.sh sprt_gain stc princhess princhess-main

.PHONY: sprt-gain-ltc
sprt-gain-ltc:
	@scripts/start-test.sh sprt_gain ltc princhess princhess-main

.PHONY: sprt-equal
sprt-equal:
	@scripts/start-test.sh sprt_equal stc princhess princhess-main

.PHONY: elo-check
elo-check:
	@scripts/start-test.sh elo_check stc princhess princhess-main

.PHONY: elo-check-25k
elo-check-25k:
	@scripts/start-test.sh elo_check nodes25k princhess princhess-main

.PHONY: resume-test
resume-test:
	@scripts/resume-test.sh

.PHONY: start-test
start-test:
	@if [ -z "$(TYPE)" ] || [ -z "$(TC)" ] || [ -z "$(ENGINE1)" ] || [ -z "$(ENGINE2)" ]; then \
		echo "Usage: make start-test TYPE=<type> TC=<tc> ENGINE1=<engine1> ENGINE2=<engine2>"; \
		echo "  TYPE: sprt_gain, sprt_equal, elo_check"; \
		echo "  TC: stc, ltc, nodes25k"; \
		echo "  ENGINE1, ENGINE2: princhess, princhess-main, etc"; \
		exit 1; \
	fi
	@scripts/start-test.sh $(TYPE) $(TC) $(ENGINE1) $(ENGINE2)

# Tuning targets
.PHONY: tune
tune:
	docker compose run --rm tune