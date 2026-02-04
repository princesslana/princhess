# Princhess Build and Test Makefile

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build         - Build release binary"
	@echo "  native        - Build with native CPU optimizations"
	@echo "  clean         - Clean all build artifacts"
	@echo ""
	@echo "Lichess Bots:"
	@echo "  lichess-bot        - Build and run lichess bot (requires LICHESS_TOKEN)"
	@echo "  lichess-bot-policy - Build and run policy-only lichess bot (requires LICHESS_TOKEN)"
	@echo ""
	@echo "Testing:"
	@echo "  sprt-gain     - Start SPRT gain test (STC)"
	@echo "  sprt-gain-ltc - Start SPRT gain test (LTC)"
	@echo "  sprt-gain-2t  - Start SPRT gain test (STC, 2 threads)"
	@echo "  sprt-equal    - Start SPRT equal test (STC)"
	@echo "  sprt-equal-2t - Start SPRT equal test (STC, 2 threads)"
	@echo "  elo-check     - Start ELO check test (STC)"
	@echo "  elo-check-25k - Start ELO check test (25k nodes)"
	@echo "  resume-test   - Resume previous test"
	@echo "  start-test    - Start custom test (requires TYPE, TC, ENGINE1, ENGINE2)"
	@echo ""
	@echo "Examples:"
	@echo "  make lichess-bot LICHESS_TOKEN=lip_xxxxxxxxxx"
	@echo "  make sprt-gain"
	@echo "  make start-test TYPE=sprt_gain TC=ltc ENGINE1=princhess ENGINE2=princhess-main"

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

# Lichess bot targets
.PHONY: lichess-bot
lichess-bot:
	@if [ -z "$(LICHESS_TOKEN)" ]; then echo "LICHESS_TOKEN is required. Usage: make lichess-bot LICHESS_TOKEN=your_token"; exit 1; fi
	docker build -f bin/Dockerfile.lichess -t princhess-lichess .
	docker run -e LICHESS_TOKEN=$(LICHESS_TOKEN) -v $$(pwd)/bin/config.yml:/src/config.yml:ro princhess-lichess

.PHONY: lichess-bot-policy
lichess-bot-policy:
	@if [ -z "$(LICHESS_TOKEN)" ]; then echo "LICHESS_TOKEN is required. Usage: make lichess-bot-policy LICHESS_TOKEN=your_token"; exit 1; fi
	docker build -f bin/Dockerfile.lichess -t princhess-lichess .
	docker run -e LICHESS_TOKEN=$(LICHESS_TOKEN) -v $$(pwd)/bin/config.policy.yml:/src/config.yml:ro princhess-lichess

# Test targets
.PHONY: sprt-gain
sprt-gain:
	@$(MAKE) start-test TYPE=sprt_gain TC=stc ENGINE1=princhess ENGINE2=princhess-main

.PHONY: sprt-gain-ltc
sprt-gain-ltc:
	@$(MAKE) start-test TYPE=sprt_gain TC=ltc ENGINE1=princhess ENGINE2=princhess-main

.PHONY: sprt-gain-2t
sprt-gain-2t:
	@$(MAKE) start-test TYPE=sprt_gain TC=stc ENGINE1=princhess ENGINE2=princhess-main THREADS=2

.PHONY: sprt-equal
sprt-equal:
	@$(MAKE) start-test TYPE=sprt_equal TC=stc ENGINE1=princhess ENGINE2=princhess-main

.PHONY: sprt-equal-2t
sprt-equal-2t:
	@$(MAKE) start-test TYPE=sprt_equal TC=stc ENGINE1=princhess ENGINE2=princhess-main THREADS=2

.PHONY: elo-check
elo-check:
	@$(MAKE) start-test TYPE=elo_check TC=stc ENGINE1=princhess ENGINE2=princhess-main

.PHONY: elo-check-25k
elo-check-25k:
	@$(MAKE) start-test TYPE=elo_check TC=nodes25k ENGINE1=princhess ENGINE2=princhess-main

.PHONY: resume-test
resume-test:
	@scripts/resume-test.sh

.PHONY: start-test
start-test:
	@if [ -z "$(TYPE)" ] || [ -z "$(TC)" ] || [ -z "$(ENGINE1)" ] || [ -z "$(ENGINE2)" ]; then \
		echo "Usage: make start-test TYPE=<type> TC=<tc> ENGINE1=<engine1> ENGINE2=<engine2> [THREADS=<n>] [MAX_CORES=<n>]"; \
		echo "  TYPE: sprt_gain, sprt_equal, elo_check"; \
		echo "  TC: stc, ltc, nodes25k"; \
		echo "  ENGINE1, ENGINE2: princhess, princhess-main, etc"; \
		echo "  THREADS: threads per game (optional, default: 1)"; \
		echo "  MAX_CORES: max cores available (optional, auto-detected)"; \
		exit 1; \
	fi
	@scripts/start-test.sh --test-type $(TYPE) --tc $(TC) --engine1 $(ENGINE1) --engine2 $(ENGINE2) $(if $(THREADS),--threads $(THREADS)) $(if $(MAX_CORES),--max-cores $(MAX_CORES))
