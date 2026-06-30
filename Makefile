# Princhess Build and Test Makefile

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build         - Build release binary (x86-64-v3)"
	@echo "  native        - Build with native CPU optimizations"
	@echo "  train         - Build training binaries with native CPU (NODEFAULTFEATURES=1 to skip default features)"
	@echo "  clean         - Clean all build artifacts"
	@echo "  bench         - Compare NPS: princhess vs princhess-main (ENGINE1, ENGINE2, RUNS)"
	@echo ""
	@echo "Lichess Bots:"
	@echo "  lichess-bot        - Build and run lichess bot (requires LICHESS_TOKEN)"
	@echo "  lichess-bot-policy - Build and run policy-only lichess bot (requires LICHESS_TOKEN)"
	@echo ""
	@echo "Testing:"
	@echo "  sprt-gain          - Start SPRT gain test (STC)"
	@echo "  sprt-gain-ltc      - Start SPRT gain test (LTC)"
	@echo "  sprt-gain-2t       - Start SPRT gain test (STC, 2 threads)"
	@echo "  sprt-equal         - Start SPRT equal test (STC)"
	@echo "  sprt-equal-ltc     - Start SPRT equal test (LTC)"
	@echo "  sprt-equal-2t      - Start SPRT equal test (STC, 2 threads)"
	@echo "  elo-check          - Start ELO check test (STC)"
	@echo "  elo-check-25k      - Start ELO check test (25k nodes)"
	@echo "  sprt-gain-dfrc     - Start SPRT gain test (STC, DFRC)"
	@echo "  sprt-gain-dfrc-ltc - Start SPRT gain test (LTC, DFRC)"
	@echo "  sprt-equal-dfrc    - Start SPRT equal test (STC, DFRC)"
	@echo "  sprt-equal-dfrc-ltc - Start SPRT equal test (LTC, DFRC)"
	@echo "  resume-test        - Resume previous test"
	@echo "  start-test         - Start custom test (requires TYPE, TC, ENGINE1, ENGINE2)"
	@echo ""
	@echo "Examples:"
	@echo "  make lichess-bot LICHESS_TOKEN=lip_xxxxxxxxxx"
	@echo "  make sprt-gain"
	@echo "  make start-test TYPE=sprt_gain TC=ltc ENGINE1=princhess ENGINE2=princhess-main"

# Build targets
BUILD_FEATURES = $(if $(NODEFAULTFEATURES),--no-default-features,)

PRINCHESS_TARGET_CPU ?= x86-64-v3

.PHONY: build
build:
	PRINCHESS_TARGET_CPU=$(PRINCHESS_TARGET_CPU) cargo rustc --release --bin princhess $(BUILD_FEATURES) -- -C target-cpu=$(PRINCHESS_TARGET_CPU)

.PHONY: native
native:
	$(MAKE) build PRINCHESS_TARGET_CPU=native

TRAIN_FEATURES = $(if $(NODEFAULTFEATURES),--no-default-features,--features princhess/default)

.PHONY: train
train:
	@if [ -z "$(FORCE)" ] && [ -z "$(NODEFAULTFEATURES)" ] && ! git diff --quiet main -- src/nets/*.bin 2>/dev/null; then \
		echo "WARNING: Net files differ from main branch!"; \
		echo "Use 'make train NODEFAULTFEATURES=1' to build without default features (e.g. after changing net struct size)."; \
		echo "Use 'make train FORCE=1' to build with default features anyway."; \
		exit 1; \
	fi
	RUSTFLAGS="-C target-cpu=native" cargo build --release --package princhess-train $(TRAIN_FEATURES)

.PHONY: bench
bench:
	@scripts/bench.sh $(or $(ENGINE1),princhess) $(or $(ENGINE2),princhess-main) $(or $(RUNS),10)

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

.PHONY: sprt-gain-dfrc
sprt-gain-dfrc:
	@$(MAKE) start-test TYPE=sprt_gain TC=stc ENGINE1=princhess ENGINE2=princhess-main VARIANT=dfrc

.PHONY: sprt-gain-dfrc-ltc
sprt-gain-dfrc-ltc:
	@$(MAKE) start-test TYPE=sprt_gain TC=ltc ENGINE1=princhess ENGINE2=princhess-main VARIANT=dfrc

.PHONY: sprt-equal-dfrc
sprt-equal-dfrc:
	@$(MAKE) start-test TYPE=sprt_equal TC=stc ENGINE1=princhess ENGINE2=princhess-main VARIANT=dfrc

.PHONY: sprt-equal-ltc
sprt-equal-ltc:
	@$(MAKE) start-test TYPE=sprt_equal TC=ltc ENGINE1=princhess ENGINE2=princhess-main

.PHONY: sprt-equal-dfrc-ltc
sprt-equal-dfrc-ltc:
	@$(MAKE) start-test TYPE=sprt_equal TC=ltc ENGINE1=princhess ENGINE2=princhess-main VARIANT=dfrc

.PHONY: resume-test
resume-test:
	@scripts/resume-test.sh

.PHONY: start-test
start-test:
	@if [ -z "$(TYPE)" ] || [ -z "$(TC)" ] || [ -z "$(ENGINE1)" ] || [ -z "$(ENGINE2)" ]; then \
		echo "Usage: make start-test TYPE=<type> TC=<tc> ENGINE1=<engine1> ENGINE2=<engine2> [THREADS=<n>] [MAX_CORES=<n>] [VARIANT=<variant>]"; \
		echo "  TYPE: sprt_gain, sprt_equal, elo_check"; \
		echo "  TC: stc, ltc, nodes25k"; \
		echo "  ENGINE1, ENGINE2: princhess, princhess-main, etc"; \
		echo "  THREADS: threads per game (optional, default: 1)"; \
		echo "  MAX_CORES: max cores available (optional, auto-detected)"; \
		echo "  VARIANT: standard, dfrc (optional, default: standard)"; \
		exit 1; \
	fi
	@scripts/start-test.sh --test-type $(TYPE) --tc $(TC) --engine1 $(ENGINE1) --engine2 $(ENGINE2) $(if $(THREADS),--threads $(THREADS)) $(if $(MAX_CORES),--max-cores $(MAX_CORES)) $(if $(VARIANT),--variant $(VARIANT))
