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
	@echo "Tuning:"
	@echo "  tune-25k-small      - 25k tune (±25%, requires PARAMS)"
	@echo "  tune-25k-medium     - 25k tune (±50%, requires PARAMS)"
	@echo "  tune-25k-large      - 25k tune (±100%, requires PARAMS)"
	@echo "  tune-stc-small      - STC tune (±25%, requires PARAMS)"
	@echo "  tune-stc-medium     - STC tune (±50%, requires PARAMS)"
	@echo "  resume-tune         - Resume previous tuning session"
	@echo "  start-tune          - Start custom tune (requires TUNE_TYPE, SIZE, PARAMS)"
	@echo ""
	@echo "Examples:"
	@echo "  make lichess-bot LICHESS_TOKEN=lip_xxxxxxxxxx"
	@echo "  make sprt-gain"
	@echo "  make start-test TYPE=sprt_gain TC=ltc ENGINE1=princhess ENGINE2=princhess-main"
	@echo "  make tune-25k-small PARAMS=\"CPuct CPuctTau PolicyTemperatureRoot\""
	@echo "  make tune-stc-medium PARAMS=\"CPuctJitter\""
	@echo "  make start-tune TUNE_TYPE=25k SIZE=large PARAMS=\"CPuct CPuctTau\""
	@echo "  make resume-tune"

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
ifndef LICHESS_TOKEN
	$(error LICHESS_TOKEN is required. Usage: make lichess-bot LICHESS_TOKEN=your_token)
endif
	docker build -f bin/Dockerfile.lichess -t princhess-lichess .
	docker run -e LICHESS_TOKEN=$(LICHESS_TOKEN) -v $$(pwd)/bin/config.yml:/src/config.yml:ro princhess-lichess

.PHONY: lichess-bot-policy
lichess-bot-policy:
ifndef LICHESS_TOKEN
	$(error LICHESS_TOKEN is required. Usage: make lichess-bot-policy LICHESS_TOKEN=your_token)
endif
	docker build -f bin/Dockerfile.lichess -t princhess-lichess .
	docker run -e LICHESS_TOKEN=$(LICHESS_TOKEN) -v $$(pwd)/bin/config.policy.yml:/src/config.yml:ro princhess-lichess

# Test targets
.PHONY: sprt-gain
sprt-gain:
	@scripts/start-test.sh sprt_gain stc princhess princhess-main

.PHONY: sprt-gain-ltc
sprt-gain-ltc:
	@scripts/start-test.sh sprt_gain ltc princhess princhess-main

.PHONY: sprt-gain-2t
sprt-gain-2t:
	@scripts/start-test.sh sprt_gain stc princhess princhess-main 2t

.PHONY: sprt-equal
sprt-equal:
	@scripts/start-test.sh sprt_equal stc princhess princhess-main

.PHONY: sprt-equal-2t
sprt-equal-2t:
	@scripts/start-test.sh sprt_equal stc princhess princhess-main 2t

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
define check_params
	@if [ -z "$(PARAMS)" ]; then \
		echo "Usage: make $(1) PARAMS=\"<params>\""; \
		echo "  PARAMS: Space-separated UCI option names"; \
		echo "  Example: make $(1) PARAMS=\"CPuct CPuctTau PolicyTemperatureRoot\""; \
		exit 1; \
	fi
endef

.PHONY: tune-25k-small
tune-25k-small:
	$(call check_params,$@)
	@scripts/start-tune.sh 25k small $(PARAMS)

.PHONY: tune-25k-medium
tune-25k-medium:
	$(call check_params,$@)
	@scripts/start-tune.sh 25k medium $(PARAMS)

.PHONY: tune-25k-large
tune-25k-large:
	$(call check_params,$@)
	@scripts/start-tune.sh 25k large $(PARAMS)

.PHONY: tune-stc-small
tune-stc-small:
	$(call check_params,$@)
	@scripts/start-tune.sh stc small $(PARAMS)

.PHONY: tune-stc-medium
tune-stc-medium:
	$(call check_params,$@)
	@scripts/start-tune.sh stc medium $(PARAMS)

.PHONY: resume-tune
resume-tune:
	@scripts/resume-tune.sh

.PHONY: start-tune
start-tune:
	@if [ -z "$(TUNE_TYPE)" ] || [ -z "$(SIZE)" ] || [ -z "$(PARAMS)" ]; then \
		echo "Usage: make start-tune TUNE_TYPE=<type> SIZE=<size> PARAMS=\"<params>\""; \
		echo "  TUNE_TYPE: 25k, stc"; \
		echo "  SIZE: small (±25%), medium (±50%), large (±100%)"; \
		echo "  PARAMS: Space-separated UCI option names"; \
		exit 1; \
	fi
	@scripts/start-tune.sh $(TUNE_TYPE) $(SIZE) $(PARAMS)