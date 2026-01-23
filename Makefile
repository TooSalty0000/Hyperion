.PHONY: help install start start-vega start-altair start-all clean test-imports test

PYTHON = python3
PIP = pip3

help:
	@echo "Vega Multi-Agent Bot System"
	@echo "---------------------------"
	@echo "make install       - Install dependencies from requirements.txt"
	@echo "make start         - Run Vega bot (alias for start-vega)"
	@echo "make start-vega    - Run Vega bot"
	@echo "make start-altair  - Run Altair bot"
	@echo "make start-all     - Run all bots (Vega + Altair)"
	@echo "make test          - Run pytest test suite"
	@echo "make test-imports  - Test all module imports"
	@echo "make clean         - Remove pycache and temporary files"

install:
	$(PIP) install -r requirements.txt

start: start-all

start-vega:
	$(PYTHON) run_all.py --vega

start-altair:
	$(PYTHON) run_all.py --altair

start-all:
	$(PYTHON) run_all.py

test:
	$(PYTHON) -m pytest tests/ -v

test-imports:
	@$(PYTHON) -c "from shared import BaseAgent, AgentContext, AgentResponse; print('✅ shared imports OK')"
	@$(PYTHON) -c "from altair import AltairAgent; print('✅ altair imports OK')"
	@$(PYTHON) -c "from altair.tools import StartClaudeCodeTool; print('✅ altair tools imports OK')"
	@echo "All imports successful!"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
