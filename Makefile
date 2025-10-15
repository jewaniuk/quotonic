.PHONY: build
build: clean-build ## Build wheel file using build
	@echo "🚀 Creating wheel file"
	@python -m build

.PHONY: clean-build
clean-build: ## Clean build artifacts (remove dist)
	@rm -rf dist

.PHONY: docs
docs: ## Build and serve the documentation
	@mkdocs serve

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@mkdocs build -s

.PHONY: docs-gh
docs-gh: ## Build documentation for GitHub Pages
	@mkdocs gh-deploy

.PHONY: check
check: ## Run code quality tools
	@echo "🚀 Linting code: Running pre-commit"
	@pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@mypy

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@pytest

.PHONY: badges
badges: ## Create README badges for pytest-cov and flake8
	@echo "🚀 Testing code: Running pytest"
	@pytest --cov-report xml:badges/coverage.xml
	@echo "🚀 Creating coverage badge"
	@genbadge coverage -i ./badges/coverage.xml -o ./badges/coverage.svg
	@echo "🚀 Linting code: Running flake8"
	@flake8 --statistics --tee --output-file ./badges/flake8stats.txt
	@echo "🚀 Creating flake8 badge"
	@genbadge flake8 -i ./badges/flake8stats.txt -o ./badges/flake8.svg

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help