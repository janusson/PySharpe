# Check linting standards
uv run ruff check .

# Verify formatting compliance
uv run ruff format --check .

# Execute the comprehensive test suite
uv run pytest > pytest_results.log
