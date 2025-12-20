# Contributing to The LLM Council

Thank you for your interest in contributing to The LLM Council! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions. We're building something together.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/the-llm-council.git
   cd the-llm-council
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

## Development Workflow

### Making Changes

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our coding standards (see below)

3. Run the test suite:
   ```bash
   pytest
   ```

4. Run linting and type checking:
   ```bash
   ruff check src/
   mypy src/llm_council
   ```

5. Commit your changes with a clear message:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Open a Pull Request on GitHub

### Coding Standards

- **Python 3.10+**: Use modern Python features (type hints, match statements, etc.)
- **Type hints**: All public functions must have type annotations
- **Docstrings**: Use Google-style docstrings for public APIs
- **Formatting**: Code is formatted with `ruff format`
- **Linting**: Code must pass `ruff check` with no errors
- **Type checking**: Code must pass `mypy --strict`

### Testing

- Write tests for new functionality
- Maintain or improve code coverage
- Use `pytest` for all tests
- Use `pytest-asyncio` for async tests

### Commit Messages

Use clear, descriptive commit messages:
- Start with a verb (Add, Fix, Update, Remove, Refactor)
- Keep the first line under 72 characters
- Add details in the body if needed

Examples:
```
Add OpenRouter provider with streaming support
Fix timeout handling in CLI adapters
Update health check to include latency metrics
```

## Adding a New Provider

1. Create a new file in `src/llm_council/providers/`
2. Implement the `ProviderAdapter` interface
3. Register via entry points in `pyproject.toml`
4. Add tests in `tests/unit/providers/`
5. Update documentation

See `src/llm_council/providers/openrouter.py` for a reference implementation.

## Adding a New Subagent

1. Create a YAML config in `src/llm_council/subagents/`
2. Create a JSON schema in `src/llm_council/schemas/`
3. Add tests for the new subagent
4. Update the README subagent table

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass
- Request review from maintainers

## Reporting Issues

When reporting issues, please include:
- Python version (`python --version`)
- OS and version
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces

## Questions?

Open an issue with the "question" label or start a discussion.

Thank you for contributing!
