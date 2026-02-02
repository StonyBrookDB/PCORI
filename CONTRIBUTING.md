# Contributing to PCORI

Thank you for your interest in contributing to the PCORI Clinical Decision Support project. This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/PCORI.git
   cd PCORI
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## Code Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Documentation

- All public functions must have docstrings (Google style)
- Update README.md if adding new features
- Add inline comments for complex logic

### Example Docstring

```python
def train_model(data: pd.DataFrame, config: dict) -> Model:
    """Train a risk prediction model.

    Args:
        data: Patient features DataFrame with columns matching config.
        config: Training configuration including model_type, hyperparameters.

    Returns:
        Trained model instance with predict() method.

    Raises:
        ValueError: If required columns missing from data.
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Aim for >80% code coverage for new features

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Run the test suite** and ensure all tests pass
4. **Update CHANGELOG.md** with your changes
5. **Submit PR** with clear description of changes

### PR Title Format

```
[Component] Brief description

Examples:
[Dashboard] Add patient export feature
[Pipeline] Fix LSTM memory leak
[Docs] Update installation guide
```

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Changes
- Change 1
- Change 2

## Testing
How were these changes tested?

## Related Issues
Fixes #123
```

## Reporting Issues

### Bug Reports

Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered

## Code Review

All submissions require review. Reviewers will check:

- Code quality and style
- Test coverage
- Documentation
- Security considerations
- Performance implications

## Security

- Never commit credentials or API keys
- Use environment variables for sensitive configuration
- Report security vulnerabilities privately to the maintainers

## Questions?

Open a GitHub issue with the `question` label.

---

Thank you for contributing to clinical decision support research!
