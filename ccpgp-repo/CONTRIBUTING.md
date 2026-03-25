# Contributing to CCPGP

Thank you for your interest in contributing to CCPGP.

## How to Contribute

### Bug Reports

Open a GitHub issue with:
- Python version and OS
- Minimal code to reproduce the problem
- Expected vs actual behaviour

### Feature Proposals

Open a GitHub issue describing:
- What the feature does
- Why it is needed
- How it relates to the existing CCPGP mechanisms

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run the test suite: `python tests/test_all.py`
5. All 22 assertions must pass
6. Submit a pull request

### Code Style

- Pure Python, zero external dependencies in the core library
- Type hints on all public functions
- Docstrings on all public classes and methods

### What We Will Not Accept

- Changes that make the constitutional constraint bypassable
- Changes that allow provenance gates to be disabled at the synapse level
- External dependencies in `ccpgp/core.py` or `ccpgp/hetero.py`

The core safety properties of CCPGP are architectural commitments, not tunable
parameters. Contributions that weaken these properties will be declined
regardless of the performance benefit they may offer.

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0.
