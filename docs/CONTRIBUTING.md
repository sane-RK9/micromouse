# Contributing to Quantum Micromouse

Thank you for your interest in contributing to Quantum Micromouse! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.

## How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/sane-RK9/micromouse.git
   cd micromouse
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/MacOS
   .\.venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes**
   - Follow the coding style guide
   - Write tests for new features
   - Update documentation as needed

5. **Submit a Pull Request**
   - Describe your changes
   - Reference any related issues
   - Ensure all tests pass

## Coding Standards

- Follow PEP 8 for Python code
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Write unit tests for new features

## Documentation

- Update README.md for significant changes
- Add or update docstrings for new/modified code
- Update API documentation if needed

## Testing

- Run all tests before submitting PR:
  ```bash
  pytest
  ```
- Ensure test coverage remains above 90%

## Review Process

1. Pull requests will be reviewed by maintainers
2. Feedback will be provided within 3 business days
3. Address any feedback and update your PR
4. Once approved, your changes will be merged

## Questions?

Feel free to open an issue or contact the maintainers at micromouse-support@quantumlabs.ai 