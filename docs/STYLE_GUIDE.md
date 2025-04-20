# Quantum Micromouse Style Guide

This document outlines the coding style and conventions for the Quantum Micromouse project.

## Python Code Style

### General Rules
- Follow PEP 8 guidelines
- Maximum line length: 88 characters (Black formatter default)
- Use 4 spaces for indentation
- Use double quotes for strings

### Naming Conventions
- Class names: `PascalCase`
- Function and variable names: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- Protected members: `__double_leading_underscore`

### Type Hints
```python
def process_maze(maze: np.ndarray, size: tuple[int, int]) -> list[tuple[int, int]]:
    """Process maze and return path."""
    pass
```

### Docstrings
Use Google style docstrings:
```python
def quantum_activation(x: jnp.ndarray) -> jnp.ndarray:
    """Apply quantum-inspired activation function.

    Args:
        x: Input array of complex numbers.

    Returns:
        Activated values with quantum properties.
    """
    pass
```

## Java Code Style

### General Rules
- Follow Oracle Java Code Conventions
- Use 4 spaces for indentation
- Braces on same line for control structures
- Maximum line length: 100 characters

### Naming Conventions
- Class names: `PascalCase`
- Method names: `camelCase`
- Variables: `camelCase`
- Constants: `UPPER_SNAKE_CASE`

## Rust Code Style

### General Rules
- Follow Rust Style Guide
- Use rustfmt for formatting
- Maximum line length: 100 characters
- Use 4 spaces for indentation

### Naming Conventions
- Struct and enum names: `PascalCase`
- Function and variable names: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Module names: `snake_case`

## Documentation

### Comments
- Use comments to explain why, not what
- Keep comments up to date with code changes
- Use TODO comments for future improvements

### README Files
- Include purpose and usage instructions
- List dependencies and setup steps
- Provide examples of common operations

## Testing

### Test Naming
- Test classes: `TestPascalCase`
- Test methods: `test_snake_case`
- Test files: `test_snake_case.py`

### Test Organization
- One test file per module
- Group related tests in classes
- Use descriptive test names

## Version Control

### Commit Messages
- Use present tense
- Start with capital letter
- No period at end
- Format: `<type>(<scope>): <description>`

### Branch Naming
- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`
- Release: `release/version` 