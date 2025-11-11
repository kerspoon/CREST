# Type Checking with mypy

This project uses [mypy](https://mypy-lang.org/) for static type checking to catch interface bugs before runtime.

## Quick Start

```bash
# Run type checker
./check_types.sh

# Or run mypy directly
venv/bin/mypy crest/
```

## Current Status

**Initial mypy run:** 57 errors in 15 files (out of 24 source files)

Most errors are:
- **Optional types**: Attributes initialized with `None` but not typed as `Optional`
- **Unreachable code**: Early returns causing dead code warnings
- **Numpy types**: Minor type incompatibilities with numpy arrays

## Common Error Types and Fixes

### 1. Optional Attribute Initialization

**Error:**
```
error: Incompatible types in assignment (expression has type "None",
variable has type "Building")
```

**Fix:**
```python
# BEFORE
self.building: Building = None

# AFTER
from typing import Optional
self.building: Optional[Building] = None
```

### 2. Unreachable Code

**Error:**
```
error: Statement is unreachable
```

**Cause:** Early return or condition that always evaluates same way

**Fix:** Remove the unreachable code or fix the logic

### 3. CSV Writer Type

**Error:**
```
error: Function "_csv.writer" is not valid as a type
```

**Fix:**
```python
# BEFORE
import csv
self.writer: Optional[csv.writer] = None

# AFTER
from typing import Any
self.writer: Optional[Any] = None  # csv writer object
```

### 4. Numpy Type Mismatches

**Error:**
```
error: Incompatible return value type (got "signedinteger[_64Bit]", expected "int")
```

**Fix:**
```python
# Option 1: Cast to int
return int(self.rng.integers(low, high))

# Option 2: Type ignore for numpy compatibility
return self.rng.integers(low, high)  # type: ignore[return-value]
```

## Incremental Adoption Strategy

The mypy.ini configuration is set up for incremental adoption:

### **Phase 1 (Current): Basic checking**
- ✅ Check existing type hints
- ✅ Catch obvious errors
- ⚠️ Many false positives expected

**Action:** Run `./check_types.sh` regularly, fix obvious issues

### **Phase 2: Require function signatures**
Uncomment in mypy.ini:
```ini
disallow_incomplete_defs = True
```

**Action:** Add type hints to all public methods:
```python
def get_temperature(self, timestep: int) -> float:
    ...
```

### **Phase 3: Stricter checking**
Uncomment in mypy.ini:
```ini
disallow_untyped_calls = True
```

**Action:** Ensure all called functions have type hints

### **Phase 4: Full strictness**
Uncomment in mypy.ini:
```ini
disallow_untyped_defs = True
warn_return_any = True
```

**Action:** Complete type coverage across codebase

## Integration

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:
```bash
#!/bin/bash
./check_types.sh || {
    echo "⚠️  Type errors found. Commit anyway? (y/n)"
    read answer
    [ "$answer" = "y" ] || exit 1
}
```

### CI/CD

Add to GitHub Actions / GitLab CI:
```yaml
- name: Type check
  run: |
    pip install mypy
    mypy crest/
```

## Useful Commands

```bash
# Check specific file
venv/bin/mypy crest/core/building.py

# Check with more detail
venv/bin/mypy crest/ --show-error-context --show-column-numbers

# Generate HTML report
venv/bin/mypy crest/ --html-report mypy-report/

# Check without cache
venv/bin/mypy crest/ --no-incremental
```

## When to Ignore Errors

Use `# type: ignore` comments sparingly:

```python
# Good reasons to ignore:
x = external_lib.function()  # type: ignore[no-untyped-call]  # Library has no types
y = complicated_numpy_operation()  # type: ignore[misc]  # Numpy type too complex

# Bad reasons:
z = self.method()  # type: ignore  # Too lazy to fix - DON'T DO THIS
```

## Benefits

✅ **Catches interface bugs** - Method name mismatches found before runtime
✅ **Better IDE support** - Autocomplete and inline error checking
✅ **Living documentation** - Type hints show expected inputs/outputs
✅ **Refactoring safety** - Breaks caught immediately, not in production

## Resources

- [mypy documentation](https://mypy.readthedocs.io/)
- [Python typing module](https://docs.python.org/3/library/typing.html)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
