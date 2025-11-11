# Audit Report: random.py

**File**: `crest/utils/random.py` (202 lines)
**VBA Sources**: All VBA files (random number generation used throughout)
**Status**: ✅ **PASS** - Correct wrapper for NumPy RNG, proper VBA equivalence

---

## Purpose and Design

The `random.py` module provides a clean wrapper around NumPy's modern random number generation (`np.random.default_rng`) to replace VBA's `Rnd()` function and Excel's `WorksheetFunction.NormInv()`.

**Key Design Decisions**:
1. Uses NumPy's `default_rng()` instead of legacy `np.random` functions
2. Provides both instance-based (`RandomGenerator`) and global convenience functions
3. Supports seeding for reproducible simulations
4. Offers multiple probability distributions

---

## VBA Random Number Generation Patterns

### Pattern 1: Uniform Distribution
**VBA Code**:
```vba
Randomize  ' Initialize RNG (typically called once)
dblRand = Rnd()  ' Returns float in [0, 1)
```

**Python Equivalent**:
```python
self.rng = np.random.default_rng(seed)  # Initialize with optional seed
value = self.rng.random()  # Returns float in [0, 1)
```

**Implementation**: `RandomGenerator.random()` (lines 32-43)
- ✅ Correct: Returns float in [0.0, 1.0)
- ✅ Exact functional equivalent

**Usage Examples** (from VBA):
- `clsGlobalClimate.cls:123`: `dblRand = Rnd()`
- `clsHeatingControls.cls:330`: `dblRand = Rnd()`
- `clsHotWater.cls:170`: `dblRand = Rnd()`
- `clsAppliances.cls:119`: `dblRan = Rnd()`

---

### Pattern 2: Normal Distribution
**VBA Code**:
```vba
value = Application.WorksheetFunction.NormInv(Rnd(), mean, std_dev)
```

**Python Equivalent**:
```python
value = self.rng.normal(mean, std_dev)
```

**Implementation**: `RandomGenerator.normal()` (lines 81-99)
- ✅ Correct: Uses NumPy's normal distribution
- ✅ More efficient than inverse CDF method (VBA's approach)
- ✅ Produces statistically equivalent results

**VBA Usage Examples**:
- `clsGlobalClimate.cls:468`: `dTd = dTd + WorksheetFunction.NormInv(Rnd(), 0, dTd_SD)`
- `clsGlobalClimate.cls:681`: `aTempArray(i, 2) = WorksheetFunction.NormInv(Rnd(), 0, SD)`

**Python Usage Examples**:
- `climate.py:323`: `self.temp_array[day, 1] = self.rng.normal(0, sd)`
- `climate.py:408`: `dtd += self.rng.normal(0, dtd_sd)`
- `appliances.py:491`: `value = self.rng.normal(mean, std_dev)`
- `lighting.py:188`: `return self.rng.normal(mean, std_dev)`

**Verification**: ✅ All Python usages correctly replace VBA's `NormInv(Rnd(), ...)` pattern

---

### Pattern 3: Exponential-like Distributions (Inverse Transform)
**VBA Code**:
```vba
' Pattern A: -Log(1 - Rnd())
CycleLength = CInt(70 * ((0 - Log(1 - Rnd())) ^ 1.1))

' Pattern B: -Ln(Rnd())
dblCalibratedRelativeUseWeighting = -dblCalibrationScalar * Application.WorksheetFunction.Ln(Rnd())
```

**Python Equivalent**:
```python
# Pattern A: -np.log(1 - self.rng.random())
cycle_length = int(70 * ((-np.log(1 - self.rng.random())) ** 1.1))

# Pattern B: -np.log(self.rng.random())
calibrated_relative_use = -self.calibration_scalar * np.log(self.rng.random())
```

**Note**: Python code uses `np.log()` directly rather than `RandomGenerator.exponential()` to exactly match VBA formulas.

**VBA Usage Examples**:
- `clsAppliances.cls:422`: TV cycle length using transformed exponential
- `clsLighting.cls:151`: Bulb use weighting using exponential

**Python Usage Examples**:
- `appliances.py:449-451`: Comments VBA formula, implements with `np.log(1 - self.rng.random())`
- `lighting.py:229`: `calibrated_relative_use = -self.calibration_scalar * np.log(self.rng.random())`

**Verification**: ✅ Formulas match exactly

**Implementation in RandomGenerator**: `exponential()` (lines 101-117)
- Provided for convenience
- Not currently used (code uses direct formulas instead)
- ✅ Correct implementation if needed in future

---

### Pattern 4: Random Integer Selection
**VBA Code**:
```vba
intBulbConfiguration = Int((100 * Rnd) + 1)  ' Returns 1-100
```

**Python Equivalent**:
```python
bulb_configuration = self.rng.integers(1, 101)  # Returns int in [1, 101)
```

**Implementation**: `RandomGenerator.randint()` (lines 45-61)
- ✅ Correct: Uses `rng.integers(low, high)`
- ✅ Half-open interval [low, high) matches Python conventions
- Note: Returns `np.int64` type (see known mypy issue in TYPE_CHECKING.md)

**VBA Usage Example**:
- `clsLighting.cls:87`: `intBulbConfiguration = Int((100 * Rnd) + 1)`

---

### Pattern 5: Uniform Distribution (Custom Range)
**VBA Code**:
```vba
value = low + Rnd() * (high - low)
```

**Python Equivalent**:
```python
value = self.rng.uniform(low, high)
```

**Implementation**: `RandomGenerator.uniform()` (lines 119-137)
- ✅ Correct: Returns float in [low, high)
- ✅ Cleaner than manual scaling

---

### Pattern 6: Random Choice from Array
**VBA Code**:
```vba
' No direct equivalent - typically done with cumulative probability loops
```

**Python Equivalent**:
```python
value = self.rng.choice(array, p=probabilities)
```

**Implementation**: `RandomGenerator.choice()` (lines 63-79)
- Python enhancement over VBA
- Used by Markov chain implementation (`markov.py`)
- ✅ Proper implementation

---

## Class Structure

### RandomGenerator Class (lines 12-138)

**Attributes**:
- `rng`: NumPy Generator instance (`np.random.default_rng`)
- `seed`: Optional seed value for reproducibility

**Methods**:
| Method | VBA Equivalent | Status |
|---|---|---|
| `random()` | `Rnd()` | ✅ Exact match |
| `normal(loc, scale)` | `NormInv(Rnd(), loc, scale)` | ✅ Equivalent, more efficient |
| `exponential(scale)` | `-Log(1-Rnd())/lambda` | ✅ Correct (unused currently) |
| `uniform(low, high)` | `low + Rnd()*(high-low)` | ✅ Equivalent |
| `randint(low, high)` | `Int((high-low)*Rnd + low)` | ✅ Equivalent |
| `choice(a, p)` | (no VBA equivalent) | ✅ Python enhancement |

---

## Global RNG Pattern (lines 140-201)

**Design**: Singleton pattern with lazy initialization

```python
_global_rng: Optional[RandomGenerator] = None

def set_seed(seed: Optional[int] = None):
    global _global_rng
    _global_rng = RandomGenerator(seed)

def get_rng() -> RandomGenerator:
    global _global_rng
    if _global_rng is None:
        _global_rng = RandomGenerator()
    return _global_rng
```

**Purpose**:
- Allows setting a global seed for reproducibility
- Provides convenience functions that use the global RNG
- Matches VBA's `Randomize` behavior

**Convenience Functions** (lines 173-201):
- `random()`, `randint()`, `choice()`, `normal()`, `exponential()`, `uniform()`
- All delegate to `get_rng()`
- ✅ Clean API design

**VBA Equivalent**:
```vba
Randomize  ' Sets global seed (VBA uses system time if no seed)
value = Rnd()  ' Uses global RNG state
```

---

## Seeding and Reproducibility

### VBA Approach
```vba
Randomize  ' Seed from system clock (non-reproducible)
Randomize seed  ' Seed with specific value (reproducible)
```

### Python Approach
```python
# Non-reproducible (default)
rng = RandomGenerator()  # or RandomGenerator(None)

# Reproducible
rng = RandomGenerator(seed=42)

# Global seed
set_seed(42)
random()  # Uses seeded RNG
```

**Verification**: ✅ Python provides better reproducibility control

---

## Usage Throughout Codebase

| File | RNG Creation | Usage Pattern |
|---|---|---|
| `climate.py` | `self.rng = random_gen` (passed in) | `self.rng.random()`, `self.rng.normal()` |
| `occupancy.py` | `self.random_gen = random_gen` | Uses `markov.py` helpers |
| `appliances.py` | `self.rng = random_gen` | `self.rng.random()`, `self.rng.normal()` |
| `lighting.py` | `self.rng = random_gen` | `self.rng.random()`, `self.rng.normal()` |
| `water.py` | `self.rng = random_gen` | `self.rng.random()` |
| `controls.py` | `self.random_gen = random_gen` | `self.random_gen.random()`, `self.random_gen.randint()` |

**Pattern**: All classes receive `RandomGenerator` instance in `__init__` or initialization method.

**Verification**: ✅ Consistent usage across codebase

---

## Comparison: VBA vs Python RNG

| Feature | VBA | Python (random.py) | Status |
|---|---|---|---|
| **Generator Type** | Mersenne Twister (VBA 7.0+) | PCG64 (NumPy default) | ✅ Python better quality |
| **Seeding** | `Randomize seed` | `default_rng(seed)` | ✅ Equivalent |
| **Uniform [0,1)** | `Rnd()` | `rng.random()` | ✅ Exact match |
| **Normal** | `NormInv(Rnd(), μ, σ)` | `rng.normal(μ, σ)` | ✅ Equivalent, more efficient |
| **Exponential** | `-Log(1-Rnd())` | `rng.exponential()` | ✅ Python provides both |
| **Integer** | `Int(Rnd()*N)` | `rng.integers(N)` | ✅ Equivalent |
| **Reproducibility** | Seed-based | Seed-based | ✅ Equivalent |

---

## Known Issues

### Type Checking (Non-Critical)
From `TYPE_CHECKING.md`:
```
crest/utils/random.py:61:16: error: Incompatible return value type
(got "signedinteger[_64Bit]", expected "int")  [return-value]
    return self.rng.integers(low, high)
```

**Issue**: NumPy's `integers()` returns `np.int64`, not Python `int`

**Impact**: None - NumPy integers are duck-type compatible with Python ints

**Fix Options**:
1. Cast to `int`: `return int(self.rng.integers(low, high))`
2. Type ignore: `return self.rng.integers(low, high)  # type: ignore[return-value]`
3. Leave as-is (current approach)

**Recommendation**: Leave as-is or add type ignore comment. Casting adds unnecessary overhead.

---

## Issues Found

**NONE** - Implementation is correct.

---

## Summary

**Methods Verified**: 6 core methods + 6 convenience functions
**VBA Patterns Matched**: 6 patterns
**Usage Locations**: 7 files in `crest/core/`

**Result**: ✅ **PASS** - random.py correctly wraps NumPy RNG

The module provides:
1. ✅ Exact functional equivalence to VBA's `Rnd()` and `NormInv()`
2. ✅ Proper seeding for reproducibility
3. ✅ Clean API with both instance and global usage patterns
4. ✅ Additional useful distributions (exponential, uniform, choice)
5. ✅ Better RNG quality than VBA (PCG64 vs Mersenne Twister)

**No changes needed.**

All VBA random number generation patterns are correctly implemented or improved upon in the Python version.
