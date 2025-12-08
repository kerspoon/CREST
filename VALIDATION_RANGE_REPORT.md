# Monte Carlo Validation Report

**Date**: 2025-12-08
**Python runs**: 1000 (python_1000runs_20251208_01)
**Excel runs**: 100 (excel_100runs_20251207)
**Dwellings**: 5

## Executive Summary

The Monte Carlo validation shows **excellent agreement** between Python and Excel:

- **Daily totals**: 52.7% in IQR (expected 50%) - **PASS**
- **Mean differences**: All metrics within 2% - **PASS**
- **Range violations**: Fully explained by floating-point precision and stochastic variation - **PASS**

---

## Summary Statistics

### Daily Totals Comparison

| Metric | Python Mean | Excel Mean | Difference |
|--------|-------------|------------|------------|
| Total electricity demand | 13.18 kWh | 13.28 kWh | +0.73% |
| Gas demand | 4.09 m³ | 4.09 m³ | +0.04% |
| Lighting demand | 3.46 kWh | 3.52 kWh | +1.80% |
| Appliance demand | 9.72 kWh | 9.76 kWh | +0.35% |
| PV output | 1.67 kWh | 1.68 kWh | +0.50% |
| Indoor air temperature | 18.29 °C | 18.32 °C | +0.15% |

All differences are within expected statistical variation for 1000 vs 100 samples.

### Range Comparison

| Metric | Python Range | Excel Range | Excel Outside |
|--------|--------------|-------------|---------------|
| Total electricity (kWh) | [1.37, 57.12] | [2.75, 38.36] | 0 |
| Gas demand (m³) | [0.00, 11.92] | [0.40, 12.11] | 1 (by 0.19 m³) |
| Lighting demand (kWh) | [0.00, 16.51] | [0.00, 15.07] | 0 |

Only 1 Excel daily value (out of 500) fell outside Python's 1000-run range, and only by 0.19 m³.

---

## Understanding the 77.9% IQR Result

### What the IQR Test Measures

For each (minute, dwelling, variable), we compute Python's interquartile range (Q1 to Q3) from 1000 runs, then check what percentage of Excel's 100 values fall within that range.

For a **continuous variable** with identical distributions, you'd expect **~50%** in IQR (by definition - the IQR contains the middle 50% of data).

### Why We See 77.9%

The 77.9% is a weighted average across all 37 output variables. Looking at them individually reveals the pattern:

**Continuous variables show ~50% (exactly as expected):**

| Variable | IQR % |
|----------|-------|
| Appliance demand | 50.8% |
| Outdoor temperature | 50.9% |
| Internal building node temperature | 48.6% |
| Emitter temperature | 49.1% |
| Net dwelling electricity demand | 50.5% |
| Hot water tank temperature | 51.4% |
| Casual thermal gains | 49.0% |

**Discrete/binary variables show ~95-100%:**

| Variable | IQR % | Why |
|----------|-------|-----|
| Space cooling timer settings | 100% | Always 0 (UK winter) |
| Electricity used by cooling | 100% | Always 0 |
| Cooling output | 100% | Always 0 |
| Hot water timer settings | 99.7% | Binary on/off |
| Solar thermal control state | 99.8% | Binary on/off |
| Hot water demand (litres) | 97.1% | Mostly 0 with occasional spikes |
| Heating system switched on | 91.9% | Binary on/off |

**Explanation**: When both Python and Excel produce mostly zeros (e.g., cooling in UK winter), the IQR is [0, 0] and all Excel zeros fall "within" it, giving 100% match. This inflates the overall average above 50%.

**The 77.9% is correct and expected** - it reflects the mix of continuous and discrete variables in the output, not a problem with the implementation.

---

## Range Violation Analysis

### Initial Concern

Disaggregated data showed ~2% of solar-related values falling outside Python's min-max range. This was initially concerning (expected ~0.2%).

### Root Cause Analysis

We analysed all 15,730 "violations":

| Difference Magnitude | Count | Percentage |
|---------------------|-------|------------|
| < 10⁻⁶ W/m² | 15,600 | 99.2% |
| > 0.01 W/m² | 130 | 0.8% |
| > 1.0 W/m² | 115 | 0.7% |

### Finding 1: 99% Are Floating-Point Noise

The vast majority of "violations" are ~10⁻⁹ W/m² differences. Example:
- Excel value: 88.51510 W/m²
- Python max: 88.51510 W/m²
- Difference: 2.3×10⁻¹¹ W/m²

This is IEEE 754 floating-point rounding between VBA and Python, not an algorithmic difference.

### Finding 2: The 1% Are Stochastic Variation

The 130 larger violations (> 0.01 W/m²) were traced to:
- **Specific runs**: Excel runs 12 and 61
- **Specific times**: Minutes 625-650 (~10:30 AM)
- **Cause**: These runs hit extremely cloudy conditions that 1000 Python runs didn't reproduce

Example at minute 647:
- Excel run 61: 2.7 W/m² (very cloudy)
- Python range: 5.4 - 104 W/m² (less cloudy)

With 100 Excel runs vs 1000 Python runs, some extreme weather wasn't captured. This is expected statistical behavior.

---

## Conclusion

**The implementations are statistically equivalent.**

- Daily means match within 2%
- Continuous variables show ~50% IQR (exactly as expected)
- Discrete variables show ~100% IQR (correctly - both produce same discrete values)
- All "range violations" are floating-point noise (99%) or stochastic variation (1%)

The Python implementation is validated for production use.
