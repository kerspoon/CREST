# Monte Carlo Validation Range Violations Report

**Date**: 2025-12-08
**Python runs**: 1000 (python_1000runs_20251208_01)
**Excel runs**: 100 (excel_100runs_20251207)
**Dwellings**: 5

## Executive Summary

The Monte Carlo IQR validation shows **good practical agreement** between Python and Excel implementations, though not mathematically proven identical:

- **Daily totals**: 52.7% in IQR ✓ PASS (expected 50%, 95% CI: 45.6%-54.4%)
- **Disaggregated**: 77.9% in IQR ⚠ HIGH (expected ~50% for continuous variables)

**Daily range violations**: 9/7500 (0.12%) - better than the ~0.2% expected.

**Disaggregated range violations**: 1-2% for solar variables - 10× higher than expected, indicating minor differences in solar edge-case handling.

**Conclusion**: Probably identical for practical purposes. Daily totals match well. The elevated disaggregated IQR (77.9%) and solar range violations suggest Python may have slightly wider distributions than Excel for some variables, or there are minor differences in edge-case handling that don't significantly affect aggregate results.

---

## Daily Range Violations (9 cases out of 7500)

With 100 Excel runs × 5 dwellings × 15 variables = 7500 comparisons, 9 values (0.12%) fell outside Python's min-max range. This is better than the expected ~0.2%.

| Variable | Out of Range | Total | Percentage |
|----------|-------------|-------|------------|
| Gas demand | 3 | 500 | 0.6% |
| Lighting demand | 2 | 500 | 0.4% |
| Total self-consumption | 2 | 500 | 0.4% |
| Mean active occupancy | 1 | 500 | 0.2% |
| Proportion of day actively occupied | 1 | 500 | 0.2% |

**Analysis**: These are minor edge cases:
- Gas demand and lighting demand violations suggest occasional extreme values in Excel not captured by 1000 Python runs
- Occupancy violations are floating-point precision artifacts (differences of ~10⁻¹⁰)

**Verdict**: ✓ Acceptable - within expected statistical variation.

---

## Disaggregated Range Violations

26 variables have some minutes outside Python's range. The highest are all solar-related:

| Variable | Out of Range | Percentage |
|----------|-------------|------------|
| PV output | 15,790 | 2.19% |
| Radiation incident on PV array | 15,790 | 2.19% |
| Outdoor global radiation (horizontal) | 15,730 | 2.18% |
| Passive solar gains | 14,742 | 2.05% |
| Solar power incident on collector | 9,498 | 1.32% |

**Analysis**: These violations are 10× higher than the expected ~0.2%. All trace back to solar irradiance calculations, suggesting:

1. **Edge-case handling differs**: Sunrise/sunset boundary conditions may be handled slightly differently
2. **Trigonometric precision**: Solar position calculations involve many trig functions where small differences compound
3. **Stochastic cloud cover**: Random cloud factor may interact differently with edge cases

**Critical note**: While these don't affect daily totals significantly (which pass at 52.7%), the 2% rate indicates the implementations are not mathematically identical for solar calculations. The root cause is not fully understood.

**Verdict**: ⚠ Acceptable for practical use, but indicates minor unresolved differences in solar edge-case handling.

---

## IQR Analysis Notes

The disaggregated data shows 77.9% in IQR (higher than the expected 50%). This is partially explained by discrete/binary variables:

- Binary states (heating on/off, cooling on/off): 100% in IQR when both produce mostly 0s
- Timer settings: Discrete values cluster together
- Hot water demand: Many zero values

**However**, 77.9% is elevated even accounting for discrete variables. This suggests Python may have slightly wider distributions than Excel for some continuous variables. Possible causes:

1. **Different variance in stochastic components**: Random number usage may produce wider spread in Python
2. **Floating-point accumulation**: Small differences compound differently over 1440 minutes
3. **Edge-case handling**: Boundary conditions may be handled slightly differently

**Key IQR outliers from daily totals:**
| Variable | IQR % | Status |
|----------|-------|--------|
| Solar thermal collector heat gains | 71.8% | High - discrete, often zero |
| Space thermostat set point | 60.8% | High - discrete values |
| Average indoor air temperature | 45.6% | Low - at edge of 95% CI |

The indoor temperature being at the low edge (45.6%) could indicate a slight systematic difference in the thermal model, though it's within acceptable bounds.

---

## Conclusion

**Probably identical for practical purposes, but not proven mathematically identical.**

**What we know:**
- Daily totals match well (52.7% ≈ 50%)
- RNG validation with deterministic seed shows 40/40 columns match within floating-point precision
- Daily range violations are minimal (0.12%)

**What remains unexplained:**
- Disaggregated IQR at 77.9% (higher than expected ~50%)
- Solar range violations at 2% (10× expected rate)
- Average indoor temperature at edge of confidence interval

The Python implementation is suitable for practical use. For applications requiring mathematical equivalence, further investigation of the elevated disaggregated IQR and solar edge cases would be needed.
