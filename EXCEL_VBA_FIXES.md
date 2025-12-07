# Excel VBA Bug Fixes

This document describes the three VBA bug fixes needed to create corrected versions of the Excel models.

## Summary

| Fix | File | Line | Impact |
|-----|------|------|--------|
| 1. Day of Year Missing | `clsSolarThermal.cls` | ~378 | Solar thermal uses wrong day (0 or undefined) |
| 2. Tan(declination)/Tan(declination) | `clsSolarThermal.cls` | ~427 | Sunrise/sunset check always true (÷1) |
| 3. Cos() instead of Acos() | `clsSolarThermal.cls` | ~465 | ~2x error in direct beam radiation |

**Note:** All three fixes are in `clsSolarThermal.cls`. The `lcg_fixed.xlsm` also contains LCG RNG changes for deterministic comparison - the fixes below are ONLY the bug fixes.

---

## Fix 1: Day of Year Not Calculated

### Problem

The `GetIncidentRadiation()` function uses `dblDayOfYear` but never calculates it. In VBA, this means it defaults to 0, causing incorrect solar position calculations.

### Location

**File:** `clsSolarThermal.cls`
**Function:** `GetIncidentRadiation()`
**Line:** ~378 (after the variable declarations, before `dblLongitude = ...`)

### Original Code (BUGGY)

```vba
Dim dblSkyDiffuseFactor As Double

' // Get longitude, latitude and meridian
dblLongitude = wsMain.Range("rLongitude").Value
```

### Fixed Code

```vba
Dim dblSkyDiffuseFactor As Double

' // ADDED to fix bug - need to calculate day of year
Dim intDayOfMonth As Integer
Dim intMonthOfYear As Integer
Dim strDate As String

' // Get longitude, latitude and meridian
dblLongitude = wsMain.Range("rLongitude").Value
dblLatitude = wsMain.Range("rLatitude").Value
dblMeridian = wsMain.Range("rMeridian").Value

' // ADDED to fix bug - calculate day of year from Main Sheet settings
intDayOfMonth = wsMain.Range("rDayOfMonth").Value
intMonthOfYear = wsMain.Range("rMonthOfYear").Value
strDate = CStr(intDayOfMonth) + "/" + CStr(intMonthOfYear) + "/2015"
dblDayOfYear = DatePart("y", CDate(strDate))

' // Determine the hour and minute
```

---

## Fix 2: Tan(declination)/Tan(declination) Division Error

### Problem

The sunrise/sunset check divides `Tan(dblDeclination)` by itself, which always equals 1. The denominator should use `Tan(dblLatitude)`.

### Location

**File:** `clsSolarThermal.cls`
**Function:** `GetIncidentRadiation()`
**Line:** ~427 (search for `Tan(dblDeclination`)

### Original Code (BUGGY)

```vba
If Cos(dblHourAngle * PI / 180) >= (Tan(dblDeclination * PI / 180) / Tan(dblDeclination * PI / 180)) Then
```

### Fixed Code

```vba
' // FIX: Changed denominator from Tan(dblDeclination) to Tan(dblLatitude)
If Cos(dblHourAngle * PI / 180) >= (Tan(dblDeclination * PI / 180) / Tan(dblLatitude * PI / 180)) Then
```

### Explanation

The solar altitude check formula should be:
```
cos(hour_angle) >= tan(declination) / tan(latitude)
```

The original code had `tan(declination) / tan(declination) = 1`, which made the sunrise check effectively useless.

---

## Fix 3: Cos() Instead of Acos() for Incident Angle

### Problem

The code uses `Cos()` on a dot product when it should use `Acos()` to calculate the incident angle. This causes approximately 2x error in direct beam radiation on the solar thermal collector.

### Location

**File:** `clsSolarThermal.cls`
**Function:** `GetIncidentRadiation()`
**Line:** ~465 (search for `dblSolarIncidentAngle =`)

### Original Code (BUGGY)

```vba
' // Calculate solar incident angle on panel
dblSolarIncidentAngle = Cos((Cos(dblSolarAltitudeAngle * PI / 180) * Cos(dblAdjustedAzimuthOfSun * PI / 180 - dblAzimuth * PI / 180) * Sin(dblSlope * PI / 180)) + (Sin(dblSolarAltitudeAngle * PI / 180) * Cos(dblSlope * PI / 180)))
```

### Fixed Code

```vba
' // Calculate solar incident angle on panel
' // FIX: Changed Cos() to Acos() - need arc-cosine to get angle from dot product
dblSolarIncidentAngle = (180 / PI) * Application.WorksheetFunction.Acos((Cos(dblSolarAltitudeAngle * PI / 180) * Cos(dblAdjustedAzimuthOfSun * PI / 180 - dblAzimuth * PI / 180) * Sin(dblSlope * PI / 180)) + (Sin(dblSolarAltitudeAngle * PI / 180) * Cos(dblSlope * PI / 180)))
```

### Explanation

The expression inside the parentheses computes the **cosine** of the incident angle (a dot product between unit vectors).

- **Bug:** `Cos(dot_product)` takes cosine of an already-cosine value (nonsense)
- **Fix:** `Acos(dot_product)` converts cosine back to angle in radians, then `* (180/PI)` converts to degrees

---

## Step-by-Step Instructions

### Creating `monte_carlo_fixed.xlsm`

1. **Copy the file:**
   ```
   Copy: excel/monte_carlo_base.xlsm
   To:   excel/monte_carlo_fixed.xlsm
   ```

2. **Open in Excel and access VBA Editor:**
   - Open `monte_carlo_fixed.xlsm`
   - Press `Alt+F11` to open VBA Editor

3. **Find clsSolarThermal:**
   - In Project Explorer, expand "Class Modules"
   - Double-click `clsSolarThermal`

4. **Apply Fix 1 (Day of Year):**
   - Find line ~378: `Dim dblSkyDiffuseFactor As Double`
   - Add the new variable declarations after it
   - Find where `dblMeridian = wsMain.Range("rMeridian").Value` is set
   - Add the day of year calculation code immediately after

5. **Apply Fix 2 (Tan division):**
   - Press `Ctrl+F` and search for `Tan(dblDeclination`
   - Find the line with `/ Tan(dblDeclination`
   - Change the **second** `dblDeclination` to `dblLatitude`

6. **Apply Fix 3 (Acos):**
   - Press `Ctrl+F` and search for `dblSolarIncidentAngle =`
   - Replace the entire line with the fixed version using `Acos`

7. **Save and close:**
   - Press `Ctrl+S` to save
   - Close VBA Editor

### Creating `original_fixed.xlsm`

Follow the same steps as above, but:
```
Copy: excel/original.xlsm
To:   excel/original_fixed.xlsm
```

---

## Quick Reference - All Changes

In `clsSolarThermal.cls`, make these three changes:

```vba
' === FIX 1: Add after "Dim dblSkyDiffuseFactor As Double" (~line 378) ===
Dim intDayOfMonth As Integer
Dim intMonthOfYear As Integer
Dim strDate As String

' === FIX 1: Add after "dblMeridian = ..." line (~line 387) ===
intDayOfMonth = wsMain.Range("rDayOfMonth").Value
intMonthOfYear = wsMain.Range("rMonthOfYear").Value
strDate = CStr(intDayOfMonth) + "/" + CStr(intMonthOfYear) + "/2015"
dblDayOfYear = DatePart("y", CDate(strDate))

' === FIX 2: Change (~line 427) ===
' FROM: ... / Tan(dblDeclination * PI / 180)) Then
' TO:   ... / Tan(dblLatitude * PI / 180)) Then

' === FIX 3: Change (~line 465) ===
' FROM: dblSolarIncidentAngle = Cos((...))
' TO:   dblSolarIncidentAngle = (180 / PI) * Application.WorksheetFunction.Acos((...))
```

---

## Verification

After applying all three fixes, solar thermal collector heat gains should match Python output within floating-point tolerance (~0.001 kWh/day).

To verify:
1. Run `venv/bin/python3 scripts/rng_validation_run.py --validation-log --no-export`
2. Compare results using `venv/bin/python3 scripts/compare_results.py`

---

## Files That Need Fixing

| Source File | Fixed File | Purpose |
|-------------|------------|---------|
| `original.xlsm` | `original_fixed.xlsm` | General use |
| `monte_carlo_base.xlsm` | `monte_carlo_fixed.xlsm` | Monte Carlo validation |

**Note:** `lcg_fixed.xlsm` already has all three fixes applied (along with LCG RNG for deterministic comparison).
