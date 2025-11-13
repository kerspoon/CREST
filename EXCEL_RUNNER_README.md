# Excel Macro Runner for CREST

This PowerShell script allows you to repeatedly run the CREST demand model macro in Excel and automatically save the output sheets as CSV files.

## Files

- `run_excel_macro.ps1` - Main PowerShell script
- `run_excel_example.bat` - Example batch file for easy execution

## Quick Start

### Option 1: Using the Batch File (Easiest)

1. Edit `run_excel_example.bat` and set:
   - `EXCEL_FILE` - Path to your .xlsm file
   - `RUN_COUNT` - Number of times to run the macro
   - `OUTPUT_DIR` - Where to save results

2. Double-click `run_excel_example.bat`

### Option 2: Using PowerShell Directly

Open PowerShell and run:

```powershell
.\run_excel_macro.ps1 -ExcelFile "excel\monte_carlo_base.xlsm" -RunCount 10
```

## Parameters

- **`-ExcelFile`** (required): Path to the Excel .xlsm file
- **`-RunCount`** (optional): Number of times to run the macro (default: 1)
- **`-OutputDir`** (optional): Directory to save CSV outputs (default: ".\excel_runs")
- **`-AddTimestamp`** (optional): Add timestamp to run folders (default: true)

## Examples

### Run once:
```powershell
.\run_excel_macro.ps1 -ExcelFile "excel\original.xlsm"
```

### Run 20 times for Monte Carlo:
```powershell
.\run_excel_macro.ps1 -ExcelFile "excel\monte_carlo_base.xlsm" -RunCount 20 -OutputDir "output\monte_carlo\excel_20runs_20251113_01"
```

### Run without timestamps in folder names:
```powershell
.\run_excel_macro.ps1 -ExcelFile "excel\original.xlsm" -RunCount 5 -AddTimestamp $false
```

## Output Structure

Each run creates a folder with three CSV files:

```
excel_runs/
├── run_1_20251113_143052/
│   ├── results_daily_summary.csv    # From "Results - daily totals" sheet
│   ├── results_minute_level.csv     # From "Results - disaggregated" sheet
│   └── results_aggregated.csv       # From "Results - aggregated" sheet
├── run_2_20251113_143125/
│   ├── results_daily_summary.csv
│   ├── results_minute_level.csv
│   └── results_aggregated.csv
...
```

## Troubleshooting

### PowerShell Execution Policy Error

If you get an error about execution policy, run PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or run the script with:

```powershell
powershell.exe -ExecutionPolicy Bypass -File run_excel_macro.ps1 -ExcelFile "excel\original.xlsm"
```

### Macro Security

Ensure your Excel file has macros enabled. If you get a macro error:

1. Open the Excel file manually
2. Enable macros when prompted
3. Test the macro by running it manually (Alt+F8, select "RunThermalElectricalDemandModel", click Run)
4. Close Excel and try the script again

### Excel Hangs or Doesn't Close

If Excel doesn't close properly:

1. Open Task Manager (Ctrl+Shift+Esc)
2. Find "Microsoft Excel" processes
3. End all Excel tasks
4. Run the script again

## Features

- ✅ Runs Excel macro without displaying Excel window
- ✅ Automatically saves three output sheets as CSV
- ✅ Creates timestamped folders for each run
- ✅ Provides progress and timing information
- ✅ Handles errors gracefully
- ✅ Properly closes Excel and releases COM objects
- ✅ Summary report at the end

## Technical Details

- **Macro Name**: `RunThermalElectricalDemandModel`
- **Required Sheets**:
  - "Results - daily totals"
  - "Results - disaggregated"
  - "Results - aggregated"
- **Excel Visibility**: Hidden (runs in background)
- **Alerts**: Disabled for clean execution

## Notes

- The script does NOT save the Excel file after running the macro
- Each run is independent (the macro is re-run each time)
- Excel is closed and reopened for each batch of runs to ensure clean state
- CSV files use Excel's native CSV export format
