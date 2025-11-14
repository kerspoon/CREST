@echo off
REM ===============================================================================
REM Example batch file to run the CREST Excel macro runner
REM ===============================================================================
REM
REM Edit this file to customize:
REM - The Excel file path
REM - Number of runs
REM - Output directory
REM
REM ===============================================================================

REM Configuration - EDIT THESE VALUES
set EXCEL_FILE=excel\monte_carlo_base.xlsm
set RUN_COUNT=10
set OUTPUT_DIR=excel_runs

REM ===============================================================================
REM Run the PowerShell script
REM ===============================================================================

echo Running CREST Excel Macro Runner...
echo.
echo Excel File: %EXCEL_FILE%
echo Run Count:  %RUN_COUNT%
echo Output Dir: %OUTPUT_DIR%
echo.

powershell.exe -ExecutionPolicy Bypass -File run_excel_macro.ps1 -ExcelFile "%EXCEL_FILE%" -RunCount %RUN_COUNT% -OutputDir "%OUTPUT_DIR%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ===============================================================================
    echo SUCCESS: All runs completed successfully
    echo ===============================================================================
    echo.
) else (
    echo.
    echo ===============================================================================
    echo ERROR: Some runs failed - check output above
    echo ===============================================================================
    echo.
)

pause
