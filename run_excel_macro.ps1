# ===============================================================================
# CREST Excel Macro Runner
# ===============================================================================
# This script repeatedly runs the CREST demand model macro in Excel and saves
# the output sheets as CSV files.
#
# Usage:
#   .\run_excel_macro.ps1 -ExcelFile "path\to\file.xlsm" -RunCount 10 -OutputDir "output"
#
# Parameters:
#   -ExcelFile   : Path to the Excel .xlsm file (required)
#   -RunCount    : Number of times to run the macro (default: 1)
#   -OutputDir   : Directory to save CSV outputs (default: ".\excel_runs")
#   -AddTimestamp: Add timestamp to run folders (default: $true)
# ===============================================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$ExcelFile,

    [Parameter(Mandatory=$false)]
    [int]$RunCount = 1,

    [Parameter(Mandatory=$false)]
    [string]$OutputDir = ".\excel_runs",

    [Parameter(Mandatory=$false)]
    [bool]$AddTimestamp = $true
)

# ===============================================================================
# Configuration
# ===============================================================================

$MacroName = "RunThermalElectricalDemandModel"
$SheetNames = @{
    "Results - daily totals" = "results_daily_summary.csv"
    "Results - disaggregated" = "results_minute_level.csv"
    "Results - aggregated" = "results_aggregated.csv"
}

# ===============================================================================
# Validation
# ===============================================================================

# Check if Excel file exists
if (-not (Test-Path $ExcelFile)) {
    Write-Error "Excel file not found: $ExcelFile"
    exit 1
}

# Get absolute path
$ExcelFile = (Resolve-Path $ExcelFile).Path

# Check if file is .xlsm
if (-not ($ExcelFile -like "*.xlsm")) {
    Write-Warning "File does not have .xlsm extension. This may not work if macros are not enabled."
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    Write-Host "Created output directory: $OutputDir"
}

# ===============================================================================
# Helper Functions
# ===============================================================================

function Save-SheetAsCSV {
    param(
        [Parameter(Mandatory=$true)]
        $Excel,

        [Parameter(Mandatory=$true)]
        $Workbook,

        [Parameter(Mandatory=$true)]
        [string]$SheetName,

        [Parameter(Mandatory=$true)]
        [string]$OutputPath
    )

    $tempWorkbook = $null
    try {
        $sheet = $Workbook.Sheets.Item($SheetName)

        # Check if sheet is empty
        $usedRange = $sheet.UsedRange
        if ($null -eq $usedRange -or $usedRange.Rows.Count -eq 0) {
            Write-Warning "  Sheet '$SheetName' is empty - skipping"
            return $false
        }

        # Ensure output path is absolute
        if (-not [System.IO.Path]::IsPathRooted($OutputPath)) {
            $OutputPath = Join-Path (Get-Location) $OutputPath
        }
        $OutputPath = [System.IO.Path]::GetFullPath($OutputPath)

        # Delete existing file if it exists (Excel can't overwrite CSV files)
        if (Test-Path $OutputPath) {
            Remove-Item $OutputPath -Force
        }

        # Create a new temporary workbook
        $tempWorkbook = $Excel.Workbooks.Add()
        $tempSheet = $tempWorkbook.Sheets.Item(1)

        # Copy the used range (data only, no charts/objects)
        $usedRange.Copy() | Out-Null

        # Paste values only into the temporary sheet
        $tempSheet.Range("A1").PasteSpecial(-4163) | Out-Null  # xlPasteValues = -4163

        # Save the temporary workbook as CSV (xlCSV = 6)
        $tempWorkbook.SaveAs($OutputPath, 6)

        $fileSize = (Get-Item $OutputPath).Length
        Write-Host "  Saved: $SheetName -> $(Split-Path $OutputPath -Leaf) ($fileSize bytes)"
        return $true
    }
    catch {
        Write-Error "  Failed to save sheet '$SheetName': $_"
        Write-Error "  Error details: $($_.Exception.Message)"
        return $false
    }
    finally {
        # Clean up temporary workbook
        if ($null -ne $tempWorkbook) {
            $tempWorkbook.Close($false)
            [System.Runtime.Interopservices.Marshal]::ReleaseComObject($tempWorkbook) | Out-Null
        }
    }
}

function Run-ExcelMacro {
    param(
        [Parameter(Mandatory=$true)]
        $Excel,

        [Parameter(Mandatory=$true)]
        $Workbook,

        [Parameter(Mandatory=$true)]
        [string]$MacroName
    )

    try {
        Write-Host "  Running macro: $MacroName"
        $startTime = Get-Date

        # Run the macro
        $Excel.Run($MacroName)

        $duration = (Get-Date) - $startTime
        Write-Host "  Macro completed in $($duration.TotalSeconds) seconds"
        return $true
    }
    catch {
        Write-Error "  Failed to run macro '$MacroName': $_"
        Write-Error "  Error details: $($_.Exception.Message)"
        return $false
    }
}

# ===============================================================================
# Main Script
# ===============================================================================

Write-Host ""
Write-Host "==============================================================================="
Write-Host "CREST Excel Macro Runner"
Write-Host "==============================================================================="
Write-Host ""
Write-Host "Excel File:    $ExcelFile"
Write-Host "Run Count:     $RunCount"
Write-Host "Output Dir:    $OutputDir"
Write-Host "Macro:         $MacroName"
Write-Host ""

# Create Excel COM object
Write-Host "Initializing Excel..."
$Excel = New-Object -ComObject Excel.Application
$Excel.Visible = $false
$Excel.DisplayAlerts = $false
$Excel.ScreenUpdating = $false
$Excel.EnableEvents = $true  # Need this enabled for macros to work properly

# Track successful runs
$successCount = 0
$failCount = 0

try {
    # Open the workbook
    Write-Host "Opening workbook: $(Split-Path $ExcelFile -Leaf)"
    $Workbook = $Excel.Workbooks.Open($ExcelFile)
    Write-Host "Workbook opened successfully"
    Write-Host ""

    # Run the macro multiple times
    for ($runIndex = 1; $runIndex -le $RunCount; $runIndex++) {
        Write-Host "==============================================================================="
        Write-Host "Run $runIndex of $RunCount"
        Write-Host "==============================================================================="

        # Create run-specific output directory
        if ($AddTimestamp) {
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $runDir = Join-Path $OutputDir "run_${runIndex}_${timestamp}"
        } else {
            $runDir = Join-Path $OutputDir "run_${runIndex}"
        }

        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
        Write-Host "Output directory: $runDir"
        Write-Host ""

        # Run the macro
        $macroSuccess = Run-ExcelMacro -Excel $Excel -Workbook $Workbook -MacroName $MacroName

        if (-not $macroSuccess) {
            Write-Error "Macro execution failed for run $runIndex - stopping"
            $failCount++
            break
        }

        Write-Host ""
        Write-Host "Saving output sheets..."

        # Save each sheet as CSV
        $allSheetsSuccess = $true
        foreach ($sheetEntry in $SheetNames.GetEnumerator()) {
            $sheetName = $sheetEntry.Key
            $csvFileName = $sheetEntry.Value
            $csvPath = Join-Path $runDir $csvFileName

            $saveSuccess = Save-SheetAsCSV -Excel $Excel -Workbook $Workbook -SheetName $sheetName -OutputPath $csvPath

            if (-not $saveSuccess) {
                $allSheetsSuccess = $false
            }
        }

        if ($allSheetsSuccess) {
            $successCount++
            Write-Host ""
            Write-Host "Run $runIndex completed successfully" -ForegroundColor Green
        } else {
            $failCount++
            Write-Host ""
            Write-Host "Run $runIndex completed with errors" -ForegroundColor Yellow
        }

        Write-Host ""
    }
}
catch {
    Write-Error "Critical error: $_"
    Write-Error "Stack trace: $($_.ScriptStackTrace)"
}
finally {
    # Clean up
    Write-Host "==============================================================================="
    Write-Host "Cleanup"
    Write-Host "==============================================================================="

    if ($null -ne $Workbook) {
        Write-Host "Closing workbook (without saving)..."
        $Workbook.Close($false)
    }

    if ($null -ne $Excel) {
        Write-Host "Quitting Excel..."
        $Excel.Quit()
        [System.Runtime.Interopservices.Marshal]::ReleaseComObject($Excel) | Out-Null
    }

    # Force garbage collection to release Excel COM objects
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()

    Write-Host "Cleanup complete"
}

# ===============================================================================
# Summary
# ===============================================================================

Write-Host ""
Write-Host "==============================================================================="
Write-Host "Summary"
Write-Host "==============================================================================="
Write-Host "Total runs:       $RunCount"
Write-Host "Successful:       $successCount" -ForegroundColor Green
Write-Host "Failed:           $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })
Write-Host "Output location:  $(Resolve-Path $OutputDir)"
Write-Host "==============================================================================="
Write-Host ""

# Exit with appropriate code
if ($failCount -gt 0) {
    exit 1
} else {
    exit 0
}
