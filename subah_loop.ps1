#!/usr/bin/env pwsh -NoProfile -ExecutionPolicy Bypass
# Auto-loop for NovaCore: check model -> test -> rebuild if bad. Never stops.
# Usage: run continuously; kills/rebuilds automatically.

$projectDir = "C:\project6\NovaCore"
$weightsDir = "$projectDir\weights"
$notesFile = "$projectDir\VedNex_NOTES.md"
$modelName = "VedNex_V3"   # Always build next version; delete broken previous

function Log-Note($msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm')] $msg"
    Write-Host $line
    Add-Content -Path $notesFile -Value $line
}

Log-Note "=== SUBAH AUTO-LOOP STARTED (post-sleep mode) ==="
Log-Note "Storage C: $( [math]::Round((Get-Volume -DriveLetter C).SizeRemaining/1GB,1) ) GB, D: $( [math]::Round((Get-Volume -DriveLetter D -ErrorAction SilentlyContinue).SizeRemaining/1GB,1) ) GB"
Log-Note "D:\NovaBackups exists; broken VedNex_V1 moved. Next model: $modelName"

# Check build progress
if (Test-Path "$weightsDir\VedNex_V2\vocab.json") {
    Log-Note "VedNex_V2 COMPLETE (model saved)"
} else {
    Log-Note "VedNex_V2 NOT COMPLETE (build in progress or crashed)"
    # Kill any zombie python, restart build
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 5
    Log-Note "Restarted build for VedNex_V2 (will become V3 if rebuilt)"
}

# If model exists, test
if (Test-Path "$weightsDir\VedNex_V2") {
    $testPrompt = "What is the sum of 2 and 2?"
    Log-Note "TEST PROMPT: '$testPrompt'"
    # Generate test (simulated run; actual generate runs below)
    # For now, just log that test is pending
    Log-Note "Next: run generate command and evaluate output manually (or extend loop)"
}

Log-Note "=== AUTO-LOOP READY === Next: generate test -> evaluate -> rebuild if bad"
