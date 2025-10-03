# setup_cuda_env.ps1
# Run this script once per PowerShell session to set up CUDA development environment
# Usage: . .\setup_cuda_env.ps1

Write-Host "Setting up CUDA development environment..." -ForegroundColor Cyan

# Find Visual Studio installation
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvarsPath = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $vcvarsPath)) {
    Write-Host "ERROR: Visual Studio Build Tools not found" -ForegroundColor Red
    exit 1
}

# Create a temporary batch file to capture environment variables
$tempBat = [System.IO.Path]::GetTempFileName() + ".bat"
$tempTxt = [System.IO.Path]::GetTempFileName() + ".txt"

@"
@echo off
call "$vcvarsPath" > nul
set > "$tempTxt"
"@ | Set-Content $tempBat

# Run the batch file
cmd /c $tempBat

# Parse and import environment variables
Get-Content $tempTxt | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        $name = $matches[1]
        $value = $matches[2]
        Set-Item -Path "env:$name" -Value $value -Force
    }
}

# Clean up
Remove-Item $tempBat -ErrorAction SilentlyContinue
Remove-Item $tempTxt -ErrorAction SilentlyContinue

# Verify cl.exe is now available
if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
    Write-Host "✓ Visual Studio compiler (cl.exe) is now available" -ForegroundColor Green
    Write-Host "✓ CUDA development environment ready" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run:" -ForegroundColor Yellow
    Write-Host "  python hello_cuda.py" -ForegroundColor White
    Write-Host "  python minimal_cuda.py" -ForegroundColor White
    Write-Host "  python simplest_cuda_demo.py" -ForegroundColor White
    Write-Host "  python app.py" -ForegroundColor White
} else {
    Write-Host "✗ Failed to set up environment" -ForegroundColor Red
}
