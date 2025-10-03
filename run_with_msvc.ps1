# run_with_msvc.ps1
# This script sets up the Visual Studio environment and runs your Python script

# Find Visual Studio installation
$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvarsPath = "$vsPath\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $vcvarsPath)) {
    Write-Host "ERROR: Visual Studio Build Tools not found at expected location" -ForegroundColor Red
    Write-Host "Please install Visual Studio Build Tools 2022 from:" -ForegroundColor Yellow
    Write-Host "https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    exit 1
}

# Create a temporary batch file that sets up environment and runs Python
$batchFile = [System.IO.Path]::GetTempFileName() + ".bat"

# Get the full paths for arguments
$pythonArgs = $args | ForEach-Object { 
    if (Test-Path $_) { 
        (Resolve-Path $_).Path 
    } else { 
        $_ 
    } 
}

$script = @"
@echo off
call "$vcvarsPath" > nul
cd /d "$PWD"
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python $($pythonArgs -join ' ')
"@

Set-Content -Path $batchFile -Value $script

# Run the batch file with any arguments passed to this script
& cmd.exe /c $batchFile $args

# Clean up
Remove-Item $batchFile -ErrorAction SilentlyContinue
