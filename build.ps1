# PowerShell Build Script for Signal Processing Moving-Average Project
param (
    [string]$Target = "all", # options: all, seq, omp, clean
    [string]$Config = "Release"
)

$CXX = "g++"
$CXXFLAGS = "-std=c++17 -I./include"
if ($Config -eq "Release") {
    $CXXFLAGS += " -O3 -march=native -DNDEBUG"
} else {
    $CXXFLAGS += " -O0 -g -DDEBUG"
}

# Ensure bin directory exists
if (-not (Test-Path "bin")) {
    New-Item -ItemType Directory -Path "bin" | Out-Null
}

function Build-Sequential {
    Write-Host "Compiling Sequential Baseline..." -ForegroundColor Cyan
    if ((Test-Path "src/sequential/main_seq.cpp") -and (Test-Path "src/sequential/moving_average_seq.cpp")) {
        $sources = @("src/sequential/moving_average_seq.cpp", "src/sequential/main_seq.cpp")
        $argsList = $CXXFLAGS.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) + $sources + @("-o", "bin/moving_average_seq.exe")
        & $CXX $argsList
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Built bin/moving_average_seq.exe successfully." -ForegroundColor Green
        } else {
            Write-Host "Sequential build failed." -ForegroundColor Red
        }
    } else {
        Write-Host "Sequential source files not found." -ForegroundColor Yellow
    }
}

function Build-OpenMP {
    Write-Host "Compiling OpenMP Parallel Implementation..." -ForegroundColor Cyan
    if ((Test-Path "src/openmp/main_omp.cpp") -and (Test-Path "src/openmp/moving_average_omp.cpp")) {
        $sources = @(
            "src/openmp/moving_average_omp.cpp",
            "src/sequential/moving_average_seq.cpp",
            "src/openmp/main_omp.cpp"
        )
        $ompFlags = $CXXFLAGS.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries) + @("-fopenmp") + $sources + @("-o", "bin/moving_average_omp.exe")
        & $CXX $ompFlags
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Built bin/moving_average_omp.exe successfully." -ForegroundColor Green
        } else {
            Write-Host "OpenMP build failed." -ForegroundColor Red
        }
    } else {
        Write-Host "OpenMP source files not found." -ForegroundColor Yellow
    }
}

function Clean-Build {
    Write-Host "Cleaning build artifacts..." -ForegroundColor Cyan
    if (Test-Path "bin") {
        Remove-Item -Recurse -Force "bin"
    }
    Write-Host "Clean complete." -ForegroundColor Green
}

switch ($Target) {
    "seq"   { Build-Sequential }
    "omp"   { Build-OpenMP }
    "clean" { Clean-Build }
    "all"   { 
        Build-Sequential
        Build-OpenMP
    }
    default { Write-Host "Unknown target: $Target" -ForegroundColor Red }
}
