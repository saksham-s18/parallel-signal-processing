# PowerShell Build Script for Signal Processing Moving-Average Project
param (
    [string]$Target = "all", # options: all, seq, omp, cuda, opt, clean
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

# Locate vcvars64.bat for MSVC host compiler needed by nvcc
function Get-VcvarsPath {
    $standardPath = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $standardPath) {
        return $standardPath
    }
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath) {
            $found = Join-Path $installPath "VC\Auxiliary\Build\vcvars64.bat"
            if (Test-Path $found) { return $found }
        }
    }
    return $null
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

function Build-CUDA {
    Write-Host "Compiling Baseline CUDA Implementation..." -ForegroundColor Cyan
    if ((Test-Path "src/cuda/main_cuda.cu") -and (Test-Path "src/cuda/moving_average_cuda.cu")) {
        $vcvars = Get-VcvarsPath
        $cmd = if ($vcvars) {
            "call `"$vcvars`" && nvcc -arch=sm_89 -O3 -I./include src/cuda/moving_average_cuda.cu src/cuda/main_cuda.cu src/sequential/moving_average_seq.cpp -o bin/moving_average_cuda.exe"
        } else {
            "nvcc -arch=sm_89 -O3 -I./include src/cuda/moving_average_cuda.cu src/cuda/main_cuda.cu src/sequential/moving_average_seq.cpp -o bin/moving_average_cuda.exe"
        }
        cmd.exe /c $cmd
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Built bin/moving_average_cuda.exe successfully." -ForegroundColor Green
        } else {
            Write-Host "CUDA build failed." -ForegroundColor Red
        }
    } else {
        Write-Host "CUDA source files not found." -ForegroundColor Yellow
    }
}

function Build-Optimized {
    Write-Host "Compiling Shared-Memory Optimized CUDA Implementation..." -ForegroundColor Cyan
    if ((Test-Path "src/optimized/main_opt.cu") -and (Test-Path "src/optimized/moving_average_opt.cu")) {
        $vcvars = Get-VcvarsPath
        $cmd = if ($vcvars) {
            "call `"$vcvars`" && nvcc -arch=sm_89 -O3 -I./include src/optimized/moving_average_opt.cu src/optimized/main_opt.cu src/sequential/moving_average_seq.cpp -o bin/moving_average_opt.exe"
        } else {
            "nvcc -arch=sm_89 -O3 -I./include src/optimized/moving_average_opt.cu src/optimized/main_opt.cu src/sequential/moving_average_seq.cpp -o bin/moving_average_opt.exe"
        }
        cmd.exe /c $cmd
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Built bin/moving_average_opt.exe successfully." -ForegroundColor Green
        } else {
            Write-Host "Optimized CUDA build failed." -ForegroundColor Red
        }
    } else {
        Write-Host "Optimized CUDA source files not found." -ForegroundColor Yellow
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
    "cuda"  { Build-CUDA }
    "opt"   { Build-Optimized }
    "clean" { Clean-Build }
    "all"   { 
        Build-Sequential
        Build-OpenMP
        Build-CUDA
        Build-Optimized
    }
    default { Write-Host "Unknown target: $Target" -ForegroundColor Red }
}
