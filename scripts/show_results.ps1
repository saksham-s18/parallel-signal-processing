# scripts/show_results.ps1
# Read-only display of existing benchmark results for Live Presentation
# Does NOT run any benchmarks or modify data.

$benchPath = Join-Path $PSScriptRoot "..\results\tables\full_benchmark_comparison.csv"
$cudaPath  = Join-Path $PSScriptRoot "..\results\tables\cuda_breakdown.csv"
$ompPath   = Join-Path $PSScriptRoot "..\results\tables\omp_thread_scaling.csv"

if (-not (Test-Path $benchPath)) {
    Write-Host "Error: Cannot find $benchPath" -ForegroundColor Red
    exit 1
}

$benchData = Import-Csv $benchPath

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "                  MOVING-AVERAGE PERFORMANCE COMPARISON                         " -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Workloads:"
Write-Host "  Small  : N =    10,000, Window W = 15"
Write-Host "  Medium : N =   100,000, Window W = 31"
Write-Host "  Large  : N = 1,000,000, Window W = 63"
Write-Host "--------------------------------------------------------------------------------"

# Build matrix of implementations
$targets = @(
    @{ Key = "Sequential";     Label = "Sequential" },
    @{ Key = "OpenMP (T=1)";   Label = "OpenMP (1 thread)" },
    @{ Key = "OpenMP (T=8)";   Label = "OpenMP (8 threads)" },
    @{ Key = "OpenMP (T=24)";  Label = "OpenMP (24 threads)" },
    @{ Key = "CUDA Baseline";  Label = "CUDA Baseline" },
    @{ Key = "CUDA Optimized"; Label = "CUDA Optimized (Shared Mem)" }
)

Write-Host ("{0,-30} | {1,14} | {2,14} | {3,14}" -f "Implementation", "Small (ms)", "Medium (ms)", "Large (ms)") -ForegroundColor Green
Write-Host ("-" * 80)

foreach ($t in $targets) {
    $smallRow = $benchData | Where-Object { $_.Implementation -eq $t.Key -and $_.'Input Size' -eq 'Small' }
    $medRow   = $benchData | Where-Object { $_.Implementation -eq $t.Key -and $_.'Input Size' -eq 'Medium' }
    $largeRow = $benchData | Where-Object { $_.Implementation -eq $t.Key -and $_.'Input Size' -eq 'Large' }

    $smallTime = if ($smallRow) { "{0:N4} ms" -f [double]$smallRow.'Avg Time (ms)' } else { "N/A" }
    $medTime   = if ($medRow)   { "{0:N4} ms" -f [double]$medRow.'Avg Time (ms)' } else { "N/A" }
    $largeTime = if ($largeRow) { "{0:N4} ms" -f [double]$largeRow.'Avg Time (ms)' } else { "N/A" }

    Write-Host ("{0,-30} | {1,14} | {2,14} | {3,14}" -f $t.Label, $smallTime, $medTime, $largeTime)
}

Write-Host ("-" * 80)
Write-Host ("{0,-30} | {1,14} | {2,14} | {3,14}" -f "Speedup vs Sequential", "Small Speedup", "Medium Speedup", "Large Speedup") -ForegroundColor Green
Write-Host ("-" * 80)

foreach ($t in $targets | Where-Object { $_.Key -ne "Sequential" }) {
    $smallRow = $benchData | Where-Object { $_.Implementation -eq $t.Key -and $_.'Input Size' -eq 'Small' }
    $medRow   = $benchData | Where-Object { $_.Implementation -eq $t.Key -and $_.'Input Size' -eq 'Medium' }
    $largeRow = $benchData | Where-Object { $_.Implementation -eq $t.Key -and $_.'Input Size' -eq 'Large' }

    $smallSpd = if ($smallRow) { "{0:N2}x" -f [double]$smallRow.'Speedup' } else { "N/A" }
    $medSpd   = if ($medRow)   { "{0:N2}x" -f [double]$medRow.'Speedup' } else { "N/A" }
    $largeSpd = if ($largeRow) { "{0:N2}x" -f [double]$largeRow.'Speedup' } else { "N/A" }

    Write-Host ("{0,-30} | {1,14} | {2,14} | {3,14}" -f $t.Label, $smallSpd, $medSpd, $largeSpd)
}
Write-Host "================================================================================" -ForegroundColor Cyan

# Section 2: CUDA Timing Breakdown
if (Test-Path $cudaPath) {
    $cudaData = Import-Csv $cudaPath
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "             CUDA TIMING BREAKDOWN: MEMORY TRANSFER VS COMPUTE                  " -ForegroundColor Yellow
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ("{0,-16} | {1,-7} | {2,10} | {3,11} | {4,10} | {5,10} | {6,11}" -f "Implementation", "Size", "H2D (ms)", "Kernel (ms)", "D2H (ms)", "E2E (ms)", "Transfer %") -ForegroundColor Green
    Write-Host ("-" * 80)
    foreach ($row in $cudaData) {
        $h2d = "{0:N4}" -f [double]$row.'H2D (ms)'
        $krn = "{0:N4}" -f [double]$row.'Kernel (ms)'
        $d2h = "{0:N4}" -f [double]$row.'D2H (ms)'
        $e2e = "{0:N4}" -f [double]$row.'End-to-End (ms)'
        $xfr = $row.'Transfer Ratio (%)'
        Write-Host ("{0,-16} | {1,-7} | {2,10} | {3,11} | {4,10} | {5,10} | {6,11}" -f $row.Implementation, $row.'Input Size', $h2d, $krn, $d2h, $e2e, $xfr)
    }
    Write-Host "================================================================================" -ForegroundColor Cyan
}

# Section 3: OpenMP Thread Scaling
if (Test-Path $ompPath) {
    $ompData = Import-Csv $ompPath
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "        OPENMP THREAD SCALING ANALYSIS (N = 1,000,000, Window W = 31)          " -ForegroundColor Yellow
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ("{0,8} | {1,14} | {2,10} | {3,16} | {4,12}" -f "Threads", "Time (ms)", "Speedup", "Efficiency (%)", "Status") -ForegroundColor Green
    Write-Host ("-" * 80)
    foreach ($row in $ompData) {
        $t    = $row.Threads
        $time = "{0:N4}" -f [double]$row.AvgTimeMs
        $spd  = "{0:N2}x" -f [double]$row.Speedup
        $eff  = "{0:N1}%" -f [double]$row.EfficiencyPercent
        $st   = $row.Status
        Write-Host ("{0,8} | {1,14} | {2,10} | {3,16} | {4,12}" -f $t, $time, $spd, $eff, $st)
    }
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
}
