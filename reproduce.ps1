param(
    [ValidateSet("smoke", "full", "dashboard")]
    [string]$Mode = "smoke",

    [string[]]$Symbols = @("BTCUSDT", "ETHUSDT"),

    [switch]$CreateEnv,

    [switch]$Archive
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Resolve-Python {
    $venvPython = Join-Path $Root "env\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    return "python"
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Name"
    & $script:Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Name"
    }
}

if ($CreateEnv) {
    if (-not (Test-Path "env\Scripts\python.exe")) {
        Write-Host "Creating Python 3.11 virtual environment in env/"
        py -3.11 -m venv env
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create env with py -3.11. Install Python 3.11 or create env manually."
        }
    }
}

$script:Python = Resolve-Python
Write-Host "Using Python: $script:Python"

if ($CreateEnv) {
    Invoke-PythonStep "Install research dependencies" @("-m", "pip", "install", "-r", "requirements-research.txt")
}

if ($Mode -eq "dashboard") {
    Invoke-PythonStep "Launch Streamlit dashboard" @("-m", "streamlit", "run", "streamlit_app.py")
    exit 0
}

if ($Mode -eq "smoke") {
    Invoke-PythonStep "Compile Python sources" @("-m", "compileall", "src", "dashboard.py", "streamlit_app.py")
    Invoke-PythonStep "Test Phase 39 fold-local safeguards" @("-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v")
    Invoke-PythonStep "Verify or initialize paper artifacts" @("src\paper_skeleton.py")
    Invoke-PythonStep "Run validation audit" (@("src\validation_audit.py", "--symbols") + $Symbols)
    Write-Host ""
    Write-Host "Smoke reproduction complete."
    exit 0
}

if ($Mode -eq "full") {
    Invoke-PythonStep "Data health check" @("src\check.py")
    Invoke-PythonStep "Generate targets" (@("src\targets.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Train vanilla contrastive encoder" (@("src\train_encoder.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Visualize dense regimes" (@("src\visualize_regimes.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Run classical baselines" (@("src\baselines.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Run alpha models" (@("src\alpha_models.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Regime stability diagnostics" (@("src\regime_stability.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Regime quality diagnostics" (@("src\regime_quality.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Compute budget refresh" (@("src\compute_plan.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Train HMM-guided encoder" (@("src\guided_encoder.py", "--symbols") + $Symbols + @("--epochs", "30"))
    Invoke-PythonStep "Run time-frequency guided prototype" (@("src\guided_encoder.py", "--symbols") + $Symbols + @("--augmentation", "time_frequency", "--epochs", "3"))
    Invoke-PythonStep "Run fold-local regime benchmark" (@("src\walkforward_regimes.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Run robustness matrix" @("src\robustness.py")
    Invoke-PythonStep "Run stress robustness" @("src\robustness_stress.py")
    Invoke-PythonStep "Run statistical tests" @("src\statistical_tests.py")
    Invoke-PythonStep "Run interpretability" (@("src\interpretability.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Run ablation suite" @("src\ablation_suite.py")
    Invoke-PythonStep "Run paper claim tests" @("src\paper_claim_tests.py")
    Invoke-PythonStep "Verify or initialize paper artifacts" @("src\paper_skeleton.py")
    Invoke-PythonStep "Run validation audit" (@("src\validation_audit.py", "--symbols") + $Symbols)
    Invoke-PythonStep "Build backtest dashboard artifacts" @("src\backtest.py")
    Invoke-PythonStep "Compile Python sources" @("-m", "compileall", "src", "dashboard.py", "streamlit_app.py")

    if ($Archive) {
        $runId = "local_phase28_reproduction_" + (Get-Date -Format "yyyyMMdd_HHmmss")
        Invoke-PythonStep "Archive curated run" @(
            "src\archive_run.py",
            "--phase",
            "phase28_reproduction",
            "--run-id",
            $runId,
            "--source-ref",
            "HEAD",
            "--notes",
            "Local Phase 28 full reproduction run."
        )
    }

    Write-Host ""
    Write-Host "Full reproduction complete."
}
