[CmdletBinding()]
param([string]$Python = "")

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
if (-not $Python) {
    $hostPython = Join-Path (Join-Path (Split-Path -Parent $RepoRoot) "Umi-OCR") "build\staging\Umi-OCR_Windows_x64\UmiOCR-data\runtime\python.exe"
    $Python = if (Test-Path -LiteralPath $hostPython -PathType Leaf) {
        $hostPython
    } else {
        (Get-Command python.exe -ErrorAction Stop).Source
    }
}
$Python = [IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python interpreter does not exist: $Python" }
$pythonVersion = (& $Python -I -S -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if ($LASTEXITCODE -ne 0) { throw "Unable to determine Python interpreter version: $Python" }
Write-Host "Resolving PaddleOCR-VL plugin lock with Python $pythonVersion."
$report = Join-Path $RepoRoot "build\lock-report.json"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $report) | Out-Null

& $Python -m pip install --dry-run --ignore-installed --report $report --only-binary=:all: -r (Join-Path $RepoRoot "requirements.in")
if ($LASTEXITCODE -ne 0) { throw "Plugin dependency resolution failed." }
& $Python (Join-Path $RepoRoot "tools\report_to_lock.py") $report (Join-Path $RepoRoot "requirements.lock") --comment "Regenerate with Resolve-Lock.ps1." --extra-index-url ""
if ($LASTEXITCODE -ne 0) { throw "Plugin lock generation failed." }
