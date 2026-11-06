[CmdletBinding()]
param(
    [string]$StagingRoot = "",
    [string]$OutputRoot = "",
    [string]$Python = "",
    [string]$VendorRoot = "",
    [string]$PaddleOCRWheel = "",
    [string]$PaddleXWheel = "",
    [string]$RuntimeLock = "",
    [string]$LlamaRuntimeBundle = "",
    [ValidateSet("loose", "stored")][string]$RuntimeLayout = "loose",
    [ValidateSet("minimal", "complete")][string]$DistributionLayout = "minimal",
    [string]$MinimalRoot = "",
    [string]$DistributionRecord = "",
    [switch]$SkipTests,
    [switch]$SkipArchive,
    [switch]$LocalCandidate
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path $PSScriptRoot).Path
$BuildRoot = [IO.Path]::GetFullPath((Join-Path $RepoRoot "build"))
if (-not $LlamaRuntimeBundle) {
    throw "LlamaRuntimeBundle is required. Provide the reviewed local llama.cpp runtime bundle; no external-server fallback or download is allowed."
}
$LlamaRuntimeBundle = [IO.Path]::GetFullPath($LlamaRuntimeBundle)
if (-not (Test-Path -LiteralPath (Join-Path $LlamaRuntimeBundle "runtime-manifest.json") -PathType Leaf)) {
    throw "Bundled llama.cpp runtime manifest is missing: $LlamaRuntimeBundle"
}
$llamaBundleTool = Join-Path $RepoRoot "tools\bundle_llama_runtime.py"
if (-not (Test-Path -LiteralPath $llamaBundleTool -PathType Leaf)) {
    throw "llama.cpp bundle validator is missing: $llamaBundleTool"
}
$HostRepoRoot = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $RepoRoot) "Umi-OCR"))
$LicenseRoot = Join-Path $HostRepoRoot "LICENSES"
$LicenseFiles = @(
    "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_COMPONENTS.md",
    "LGPL-2.1.txt", "LGPL-3.0.txt", "GPL-3.0.txt", "PyMuPDF-AGPL-3.0.txt"
)
$PluginLicenseFiles = @("VL_EXTRA_THIRD_PARTY_NOTICES.md")
$Version = (Get-Content -Raw (Join-Path $RepoRoot "VERSION")).Trim()
if (-not $StagingRoot) { $StagingRoot = Join-Path $BuildRoot "staging\structured-document-plugin" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot "release" }
$StagingRoot = [IO.Path]::GetFullPath($StagingRoot)
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
if (-not $VendorRoot) {
    throw "VendorRoot is required. Provide a reviewed private Python runtime closure; this plugin never installs dependencies into the host."
}
$VendorRoot = [IO.Path]::GetFullPath($VendorRoot)
if (-not (Test-Path -LiteralPath $VendorRoot -PathType Container)) {
    throw "VendorRoot does not exist: $VendorRoot"
}
if (-not $RuntimeLock) {
    throw "RuntimeLock is required. Provide the reviewed, hash-pinned lock used to assemble VendorRoot."
}
$RuntimeLock = [IO.Path]::GetFullPath($RuntimeLock)
if (-not (Test-Path -LiteralPath $RuntimeLock -PathType Leaf)) {
    throw "RuntimeLock does not exist: $RuntimeLock"
}
if (-not (Test-Path -LiteralPath $LicenseRoot -PathType Container)) {
    throw "LicenseRoot does not exist: $LicenseRoot"
}

# The GPU OCR closure is intentionally stripped of the optional PaddleOCR-VL
# API.  The document plugin restores the exact PaddleOCR 3.7.0 wheel files
# needed by PaddleOCRVL, keeping this package independent while preserving the
# hash-pinned third-party closure.
$PaddleOCRWheelSha256 = "c0f0a81ad4112727f30c6fcf986ac0ef6a120d31ee0991a01fae0357ee32d338"
$PaddleXWheelSha256 = "f1678bf650bbaccfd8f0d4e49d0ae631b4685c829fdae6e802ccd90d4fcb9a7f"
if (-not $PaddleOCRWheel) {
    $PaddleOCRWheel = Join-Path $BuildRoot "paddleocr-wheel-inspect\paddleocr-3.7.0-py3-none-any.whl"
}
if (-not $PaddleXWheel) {
    $PaddleXWheel = Join-Path $BuildRoot "cpu-onnx-wheels-20260808\paddlex-3.7.2-py3-none-any.whl"
}
$PaddleOCRWheel = [IO.Path]::GetFullPath($PaddleOCRWheel)
$PaddleXWheel = [IO.Path]::GetFullPath($PaddleXWheel)
if (-not (Test-Path -LiteralPath $PaddleOCRWheel -PathType Leaf)) {
    throw "PaddleOCR 3.7.0 wheel is required to restore PaddleOCRVL: $PaddleOCRWheel"
}
if (-not (Test-Path -LiteralPath $PaddleXWheel -PathType Leaf)) {
    throw "PaddleX 3.7.2 wheel is required to restore PaddleOCRVL: $PaddleXWheel"
}
$actualPaddleOCRWheelSha256 = (Get-FileHash -LiteralPath $PaddleOCRWheel -Algorithm SHA256).Hash.ToLower()
$actualPaddleXWheelSha256 = (Get-FileHash -LiteralPath $PaddleXWheel -Algorithm SHA256).Hash.ToLower()
if ($actualPaddleOCRWheelSha256 -ne $PaddleOCRWheelSha256) {
    throw "PaddleOCR wheel SHA-256 mismatch: expected $PaddleOCRWheelSha256, got $actualPaddleOCRWheelSha256"
}
if ($actualPaddleXWheelSha256 -ne $PaddleXWheelSha256) {
    throw "PaddleX wheel SHA-256 mismatch: expected $PaddleXWheelSha256, got $actualPaddleXWheelSha256"
}

$pluginName = "win32_PaddleOCRVL_1_6"
if (-not $StagingRoot.StartsWith($BuildRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw "StagingRoot must stay inside this plugin project's build/: $StagingRoot"
}
if (Test-Path -LiteralPath $StagingRoot) {
    throw "Use a new staging directory; existing builds are never deleted: $StagingRoot"
}
$WheelInputsRoot = "${StagingRoot}-wheel-inputs"
if (Test-Path -LiteralPath $WheelInputsRoot) {
    throw "Use a new wheel extraction directory; existing inputs are never deleted: $WheelInputsRoot"
}
$pluginRoot = Join-Path $StagingRoot "UmiOCR-data\\plugins\\$pluginName"
$sourceRoot = Join-Path $StagingRoot "source"
New-Item -ItemType Directory -Force -Path $StagingRoot,$OutputRoot | Out-Null
$sourceFiles = @(
    "Build-Plugin.ps1", "Resolve-Lock.ps1", "README.md", "RELEASE_NOTES.md",
    "LICENSE", "COPYING", "NOTICE.md", "VERSION", "requirements.in",
    "requirements.lock", "requirements-vendor.lock", "requirements-vl-extra.lock", "tests\test_plugin.py", "tests\test_vl_table_closure.py",
    "tools\report_to_lock.py", "tools\patch_paddlex_vl_3_7_2.py", "tools\bundle_llama_runtime.py", "tests\test_bundle_llama_runtime.py", "tests\test_model_settings_vl.py", "SOURCE-CODE.md"
)
foreach ($relative in $sourceFiles) {
    $source = Join-Path $RepoRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
    $target = Join-Path $sourceRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -Force -LiteralPath $source -Destination $target
}
Copy-Item -Recurse -Force -LiteralPath (Join-Path $RepoRoot $pluginName) -Destination $pluginRoot
$sourcePluginRoot = Join-Path $sourceRoot $pluginName
New-Item -ItemType Directory -Force -Path $sourcePluginRoot | Out-Null
# tracked files may already have created source/<pluginName>; merge its contents
# instead of Copy-Item nesting a second source/<pluginName>/<pluginName> level.
Copy-Item -Recurse -Force -Path (Join-Path $RepoRoot "$pluginName\*") -Destination $sourcePluginRoot
$vendorLock = Join-Path $RepoRoot "requirements-vendor.lock"
$vlExtraLock = Join-Path $RepoRoot "requirements-vl-extra.lock"
$pipBootstrap = Join-Path $HostRepoRoot "build\cache\pip-bootstrap"
$pipRunner = Join-Path $HostRepoRoot "build\tools\run_pip.py"
$hostPython = Join-Path $HostRepoRoot "build\staging\Umi-OCR_Windows_x64\UmiOCR-data\runtime\python.exe"
$vendorPython = if ($Python) {
    $Python
} elseif (Test-Path -LiteralPath $hostPython -PathType Leaf) {
    # The plugin worker inherits this portable interpreter at runtime.  Using
    # it for wheel installation keeps native extensions on the same CPython
    # ABI instead of silently selecting the developer machine's Python.
    $hostPython
} else {
    (Get-Command python.exe -ErrorAction Stop).Source
}
$vendorPython = [IO.Path]::GetFullPath($vendorPython)
if (-not (Test-Path -LiteralPath $vendorPython -PathType Leaf)) {
    throw "Python interpreter does not exist: $vendorPython"
}
$pythonVersion = (& $vendorPython -I -S -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
if ($LASTEXITCODE -ne 0 -or $pythonVersion -notmatch '^3\.\d+$') {
    throw "Unable to determine the selected Python interpreter version: $vendorPython"
}
$expectedAbi = $pythonVersion.Replace('.', '')
$abiTags = @(
    Get-ChildItem -Recurse -File -LiteralPath $VendorRoot -Filter "*.pyd" -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.Name -match 'cp(?<abi>\d{2,3})-win_amd64\.pyd$') { $Matches.abi }
        } | Sort-Object -Unique
)
if ($abiTags.Count -and $abiTags -notcontains $expectedAbi) {
    throw "VendorRoot contains CPython ABI cp$($abiTags -join ', cp'), but selected Python is $pythonVersion. Provide a matching -Python interpreter and vendor closure."
}
Write-Host "Using Python $pythonVersion for PaddleOCR-VL vendor installation."
# Keep every subsequent patch and regression step on the same interpreter;
# otherwise an empty -Python parameter would silently switch back to the
# developer machine's Python after the vendor install.
$Python = $vendorPython
$llamaRuntimeTarget = Join-Path $pluginRoot "resources\llama.cpp"
& $vendorPython -I -S $llamaBundleTool --bundle $LlamaRuntimeBundle --copy-to $llamaRuntimeTarget
if ($LASTEXITCODE -ne 0) { throw "Bundled llama.cpp manifest/hash/dependency closure verification failed." }
$llamaManifestHash = (Get-FileHash -LiteralPath (Join-Path $llamaRuntimeTarget "runtime-manifest.json") -Algorithm SHA256).Hash.ToLowerInvariant()
$vendorTarget = Join-Path $pluginRoot "vendor"
New-Item -ItemType Directory -Force -Path $vendorTarget | Out-Null
Copy-Item -Recurse -Force -Path (Join-Path $VendorRoot "*") -Destination $vendorTarget
if (-not (Test-Path -LiteralPath $vendorLock -PathType Leaf) -or
    -not (Test-Path -LiteralPath $vlExtraLock -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $pipBootstrap "pip\__init__.py") -PathType Leaf) -or
    -not (Test-Path -LiteralPath $pipRunner -PathType Leaf)) {
    throw "PaddleOCR-VL vendor lock bootstrap is incomplete."
}
& $vendorPython -I -S $pipRunner $pipBootstrap install --require-hashes --no-deps --no-compile --target $vendorTarget -r $vendorLock
if ($LASTEXITCODE -ne 0) { throw "PaddleOCR-VL client dependency installation failed." }
& $vendorPython -I -S $pipRunner $pipBootstrap install --require-hashes --no-deps --no-compile --target $vendorTarget -r $vlExtraLock
if ($LASTEXITCODE -ne 0) { throw "PaddleOCR-VL title/table restructuring dependency installation failed." }
# 官方跨页表格合并会延迟导入 bs4；构建时验证闭包，而非等整篇文档才失败。
& $vendorPython -I -B -S -c "import sys; sys.path.insert(0,sys.argv[1]); from bs4 import BeautifulSoup; import soupsieve; table=BeautifulSoup('<table><tr><td>closure</td></tr></table>','html.parser'); assert table.select_one('td').get_text() == 'closure'; print('HTML table dependency closure verified')" $vendorTarget
if ($LASTEXITCODE -ne 0) { throw "PaddleOCR-VL cross-page HTML dependency closure failed." }

$wheelExtract = Join-Path $WheelInputsRoot "paddleocr-wheel"
New-Item -ItemType Directory -Force -Path $wheelExtract | Out-Null
Add-Type -AssemblyName System.IO.Compression.FileSystem
[IO.Compression.ZipFile]::ExtractToDirectory($PaddleOCRWheel, $wheelExtract)
$wheelPaddleOCR = Join-Path $wheelExtract "paddleocr"
if (-not (Test-Path -LiteralPath $wheelPaddleOCR -PathType Container)) {
    throw "PaddleOCR wheel has no paddleocr package: $PaddleOCRWheel"
}
$paddleOCRVLFiles = @(
    "__init__.py",
    "_pipelines\__init__.py",
    "_pipelines\paddleocr_vl.py",
    "_api_client\__init__.py",
    "_api_client\_async_http.py",
    "_api_client\_async_poller.py",
    "_api_client\_core.py",
    "_api_client\_http.py",
    "_api_client\_naming.py",
    "_api_client\_poller.py",
    "_api_client\_resources.py",
    "_api_client\async_client.py",
    "_api_client\cli.py",
    "_api_client\client.py",
    "_api_client\errors.py",
    "_api_client\models.py",
    "_api_client\results.py"
)
foreach ($relative in $paddleOCRVLFiles) {
    $source = Join-Path $wheelPaddleOCR $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "PaddleOCRVL wheel file is missing: $relative"
    }
    $target = Join-Path $vendorTarget (Join-Path "paddleocr" $relative)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -Force -LiteralPath $source -Destination $target
}

$paddlexExtract = Join-Path $WheelInputsRoot "paddlex-wheel"
New-Item -ItemType Directory -Force -Path $paddlexExtract | Out-Null
[IO.Compression.ZipFile]::ExtractToDirectory($PaddleXWheel, $paddlexExtract)
$wheelPaddleX = Join-Path $paddlexExtract "paddlex"
if (-not (Test-Path -LiteralPath $wheelPaddleX -PathType Container)) {
    throw "PaddleX wheel has no paddlex package: $PaddleXWheel"
}
$existingPaddleX = Join-Path $vendorTarget "paddlex"
if (Test-Path -LiteralPath $existingPaddleX) {
    Remove-Item -Recurse -Force -LiteralPath $existingPaddleX
}
Copy-Item -Recurse -Force -LiteralPath $wheelPaddleX -Destination $existingPaddleX
$bundledLayoutConfigs = Get-ChildItem -Recurse -File -LiteralPath $existingPaddleX |
    Where-Object { $_.Name -like "PP-DocLayoutV3*" }
foreach ($file in $bundledLayoutConfigs) {
    Remove-Item -Force -LiteralPath $file.FullName
}
$patchPaddleX = Join-Path $HostRepoRoot "build\tools\patch_paddlex_3_7_2.py"
if (Test-Path -LiteralPath $patchPaddleX -PathType Leaf) {
    $patchPython = if ($Python) { $Python } else { (Get-Command python.exe -ErrorAction Stop).Source }
    & $patchPython -I -S $patchPaddleX $vendorTarget
    if ($LASTEXITCODE -ne 0) { throw "PaddleX compatibility patch failed after VL restore." }
}
$patchPaddleXVL = Join-Path $RepoRoot "tools\patch_paddlex_vl_3_7_2.py"
& $vendorPython -I -S $patchPaddleXVL $vendorTarget
if ($LASTEXITCODE -ne 0) { throw "PaddleX VL dependency-scope patch failed." }
$paddleXVLPath = Join-Path $vendorTarget "paddlex\inference\pipelines\paddleocr_vl\pipeline.py"
$paddleXVLHash = (Get-FileHash -LiteralPath $paddleXVLPath -Algorithm SHA256).Hash.ToLower()
& $vendorPython -I -B -S (Join-Path $RepoRoot "tests/test_vl_table_closure.py") $vendorTarget
if ($LASTEXITCODE -ne 0) { throw "PaddleX cross-page table merge dependency verification failed." }
# 小文件聚合不按后缀删减依赖。保留头文件、类型信息、静态库和源码材料；
# 下列具名模型配置排除仍属于原有独立运行闭包，不计为本轮小文件收益。
foreach ($pattern in @("*PP-OCRv2*", "*PP-OCRv3*", "*PP-OCRv4*", "*PP-OCRv5*", "*PP-OCRv6_small*", "*PP-OCRv6_tiny*", "PP-DocLayoutV3*")) {
    Get-ChildItem -Recurse -Force -LiteralPath $vendorTarget -Filter $pattern -ErrorAction SilentlyContinue |
        Remove-Item -Force
}
Copy-Item -Force -LiteralPath $RuntimeLock -Destination (Join-Path $pluginRoot "runtime-requirements.lock")
Copy-Item -Force -LiteralPath $vlExtraLock -Destination (Join-Path $pluginRoot "runtime-vl-extra.lock")
foreach ($name in @("COPYING", "LICENSE", "NOTICE.md")) {
    Copy-Item -Force -LiteralPath (Join-Path $RepoRoot $name) -Destination $StagingRoot
}
New-Item -ItemType Directory -Force -Path (Join-Path $StagingRoot "LICENSES"), (Join-Path $sourceRoot "LICENSES") | Out-Null
foreach ($name in $LicenseFiles) {
    Copy-Item -Force -LiteralPath (Join-Path $LicenseRoot $name) -Destination (Join-Path $StagingRoot "LICENSES")
    Copy-Item -Force -LiteralPath (Join-Path $LicenseRoot $name) -Destination (Join-Path $sourceRoot "LICENSES")
}
foreach ($name in $PluginLicenseFiles) {
    $source = Join-Path $RepoRoot $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Plugin license notice is missing: $name" }
    Copy-Item -Force -LiteralPath $source -Destination (Join-Path $StagingRoot "LICENSES")
    Copy-Item -Force -LiteralPath $source -Destination (Join-Path $sourceRoot "LICENSES")
}

$commit = (& git -C $RepoRoot rev-parse HEAD).Trim()
$sourceWorktreeDirty = [bool](& git -C $RepoRoot status --porcelain=v1)
# 未打发行标签的本地候选允许 pending；Stop 下 native 无输出可为 Null，必须先判空。
$tag = & git -C $RepoRoot describe --tags --exact-match HEAD 2>$null
if ($null -ne $tag) { $tag = $tag.Trim() }
if (-not $tag) { $tag = "v$Version (local tag pending at build time)" }
$lockHash = (Get-FileHash -LiteralPath $RuntimeLock -Algorithm SHA256).Hash.ToLower()
@"
# PaddleOCR-VL $Version local-candidate source notice

This archive retains the minimal corresponding plugin source under source/;
it does not contain GGUF, mmproj, or PP-DocLayoutV3 weights.
The fixed llama.cpp runtime, its licenses and source/provenance are included in
``$pluginName/resources/llama.cpp/`` inside the installable plugin.

- Plugin repository commit: $commit
- Uncommitted source changes included: $sourceWorktreeDirty. The commit alone is not the artifact identity; retained source files and runtime hashes bind this local candidate.
- Local tag: $tag
- Upstream/reference repository: https://github.com/Dominic485649/Umi-OCR-Plugin
- Public source URL: **PENDING** (no user-owned $Version remote is published)
- Build entry: Build-Plugin.ps1 -LocalCandidate
- Runtime lock SHA-256: $lockHash
- Bundled llama.cpp manifest SHA-256: $llamaManifestHash
- Bundled llama.cpp build identity, exact source archive, build argv, binary
  origins and license terms: ``resources/llama.cpp/source/`` and ``LICENSES/``.
- VL extra lock SHA-256: $((Get-FileHash -LiteralPath $vlExtraLock -Algorithm SHA256).Hash.ToLower())
- PaddleOCR 3.7.0 wheel: https://pypi.org/project/paddleocr/3.7.0/
  (SHA-256: $PaddleOCRWheelSha256); restored PaddleOCRVL files come from this
  exact wheel.
- PaddleX 3.7.2 wheel: https://pypi.org/project/paddlex/3.7.2/
  (SHA-256: $PaddleXWheelSha256); the VL package contains this independent
  locked PaddleX runtime after the recorded compatibility patch.
- PaddleX VL source patch: tools/patch_paddlex_vl_3_7_2.py narrows the upstream
  ``ocr`` gate to the bundled ``ocr-core`` feature for the llama.cpp-server
  path; patched file SHA-256: $paddleXVLHash.
- Other third-party source URLs/revisions/hashes: recorded by the included
  lock and package metadata; no unpinned source is implied.

The candidate is not approved for public release until the exact source URL is
available without an additional fee.
"@ | Set-Content -Encoding utf8 (Join-Path $StagingRoot "SOURCE-CODE.md")
Copy-Item -Force -LiteralPath (Join-Path $StagingRoot "SOURCE-CODE.md") -Destination (Join-Path $sourceRoot "SOURCE-CODE.md")

# Reject links before canonicalizing paths; only the exact verified bundle executable is allowed.
$ancestor = Get-Item -Force -LiteralPath $StagingRoot
while ($null -ne $ancestor) {
    if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse point in staging ancestry: $($ancestor.FullName)" }
    $ancestor = $ancestor.Parent
}
$stagingEntries = @(Get-ChildItem -Recurse -Force -LiteralPath $StagingRoot)
$linkedEntries = @($stagingEntries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
if ($linkedEntries) { throw "Reparse point in plugin staging: $($linkedEntries.FullName -join ', ')" }
$forbidden = $stagingEntries | Where-Object {
    $_.Name -like "*.gguf" -or $_.Name -like "*.mmproj" -or $_.Name -like "PP-DocLayoutV3*"
}
if ($forbidden) { throw "Model asset found in plugin staging: $($forbidden.FullName -join ', ')" }
$expectedServer = [IO.Path]::GetFullPath((Join-Path $llamaRuntimeTarget "llama-server.exe"))
$unexpectedServers = $stagingEntries | Where-Object { -not $_.PSIsContainer -and $_.Name -ieq "llama-server.exe" } | Where-Object {
    -not ([IO.Path]::GetFullPath($_.FullName)).Equals($expectedServer, [StringComparison]::OrdinalIgnoreCase)
}
if ($unexpectedServers) { throw "llama-server outside verified runtime bundle: $($unexpectedServers.FullName -join ', ')" }
& $vendorPython -I -S $llamaBundleTool --validate $llamaRuntimeTarget
if ($LASTEXITCODE -ne 0) { throw "Installed llama.cpp runtime validation failed." }
$sourceRequired = @(
    "Build-Plugin.ps1", "Resolve-Lock.ps1", "README.md", "RELEASE_NOTES.md",
    "LICENSE", "COPYING", "NOTICE.md", "VERSION", "requirements.in",
    "requirements.lock", "requirements-vendor.lock", "requirements-vl-extra.lock", "tests\test_plugin.py", "tests\test_vl_table_closure.py",
    "tools\report_to_lock.py", "tools\patch_paddlex_vl_3_7_2.py", "tools\bundle_llama_runtime.py", "tests\test_bundle_llama_runtime.py", "tests\test_model_settings_vl.py"
)
foreach ($relative in $sourceRequired) {
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot $relative) -PathType Leaf)) {
        throw "Required plugin source-delivery file is missing: $relative"
    }
}
$pluginRequired = @(
    "__init__.py", "umi-plugin.json", "i18n.csv", "api.py", "ocr_api.py", "worker.py", "runtime.py",
    "checks.py", "health.py", "page.py", "service.py", "StructuredDocument.qml", "runtime-requirements.lock", "runtime-vl-extra.lock"
)
foreach ($relative in $pluginRequired) {
    if (-not (Test-Path -LiteralPath (Join-Path $pluginRoot $relative) -PathType Leaf)) {
        throw "Required installable plugin file is missing: $relative"
    }
}

if (-not $SkipTests) {
    if (-not $Python) { $Python = (Get-Command python.exe -ErrorAction Stop).Source }
    & $Python -B (Join-Path $RepoRoot "tests\test_plugin.py") $pluginRoot
    if ($LASTEXITCODE -ne 0) { throw "Plugin mock regression failed." }
}

# Runtime/export scratch files are never part of the installable plugin or its
# corresponding source.  Tests may exercise asset materialization before the
# runtime lock is generated, and a developer tree may already contain ignored
# temp_doc output from real acceptance runs.
foreach ($runtimeTemp in @(
    (Join-Path $pluginRoot "temp_doc"),
    (Join-Path $sourcePluginRoot "temp_doc")
)) {
    $fullRuntimeTemp = [IO.Path]::GetFullPath($runtimeTemp)
    if (-not $fullRuntimeTemp.StartsWith($StagingRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove runtime scratch outside staging: $fullRuntimeTemp"
    }
    if (Test-Path -LiteralPath $fullRuntimeTemp) {
        Remove-Item -Recurse -Force -LiteralPath $fullRuntimeTemp
    }
}

Get-ChildItem -Recurse -Directory -LiteralPath $pluginRoot -Filter "__pycache__" |
    Sort-Object { $_.FullName.Length } -Descending |
    Remove-Item -Recurse -Force
Get-ChildItem -Recurse -File -LiteralPath $pluginRoot -Filter "*.pyc" | Remove-Item -Force
Get-ChildItem -Recurse -Directory -LiteralPath $sourceRoot -Filter "__pycache__" |
    Sort-Object { $_.FullName.Length } -Descending |
    Remove-Item -Recurse -Force
Get-ChildItem -Recurse -File -LiteralPath $sourceRoot -Filter "*.pyc" | Remove-Item -Force

# 私有依赖只在本项目新候选中聚合；源工具与策略随对应源码保留。
if ($RuntimeLayout -eq "stored") {
    $archiveTools = Join-Path $HostRepoRoot "build/tools"
    $archiveSource = Join-Path $sourceRoot "host-archive-tools"
    New-Item -ItemType Directory -Path $archiveSource | Out-Null
    foreach ($name in @("merge_runtime.py", "runtime_archive_layout.py", "audit_runtime_archives.py")) {
        $tool = Join-Path $archiveTools $name
        if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Missing archive tool: $tool" }
        Copy-Item -LiteralPath $tool -Destination $archiveSource
    }
    foreach ($name in @("COPYING", "LICENSE")) {
        Copy-Item -LiteralPath (Join-Path $HostRepoRoot $name) -Destination $archiveSource
    }
    $archivePolicy = Join-Path $RepoRoot "tools/runtime_archives_vl.json"
    Copy-Item -LiteralPath $archivePolicy -Destination (Join-Path $sourceRoot "tools/runtime_archives_vl.json")
    & $vendorPython -I -B -S -X utf8 (Join-Path $archiveTools "merge_runtime.py") $pluginRoot `
        --kind plugin --policy $archivePolicy --project-root $RepoRoot
    if ($LASTEXITCODE -ne 0) { throw "PaddleVL Stored-ZIP aggregation failed." }
    & $vendorPython -I -B -S -X utf8 (Join-Path $archiveTools "audit_runtime_archives.py") --root $pluginRoot
    if ($LASTEXITCODE -ne 0) { throw "PaddleVL Stored-ZIP source verification failed." }
}

# PaddleVL锁覆盖聚合后的全部实际文件，不借用CUDA的native-only锁语义。
$runtimeFiles = Get-ChildItem -Recurse -File -LiteralPath $pluginRoot | ForEach-Object {
    [pscustomobject]@{
        path = [IO.Path]::GetRelativePath($pluginRoot, $_.FullName).Replace("\", "/")
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower()
    }
}
@{ pluginId = $pluginName; files = @($runtimeFiles) } |
    ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 (Join-Path $pluginRoot "runtime.lock.json")

if ($DistributionLayout -eq "minimal") {
    if (-not $MinimalRoot) { $MinimalRoot = "$StagingRoot-runtime" }
    if (-not $DistributionRecord) { $DistributionRecord = Join-Path $BuildRoot "audit/minimal-vl-$([guid]::NewGuid().ToString('N')).json" }
    & $Python -I -B (Join-Path $HostRepoRoot "build/tools/prepare_minimal_distribution.py") prepare --component vl --root $StagingRoot --output $MinimalRoot --report $DistributionRecord
    if ($LASTEXITCODE -ne 0) { throw "PaddleVL 最简运行覆盖层或外置材料绑定失败。" }
    if (-not $SkipArchive) {
        $candidateSuffix = if ($LocalCandidate) { "-local-candidate" } else { "" }
        $archive = Join-Path $OutputRoot "Umi-OCR-Plugin_v${Version}${candidateSuffix}_Windows_x64_PaddleVL.tar.xz"
        & $Python -I -B (Join-Path $HostRepoRoot "build/tools/package_tar_xz.py") create --root $MinimalRoot --output $archive --integrity-output "$archive.integrity.json"
        if ($LASTEXITCODE -ne 0) { throw "最简 PaddleVL 归档失败。" }
    }
    Write-Host "最简运行目录：$MinimalRoot；完整源码/证据目录：$StagingRoot；清单：$DistributionRecord"
}
elseif (-not $SkipArchive) {
    $candidateSuffix = if ($LocalCandidate) { "-local-candidate" } else { "" }
    $archive = Join-Path $OutputRoot "Umi-OCR-Plugin_PaddleOCR-VL_v${Version}${candidateSuffix}_Windows_x64_CUDA12.6.7z"
    if (Test-Path -LiteralPath $archive) { Remove-Item -Force -LiteralPath $archive }
    $sevenZip = (Get-Command 7z.exe -ErrorAction Stop).Source
    Push-Location $StagingRoot
    try {
        & $sevenZip a -t7z -mx=7 $archive "*"
        $archiveExit = $LASTEXITCODE
    }
    finally { Pop-Location }
    if ($archiveExit -ne 0) { throw "Plugin archive creation failed." }
    & $sevenZip t $archive
    if ($LASTEXITCODE -ne 0) { throw "Plugin archive integrity check failed." }
}

Write-Host "Plugin staging ready: $StagingRoot"
