param(
    [ValidateSet('Start','Stop','Status','Pause','Resume','Once')]
    [string]$Action = 'Status',
    [string]$Python = 'E:\conda_envs\llm\python.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$syncScript = Join-Path $PSScriptRoot 'sync_project.py'
$stateDirectory = Join-Path $projectRoot 'tmp\sync'
$env:TEMP = 'E:\codex\tmp'
$env:TMP = 'E:\codex\tmp'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null

if ($Action -eq 'Start') {
    & $Python $syncScript --status
    if ($LASTEXITCODE -eq 0) {
        Write-Output 'Sync watcher is already running.'
        exit 0
    }
    # Keep this wrapper ASCII for Windows PowerShell 5.1; all state stays on E:.
    $startOptions = @{
        FilePath = $Python
        ArgumentList = @(('"' + $syncScript + '"'), '--watch')
        WorkingDirectory = $projectRoot
        WindowStyle = 'Hidden'
        PassThru = $true
        RedirectStandardOutput = (Join-Path $stateDirectory 'stdout.log')
        RedirectStandardError = (Join-Path $stateDirectory 'stderr.log')
    }
    $process = Start-Process @startOptions
    Start-Sleep -Milliseconds 700
    if ($process.HasExited) {
        Get-Content -LiteralPath (Join-Path $stateDirectory 'stderr.log') -ErrorAction SilentlyContinue
        throw 'Sync watcher exited during startup.'
    }
    Write-Output ('Sync watcher started. PID=' + $process.Id)
    & $Python $syncScript --status
    exit $LASTEXITCODE
}

$pythonAction = '--' + $Action.ToLowerInvariant()
& $Python $syncScript $pythonAction
exit $LASTEXITCODE
