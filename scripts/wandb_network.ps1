param([ValidateSet('Start','Status','Stop')][string]$Action='Status')
$ErrorActionPreference='Stop'
$env:TEMP='E:\codex\tmp'
$env:TMP=$env:TEMP
$env:PYTHONDONTWRITEBYTECODE='1'
$root=Split-Path -Parent $PSScriptRoot
$python='E:\conda_envs\llm\python.exe'
$pythonw='E:\conda_envs\llm\pythonw.exe'
$script=Join-Path $PSScriptRoot 'wandb_network.py'
if($Action -eq 'Start'){
    $statusPath=Join-Path $root 'tmp\wandb_network\status.json'
    if(Test-Path -LiteralPath $statusPath){
        $status=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        $age=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()-$status.updated_unix_seconds
        if($status.status -in @('starting','connecting','running','reconnecting') -and $age -lt 15){
            & $python $script status
            exit $LASTEXITCODE
        }
    }
    $config=Get-Content -LiteralPath (Join-Path $root 'configs\remote\wandb_network.json') -Raw | ConvertFrom-Json
    $logDirectory=Join-Path $root $config.log_dir
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    Start-Process -FilePath $pythonw -ArgumentList @($script,'run') -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDirectory 'watcher.stdout.log') `
        -RedirectStandardError (Join-Path $logDirectory 'watcher.stderr.log') | Out-Null
    Write-Output 'W&B network helper started in the background.'
}elseif($Action -eq 'Stop'){
    & $python $script stop
}else{
    & $python $script status
}
