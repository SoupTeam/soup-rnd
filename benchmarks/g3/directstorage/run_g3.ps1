# G3 Windows DirectStorage series (PROTOCOL.txt). Run from this folder, on mains power:
#   powershell -ExecutionPolicy Bypass -File .\run_g3.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_g3.ps1 -Blocks 3 -StartBlock 5   # top-up only
# Installs nothing. Downloads the official DirectStorage NuGet package once and checks its SHA256.
param([int]$Blocks = 5, [int]$StartBlock = 0)
# Continue: Windows PowerShell 5.1 turns a native program's stderr into errors; critical cmdlets
# below pass -ErrorAction Stop themselves.
$ErrorActionPreference = 'Continue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Set-Location -LiteralPath $PSScriptRoot
$PackageSha = '8ab6c1082537565388b79050d94fdd7f5f437cd0f1d909e7b19f786042b97177'
$PackageUrl = 'https://www.nuget.org/api/v2/package/Microsoft.Direct3D.DirectStorage/1.3.0'
$Work = Join-Path $PSScriptRoot 'work'
$Out = Join-Path $PSScriptRoot ('results_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Force -Path $Work -ErrorAction Stop | Out-Null
New-Item -ItemType Directory -Path $Out, (Join-Path $Out 'runs') -ErrorAction Stop | Out-Null
$Log = Join-Path $Out 'run_g3.log'
function Say($text) { $line = (Get-Date -Format 'HH:mm:ss') + ' ' + $text; Write-Host $line; Add-Content -LiteralPath $Log -Value $line }

if (-not (Test-Path 'ds_probe.exe')) { Say 'ds_probe.exe is missing next to this script'; exit 2 }

# 1. DirectStorage runtime from the official package, checked by SHA256.
if (-not (Test-Path 'dstorage.dll') -or -not (Test-Path 'dstoragecore.dll')) {
    $pkg = Join-Path $Work 'directstorage.1.3.0.zip'
    Say "Downloading $PackageUrl"
    Invoke-WebRequest -Uri $PackageUrl -OutFile $pkg -UseBasicParsing -ErrorAction Stop
    $sha = (Get-FileHash -Algorithm SHA256 -LiteralPath $pkg).Hash.ToLower()
    if ($sha -ne $PackageSha) { Say "Package SHA256 mismatch: $sha"; exit 3 }
    $unz = Join-Path $Work 'pkg'
    Expand-Archive -LiteralPath $pkg -DestinationPath $unz -Force -ErrorAction Stop
    Copy-Item (Join-Path $unz 'native\bin\x64\dstorage.dll'), (Join-Path $unz 'native\bin\x64\dstoragecore.dll') -Destination $PSScriptRoot -ErrorAction Stop
}

# 2. Machine record (no user or computer names, no paths beyond the drive letter).
$drive = (Get-Item -LiteralPath $Work).PSDrive.Name
$part = Get-Partition -DriveLetter $drive -ErrorAction SilentlyContinue
$disk = if ($part) { Get-PhysicalDisk | Where-Object { $_.DeviceId -eq [string]$part.DiskNumber } } else { $null }
$os = Get-CimInstance Win32_OperatingSystem
$power = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue
$machine = [ordered]@{
    schema = 'g3-ds-machine-v1'
    os = $os.Caption; os_build = $os.BuildNumber; os_version = $os.Version
    cpu = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
    ram_bytes = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory
    gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object { @{ name = $_.Name; driver = $_.DriverVersion } })
    fixture_drive = $drive
    disk = if ($disk) { @{ model = $disk.FriendlyName; bus = [string]$disk.BusType; media = [string]$disk.MediaType; size = $disk.Size } } else { $null }
    file_system = (Get-Volume -DriveLetter $drive).FileSystem
    on_battery_now = if ($power) { -not $power[0].PowerOnline } else { $null }
    probe_sha256 = (Get-FileHash -Algorithm SHA256 'ds_probe.exe').Hash.ToLower()
    dstorage_sha256 = (Get-FileHash -Algorithm SHA256 'dstorage.dll').Hash.ToLower()
    dstoragecore_sha256 = (Get-FileHash -Algorithm SHA256 'dstoragecore.dll').Hash.ToLower()
    script_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLower()
    blocks = $Blocks; start_block = $StartBlock
}
$machine | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Out 'machine.json') -Encoding UTF8
function Smi($name) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        & $smi.Source --query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv 2>&1 |
            Set-Content -LiteralPath (Join-Path $Out "runs\$name.smi.csv") -Encoding UTF8
        & $smi.Source --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 |
            Add-Content -LiteralPath (Join-Path $Out "runs\$name.smi.csv") -Encoding UTF8
    }
}

# 3. Selftest: both arms end to end on 64 MiB.
Say 'Selftest'
& .\ds_probe.exe selftest $Work 2>&1 | Set-Content -LiteralPath (Join-Path $Out 'selftest.jsonl') -Encoding UTF8
Say ('Selftest exit ' + $LASTEXITCODE)

# 4. Fixture, once.
$fixture = Join-Path $Work 'g3_fixture_1gib.bin'
if (-not (Test-Path -LiteralPath $fixture)) {
    Say 'Writing the 1 GiB fixture'
    & .\ds_probe.exe prepare $fixture 2>&1 | Set-Content -LiteralPath (Join-Path $Out 'prepare.json') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { Say 'Fixture preparation failed'; exit 4 }
}

# 5. Blocks A B B A, one fresh process per run. No retry of any run.
$order = @('A', 'B', 'B', 'A')
$series = @()
for ($b = $StartBlock; $b -lt $StartBlock + $Blocks; $b++) {
    for ($p = 0; $p -lt 4; $p++) {
        $arm = $order[$p]
        $name = 'b{0:D2}_{1}_{2}' -f $b, $p, $arm
        Smi "$name.before"
        $started = Get-Date
        & .\ds_probe.exe run $arm $fixture (Join-Path $Out "runs\$name.json") 2> (Join-Path $Out "runs\$name.stderr.txt")
        $code = $LASTEXITCODE
        Smi "$name.after"
        $series += [ordered]@{ block = $b; position = $p; arm = $arm; exit = $code; started = $started.ToString('o'); seconds = ((Get-Date) - $started).TotalSeconds }
        Say ("$name exit $code")
    }
}
$series | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Out 'series.json') -Encoding UTF8

# 6. One zip to send back.
$zip = $Out + '.zip'
Compress-Archive -Path (Join-Path $Out '*') -DestinationPath $zip
Say "Done. Send this file: $zip"
