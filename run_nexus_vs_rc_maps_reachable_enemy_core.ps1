[CmdletBinding()]
param(
    # Directory containing the .map26 files.  Relative paths are resolved from
    # the project root (the directory containing this script).
    [string]$MapsDir = "maps",

    # Each match needs its own replay so a later game cannot overwrite it.
    [string]$ReplayDir = "replays/nexus-vs-rc",

    [int]$Seed = 1
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

# Maps to run, in match order.  Edit this list to add, remove, or reorder maps.
# Keep the .map26 extension so a missing map is reported clearly before a match starts.
[string[]]$MapFiles = @(
    "building_blocks.map26", # loser
    "butterfly.map26", # loser
    "chemistry_class.map26",
    "clash_arena.map26", # loser
    "default_large1.map26",
    "default_small1.map26", # loser
    "face.map26", # loser
    "first_sound.map26",
    "flappy_bird.map26", # loser
    "hourglass.map26", # loser
    "landscape.map26", # loser
    "pixel_forest.map26", # loser
    "pong.map26", # loser
    "rush_bait.map26", # loser
    "separated.map26",
    "socket.map26", # loser
    "spikes.map26", # loser
    "starry_night.map26", # loser
    "the_great_divide.map26", # loser
    "thread_of_connection.map26", # loser
    "vase.map26", # loser
    "window_shopping.map26" # loser
)

function Resolve-ProjectPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Find-Cambc {
    $installed = Get-Command cambc -ErrorAction SilentlyContinue
    if ($null -ne $installed) {
        if ($installed.Path) {
            return $installed.Path
        }
        return $installed.Source
    }

    # This is the usual layout for this project when cambc is installed in a
    # local virtual environment instead of globally.
    foreach ($candidate in @(
        (Join-Path $projectRoot ".venv\Scripts\cambc.exe"),
        (Join-Path $projectRoot ".venv-3.13\Scripts\cambc.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    throw "cambc was not found. Install it or activate the virtual environment first."
}

$mapsPath = Resolve-ProjectPath $MapsDir
$replaysPath = Resolve-ProjectPath $ReplayDir

if (-not (Test-Path -LiteralPath $mapsPath -PathType Container)) {
    throw "Maps directory does not exist: $mapsPath"
}

$maps = @(
    foreach ($mapFile in $MapFiles) {
        $mapPath = Join-Path $mapsPath $mapFile
        if (-not (Test-Path -LiteralPath $mapPath -PathType Leaf)) {
            throw "Map from `$MapFiles was not found: $mapPath"
        }

        Get-Item -LiteralPath $mapPath
    }
)

if ($maps.Count -eq 0) {
    throw "`$MapFiles is empty. Add at least one .map26 file name."
}

$cambc = Find-Cambc
New-Item -ItemType Directory -Path $replaysPath -Force | Out-Null

$rcLosses = [System.Collections.Generic.List[object]]::new()
$incompleteMaps = [System.Collections.Generic.List[string]]::new()

Push-Location $projectRoot
try {
    for ($index = 0; $index -lt $maps.Count; $index++) {
        $map = $maps[$index]
        $replayName = "{0}_seed{1}.replay26" -f $map.BaseName, $Seed
        $replayPath = Join-Path $replaysPath $replayName

        Write-Host ("`n[{0}/{1}] nexus vs rc on {2}" -f ($index + 1), $maps.Count, $map.Name)

        # Match the production per-turn CPU limit instead of running without TLE.
        $matchOutput = & $cambc run nexus rc $map.FullName --replay $replayPath --seed $Seed --tle 2 2>&1
        $exitCode = $LASTEXITCODE
        $matchOutput | ForEach-Object { Write-Host $_ }

        $summary = $matchOutput | Out-String
        $winnerMatch = [regex]::Match($summary, "(?im)^\s*Winner:\s*(?<winner>[^\s(]+)")

        if ($exitCode -ne 0 -or -not $winnerMatch.Success) {
            $incompleteMaps.Add($map.Name)
            Write-Warning "Could not determine the winner for $($map.Name). Replay (if created): $replayPath"
            continue
        }

        $winner = $winnerMatch.Groups["winner"].Value
        if ($winner -ieq "rc") {
            Write-Host "RC won."
            continue
        }

        # A completed game always has one winner.  Anything other than rc is
        # therefore an unsuccessful game for rc; print an absolute replay path.
        $absoluteReplayPath = [System.IO.Path]::GetFullPath($replayPath)
        $rcLosses.Add([PSCustomObject]@{
            Map    = $map.Name
            Replay = $absoluteReplayPath
        })
        Write-Host "RC lost. Replay: $absoluteReplayPath" -ForegroundColor Red
    }
}
finally {
    Pop-Location
}

Write-Host "`n=== Сводка ==="
if ($rcLosses.Count -eq 0) {
    Write-Host "RC не проиграл ни на одной завершённой карте."
}
else {
    Write-Host "RC проиграл на следующих картах:"
    $rcLosses | ForEach-Object {
        Write-Host ("- {0}: {1}" -f $_.Map, $_.Replay) -ForegroundColor Red
    }
}

if ($incompleteMaps.Count -gt 0) {
    Write-Warning ("No result was determined for: " + ($incompleteMaps -join ", "))
    exit 1
}
