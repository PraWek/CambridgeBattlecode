[CmdletBinding()]
param(
    # Directory containing the .map26 files.  Relative paths are resolved from
    # the project root (the directory containing this script).
    [string]$MapsDir = "maps",

    # Each match needs its own replay so a later game cannot overwrite it.
    [string]$ReplayDir = "replays/nexus-vs-rc",

    [int]$Seed = 1,

    # Maximum number of matches that may run at once.
    [ValidateRange(1, 64)]
    [int]$ThrottleLimit = 4,

    # CSV containing completed match results. Relative paths use the project root.
    [string]$ResultsCsv = "replays/nexus-vs-rc/nexus-vs-rc-results.csv"
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

# Maps to run, in match order.  Edit this list to add, remove, or reorder maps.
# Keep the .map26 extension so a missing map is reported clearly before a match starts.
[string[]]$MapFiles = @(
    "chemistry_class.map26",
    "default_large1.map26",
    "default_small1.map26",
    "face.map26",
    "first_sound.map26"
    "flappy_bird.map26",
    "landscape.map26",
    "pixel_forest.map26",
    "socket.map26",
    "spikes.map26",
    "starry_night.map26",
    "the_great_divide.map26",
    "thread_of_connection.map26",
    "vase.map26",
    "window_shopping.map26",
    # Keep this sentinel: it lets the preceding map retain its comma when it
    # becomes the final active entry after maps below it are commented out.
    $null
) | Where-Object { $null -ne $_ }

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
$resultsCsvPath = Resolve-ProjectPath $ResultsCsv

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
New-Item -ItemType Directory -Path (Split-Path -Parent $resultsCsvPath) -Force | Out-Null

$matchJobScript = {
    param(
        [int]$Index,
        [int]$Total,
        [string]$ProjectRoot,
        [string]$Cambc,
        [string]$MapName,
        [string]$MapPath,
        [string]$ReplayName,
        [string]$ReplayPath,
        [int]$Seed
    )

    try {
        # Bot names used by cambc are resolved relative to the project root.
        Set-Location -LiteralPath $ProjectRoot

        # Match the production per-turn CPU limit instead of running without TLE.
        $matchOutput = @(& $Cambc run nexus rc $MapPath --replay $ReplayPath --seed $Seed --tle 2 2>&1)
        $exitCode = $LASTEXITCODE
        $summary = $matchOutput | Out-String
        $winnerMatch = [regex]::Match($summary, "(?im)^\s*Winner:\s*(?<winner>[^\s(]+)")
        $winner = if ($winnerMatch.Success) { $winnerMatch.Groups["winner"].Value } else { $null }

        [PSCustomObject]@{
            Index      = $Index
            Total      = $Total
            Map        = $MapName
            Replay     = $ReplayName
            ReplayPath = $ReplayPath
            ExitCode   = $exitCode
            Winner     = $winner
            Completed  = ($exitCode -eq 0 -and $winnerMatch.Success)
            Output     = $summary
            Error      = $null
        }
    }
    catch {
        [PSCustomObject]@{
            Index      = $Index
            Total      = $Total
            Map        = $MapName
            Replay     = $ReplayName
            ReplayPath = $ReplayPath
            ExitCode   = $null
            Winner     = $null
            Completed  = $false
            Output     = ""
            Error      = $_.Exception.Message
        }
    }
}

function Receive-MatchJobResult {
    param(
        [Parameter(Mandatory)]
        [System.Management.Automation.Job]$Job
    )

    try {
        $jobResults = @(Receive-Job -Job $Job -ErrorAction Stop)
        if ($jobResults.Count -ne 1) {
            throw "Expected one result from match job '$($Job.Name)', got $($jobResults.Count)."
        }

        return $jobResults[0]
    }
    catch {
        return [PSCustomObject]@{
            Index      = $Job.MatchIndex
            Total      = $Job.MatchTotal
            Map        = $Job.MapName
            Replay     = $Job.ReplayName
            ReplayPath = $Job.ReplayPath
            ExitCode   = $null
            Winner     = $null
            Completed  = $false
            Output     = ""
            Error      = $_.Exception.Message
        }
    }
    finally {
        Remove-Job -Job $Job -Force -ErrorAction SilentlyContinue
    }
}

function Complete-MatchJob {
    param(
        [Parameter(Mandatory)]
        [System.Management.Automation.Job]$Job,

        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[object]]$CompletedMatches,

        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[object]]$RcLosses,

        [Parameter(Mandatory)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[string]]$IncompleteMaps
    )

    $matchResult = Receive-MatchJobResult $Job
    Write-Host ("`n[{0}/{1}] nexus vs rc on {2}" -f $matchResult.Index, $matchResult.Total, $matchResult.Map)
    if ($matchResult.Output) {
        Write-Host $matchResult.Output.TrimEnd()
    }

    if (-not $matchResult.Completed) {
        $IncompleteMaps.Add($matchResult.Map)
        $errorDetails = if ($matchResult.Error) { ": $($matchResult.Error)" } else { "" }
        Write-Warning "Could not determine the winner for $($matchResult.Map). Replay (if created): $($matchResult.ReplayPath)$errorDetails"
        return
    }

    $result = if ($matchResult.Winner -ieq "rc") { "RC won" } else { "RC lost" }
    $completedMatch = [PSCustomObject]@{
        Index  = $matchResult.Index
        Map    = $matchResult.Map
        Replay = $matchResult.Replay
        Winner = $matchResult.Winner
        Result = $result
    }
    $CompletedMatches.Add($completedMatch)

    if ($matchResult.Winner -ieq "rc") {
        Write-Host "RC won."
        return
    }

    $RcLosses.Add([PSCustomObject]@{
        Map    = $matchResult.Map
        Replay = $matchResult.ReplayPath
    })
    Write-Host "RC lost. Replay: $($matchResult.ReplayPath)" -ForegroundColor Red
}

$rcLosses = [System.Collections.Generic.List[object]]::new()
$completedMatches = [System.Collections.Generic.List[object]]::new()
$incompleteMaps = [System.Collections.Generic.List[string]]::new()
$runningJobs = @()

try {
    $nextIndex = 0
    while ($nextIndex -lt $maps.Count -or $runningJobs.Count -gt 0) {
        while ($nextIndex -lt $maps.Count -and $runningJobs.Count -lt $ThrottleLimit) {
            $map = $maps[$nextIndex]
            $matchIndex = $nextIndex + 1
            $replayName = "{0}_seed{1}.replay26" -f $map.BaseName, $Seed
            $replayPath = Join-Path $replaysPath $replayName

            $job = Start-Job -Name ("nexus-vs-rc-{0}" -f $matchIndex) -ScriptBlock $matchJobScript -ArgumentList @(
                $matchIndex,
                $maps.Count,
                $projectRoot,
                $cambc,
                $map.Name,
                $map.FullName,
                $replayName,
                $replayPath,
                $Seed
            )
            $job | Add-Member -NotePropertyName MatchIndex -NotePropertyValue $matchIndex
            $job | Add-Member -NotePropertyName MatchTotal -NotePropertyValue $maps.Count
            $job | Add-Member -NotePropertyName MapName -NotePropertyValue $map.Name
            $job | Add-Member -NotePropertyName ReplayName -NotePropertyValue $replayName
            $job | Add-Member -NotePropertyName ReplayPath -NotePropertyValue $replayPath
            $runningJobs += $job
            Write-Host ("Started [{0}/{1}] nexus vs rc on {2}" -f $matchIndex, $maps.Count, $map.Name)
            $nextIndex++
        }

        if ($runningJobs.Count -gt 0) {
            $finishedJob = Wait-Job -Job $runningJobs -Any
            Complete-MatchJob $finishedJob $completedMatches $rcLosses $incompleteMaps
            $runningJobs = @($runningJobs | Where-Object { $_.Id -ne $finishedJob.Id })
        }
    }
}
finally {
    foreach ($job in $runningJobs) {
        if ($job.State -eq "Running") {
            Stop-Job -Job $job -ErrorAction SilentlyContinue
        }
        Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
    }
}

$csvRows = @(
    $completedMatches |
        Sort-Object Index |
        Select-Object Map, Replay, Winner, Result
)
if ($csvRows.Count -gt 0) {
    $csvRows | Export-Csv -LiteralPath $resultsCsvPath -NoTypeInformation -Encoding UTF8
}
else {
    # Still leave a valid CSV with its schema when every match is incomplete.
    '"Map","Replay","Winner","Result"' | Set-Content -LiteralPath $resultsCsvPath -Encoding UTF8
}
Write-Host ("`nРезультаты {0} завершённых матчей сохранены в: {1}" -f $completedMatches.Count, $resultsCsvPath)

Write-Host "`n=== Сводка ==="
if ($rcLosses.Count -eq 0) {
    Write-Host "RC не проиграл ни на одной завершённой карте."
}
else {
    Write-Host "Команды для просмотра проигранных матчей:"
    $rcLosses | Sort-Object Map | ForEach-Object {
        Write-Host ("cambc watch {0}" -f $_.Replay) -ForegroundColor Red
    }
}

if ($incompleteMaps.Count -gt 0) {
    Write-Warning ("No result was determined for: " + ($incompleteMaps -join ", "))
    exit 1
}
