param(
    [Parameter(Mandatory = $true)][string]$DocumentPath,
    [Parameter(Mandatory = $true)][string]$EquationManifest
)

$ErrorActionPreference = 'Stop'
$officeCli = 'C:\Users\canna\AppData\Local\OfficeCLI\officecli.exe'
$resolvedDocument = (Resolve-Path -LiteralPath $DocumentPath).Path
$resolvedManifest = (Resolve-Path -LiteralPath $EquationManifest).Path

& $officeCli open $resolvedDocument | Out-Null
try {
    $items = Get-Content -LiteralPath $resolvedManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($item in $items) {
        $selector = 'p:contains("' + $item.token + '")'
        $queryText = & $officeCli query $resolvedDocument $selector --json
        if ($LASTEXITCODE -ne 0) { throw "OfficeCLI query failed for $($item.token)" }
        $query = $queryText | ConvertFrom-Json
        if (-not $query.success -or $query.data.matches -ne 1) {
            throw "Expected one placeholder for $($item.token), found $($query.data.matches)"
        }
        $target = $query.data.results[0].path
        $formulaProperty = 'formula=' + [string]$item.formula
        & $officeCli add $resolvedDocument $target --type equation --prop mode=inline --prop $formulaProperty | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "OfficeCLI equation insertion failed for $($item.token)" }
        & $officeCli set $resolvedDocument $target --find $item.token --replace '' | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "OfficeCLI placeholder removal failed for $($item.token)" }
    }
    & $officeCli set $resolvedDocument /settings --prop updateFields=true | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'OfficeCLI updateFields setting failed' }
}
finally {
    & $officeCli close $resolvedDocument | Out-Null
}

& $officeCli validate $resolvedDocument
if ($LASTEXITCODE -ne 0) { throw 'OfficeCLI validation failed' }
