param(
    [Parameter(Mandatory = $true)][string[]]$DocumentPaths,
    [Parameter(Mandatory = $true)][string]$PdfDirectory
)

$ErrorActionPreference = 'Stop'
$pdfRoot = [System.IO.Path]::GetFullPath($PdfDirectory)
[System.IO.Directory]::CreateDirectory($pdfRoot) | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$word.AutomationSecurity = 3
try {
    foreach ($path in $DocumentPaths) {
        $fullPath = [System.IO.Path]::GetFullPath($path)
        $doc = $word.Documents.Open($fullPath, $false, $false)
        try {
            foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
            $doc.Fields.Update() | Out-Null
            $doc.Save()
            $pdfName = [System.IO.Path]::GetFileNameWithoutExtension($fullPath) + '.pdf'
            $pdfPath = [System.IO.Path]::Combine($pdfRoot, $pdfName)
            $doc.ExportAsFixedFormat($pdfPath, 17)
        }
        finally {
            $doc.Close($false)
        }
    }
}
finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) | Out-Null
}
