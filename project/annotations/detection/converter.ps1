param(
    [string]$FolderPath
)

$folder = "project\annotations\detection\anns\ls"

if (-not (Test-Path $folder)) {
    try {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
    }
    catch {
        Write-Host "Failed to create folder: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}

label-studio-converter import coco -i "project\annotations\detection\anns\coco\$($FolderPath).json" `
-o "project\annotations\detection\anns\ls\$($FolderPath).json" --image-root-url "/data/local-files/?d=NewPoseCustom/$($FolderPath)"
