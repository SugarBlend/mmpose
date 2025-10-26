#param(
#    [Parameter(Mandatory=$true)]
#    [string]$ZrokUrl
#)
#
#if ($ZrokUrl -match '^https?://') {
#    $ZrokUrl = $ZrokUrl -replace '^https?://', ''
#}
#
#$ZrokUrl = $ZrokUrl -replace '/+$', ''
#
#$env:ALLOWED_HOSTS = "localhost,127.0.0.1,$ZrokUrl"
#$env:CSRF_TRUSTED_ORIGINS = "https://$ZrokUrl"
#$env:LABEL_STUDIO_HOST = "https://$ZrokUrl"
#$env:DEBUG = "true"
#$env:LOCAL_FILES_SERVING_ENABLED = "true"
#$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "$pwd\project\dataset"
#label-studio

$env:DEBUG = "true"
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "D:\"
label-studio start --host localhost --port 8081 --no-browser