$env:DEBUG = "true"
$env:LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "D:\"
label-studio start --host localhost --port 8081 --no-browser
