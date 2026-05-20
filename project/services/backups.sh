#!/bin/bash
set -a
. "$(dirname "$0")/.env"
set +a

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/tmp/ls_backup_$DATE"
ALIAS="local"
BUCKET="$ALIAS/backups/label-studio"

mkdir -p "$BACKUP_DIR"

cp /ssd/labelstudio/label_studio.sqlite3 "$BACKUP_DIR/label_studio.sqlite3"

cp -r /ssd/labelstudio/media "$BACKUP_DIR/"

tar -czf "/tmp/ls_backup_$DATE.tar.gz" -C /tmp "ls_backup_$DATE"

mc alias set "$ALIAS" http://localhost:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
mc cp "/tmp/ls_backup_$DATE.tar.gz" "$BUCKET/"

mc find "$BUCKET" --older-than 7d | while read f; do mc rm "$f"; done

rm -rf "$BACKUP_DIR" "/tmp/ls_backup_$DATE.tar.gz"
