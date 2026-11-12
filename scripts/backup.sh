#!/usr/bin/env bash
# Back up the database and the media and artwork volumes of the running compose stack into ./backups:
#   backups/YYYY-MM-DD.sql.gz            pg_dump of the autocare database (with DROP ... IF EXISTS, see README restore)
#   backups/YYYY-MM-DD-media.tar.gz      contents of the volume mounted at /data/media in the api container
#   backups/YYYY-MM-DD-visuals.tar.gz    contents of the volume mounted at /data/visuals (vehicle artwork)
# Keeps the newest 14 of each. Run from anywhere: bash scripts/backup.sh
set -euo pipefail

KEEP=14
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p backups

day="$(date +%F)"
sql="backups/$day.sql.gz"
trap 'rm -f backups/"$day"*.partial' EXIT

docker compose exec -T db pg_dump -U autocare --clean --if-exists autocare | gzip > "$sql.partial"
mv "$sql.partial" "$sql"
echo "database -> $sql"

api="$(docker compose ps -aq api)"
if [ -z "$api" ]; then
    echo "no api container found; run docker compose up -d first" >&2
    exit 1
fi

# backup_volume <mount> <suffix>: archive the volume mounted at <mount> in the api container as backups/<day>-<suffix>.tar.gz.
# Ask Docker which volume backs the mount instead of guessing the compose project name.
backup_volume() {
    local mount="$1" suffix="$2" volume file="backups/$day-$2.tar.gz"
    volume="$(docker inspect -f "{{range .Mounts}}{{if eq .Destination \"$mount\"}}{{.Name}}{{end}}{{end}}" "$api")"
    if [ -z "$volume" ]; then
        echo "the api container has no volume at $mount" >&2
        exit 1
    fi
    docker run --rm -v "$volume:/data:ro" postgres:16-alpine tar -czf - -C /data . > "$file.partial"
    mv "$file.partial" "$file"
    echo "$suffix ($volume) -> $file"
}
backup_volume /data/media media
backup_volume /data/visuals visuals

prune() {
    local files=("$@")
    local excess=$(( ${#files[@]} - KEEP ))
    if [ "$excess" -gt 0 ]; then
        rm -f -- "${files[@]:0:excess}"
    fi
}
shopt -s nullglob
prune backups/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].sql.gz
prune backups/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-media.tar.gz
prune backups/[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-visuals.tar.gz

echo
echo "Crontab example (daily at 03:15; add with crontab -e):"
echo "15 3 * * * cd $ROOT && bash scripts/backup.sh >> backups/backup.log 2>&1"
