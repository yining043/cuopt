#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="/data/jieyi/cuopt/"
DEST_DEFAULT="jieyi@10.96.187.19:/home/jieyi/cuopt_mit/"
DEST="${DEST:-$DEST_DEFAULT}"

# Default behavior is preview-only for safety.
DRY_RUN=1
DELETE_MODE=0

print_usage() {
  cat <<'EOF'
Usage:
  ./sync_to_mit.sh [--apply] [--delete] [--dest user@host:/path/]

Options:
  --apply      Perform real sync (default is dry-run preview).
  --delete     Delete remote files that don't exist locally.
  --dest       Override destination path.
  -h, --help   Show this help.

Examples:
  ./sync_to_mit.sh
  ./sync_to_mit.sh --apply
  ./sync_to_mit.sh --apply --delete
  ./sync_to_mit.sh --apply --dest jieyi@10.96.187.19:/home/jieyi/cuopt_mit/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      DRY_RUN=0
      shift
      ;;
    --delete)
      DELETE_MODE=1
      shift
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      print_usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source directory not found: $SRC_DIR" >&2
  exit 1
fi

RSYNC_ARGS=(
  -azvh
  --info=progress2,stats
  --filter=':- .gitignore'
  --exclude=".git/"
  --exclude=".git/**"
  --exclude=".venv/"
  --exclude="venv/"
  --exclude="env/"
  --exclude=".mypy_cache/"
  --exclude=".ruff_cache/"
  --exclude=".DS_Store"
  --exclude="wandb*/"
  --exclude="**/wandb*/"
  --exclude="log/"
  --exclude="logs/"
  --exclude="out/"
  --exclude="plott_CaR/"
  --exclude="basin_callback/"
)

if [[ $DRY_RUN -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

if [[ $DELETE_MODE -eq 1 ]]; then
  RSYNC_ARGS+=(--delete)
fi

echo "Source: $SRC_DIR"
echo "Dest:   $DEST"
echo "Mode:   $([[ $DRY_RUN -eq 1 ]] && echo 'dry-run' || echo 'apply')"
echo "Delete: $([[ $DELETE_MODE -eq 1 ]] && echo 'enabled' || echo 'disabled')"
echo

rsync "${RSYNC_ARGS[@]}" "$SRC_DIR" "$DEST"
