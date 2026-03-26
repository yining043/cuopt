#!/usr/bin/env bash
set -euo pipefail

ROOT_REPO="/data/jieyi/cuopt"
SUBMODULE_REPO="/data/jieyi/cuopt/basin_callback"
ROOT_DEFAULT_BASE="auto"
ROOT_DEFAULT_TARGET="landscape"
SUB_DEFAULT_BASE="auto"
SUB_DEFAULT_TARGET="callback_basin"

ROOT_BASE="$ROOT_DEFAULT_BASE"
ROOT_TARGET="$ROOT_DEFAULT_TARGET"
SUB_BASE="$SUB_DEFAULT_BASE"
SUB_TARGET="$SUB_DEFAULT_TARGET"
SHOW_PATCH=0
SHOW_ALL_FILES=0

usage() {
  cat <<'EOF'
Usage:
  ./compare_changes.sh [options]

Options:
  --root-base <ref>    Base ref for root repo branch diff. Default: auto-detect
  --root-target <ref>  Target ref for root repo branch diff. Default: landscape
  --sub-base <ref>     Base ref for submodule branch diff. Default: auto-detect
  --sub-target <ref>   Target ref for submodule branch diff. Default: callback_basin
  --all-files          Include data/log/artifact files in branch file list.
  --patch              Show full patch for local uncommitted changes.
  -h, --help           Show this help.

Examples:
  ./compare_changes.sh
  ./compare_changes.sh --root-base origin/dev --root-target landscape
  ./compare_changes.sh --all-files
  ./compare_changes.sh --patch
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-base)
      ROOT_BASE="${2:-}"
      shift 2
      ;;
    --root-target)
      ROOT_TARGET="${2:-}"
      shift 2
      ;;
    --sub-base)
      SUB_BASE="${2:-}"
      shift 2
      ;;
    --sub-target)
      SUB_TARGET="${2:-}"
      shift 2
      ;;
    --patch)
      SHOW_PATCH=1
      shift
      ;;
    --all-files)
      SHOW_ALL_FILES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ensure_repo() {
  local repo="$1"
  if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
    echo "Not a git repo: $repo" >&2
    exit 1
  fi
}

ensure_ref() {
  local repo="$1"
  local ref="$2"
  if ! git -C "$repo" rev-parse --verify --quiet "$ref" >/dev/null; then
    echo "Ref not found in $repo: $ref" >&2
    exit 1
  fi
}

resolve_base_ref() {
  local repo="$1"
  local base="$2"

  if [[ "$base" != "auto" ]]; then
    ensure_ref "$repo" "$base"
    echo "$base"
    return
  fi

  local candidates=(
    "origin/branch-25.10"
    "origin/main"
    "origin/master"
  )
  local ref
  for ref in "${candidates[@]}"; do
    if git -C "$repo" rev-parse --verify --quiet "$ref" >/dev/null; then
      echo "$ref"
      return
    fi
  done

  echo "No default base ref found in $repo. Please pass --root-base/--sub-base explicitly." >&2
  exit 1
}

filter_non_code_paths() {
  rg -v \
    -e '(^|/)wandb(/|$)' \
    -e '(^|/)wandb_runtime_merged_backup_' \
    -e '(^|/)log(s)?(/|$)' \
    -e '(^|/)out(/|$)' \
    -e '(^|/)perturb_k1_collect(/|$)' \
    -e '(^|/)s3_data_cache(/|$)' \
    -e '(^|/)basin_datasets(/|$)' \
    -e '(^|/)basin_stat(/|$)' \
    -e '(^|/)plott_CaR(/|$)' \
    -e '(^|/)plot(s)?(_[^/]*)?(/|$)' \
    -e '\.log$' \
    -e '\.jsonl$' \
    -e '\.xlsx$' \
    -e '\.csv$' \
    -e '\.gif$' \
    -e '\.pkl$' \
    -e '\.pickle$' \
    -e '\.partial$'
}

print_branch_file_diff() {
  local repo="$1"
  local base="$2"
  local target="$3"

  if [[ $SHOW_ALL_FILES -eq 1 ]]; then
    git -C "$repo" diff --name-status "$base...$target"
    return
  fi

  git -C "$repo" diff --name-status "$base...$target" | filter_non_code_paths || true
}

print_local_changed_files() {
  local repo="$1"

  if [[ $SHOW_ALL_FILES -eq 1 ]]; then
    {
      git -C "$repo" diff --name-only
      git -C "$repo" diff --cached --name-only
      git -C "$repo" ls-files --others --exclude-standard
    } | awk 'NF' | sort -u
    return
  fi

  {
    git -C "$repo" diff --name-only
    git -C "$repo" diff --cached --name-only
    git -C "$repo" ls-files --others --exclude-standard
  } | awk 'NF' | sort -u | filter_non_code_paths || true
}

print_repo_report() {
  local repo="$1"
  local label="$2"
  local base="$3"
  local target="$4"
  local resolved_base

  ensure_repo "$repo"
  git -C "$repo" fetch origin >/dev/null 2>&1 || true
  resolved_base="$(resolve_base_ref "$repo" "$base")"
  ensure_ref "$repo" "$target"

  echo "============================================================"
  echo "[$label] $repo"
  echo "Branch diff: $resolved_base...$target"
  print_branch_file_diff "$repo" "$resolved_base" "$target"
  echo
  echo "Commits in $target not in $resolved_base:"
  git -C "$repo" log --oneline "$resolved_base..$target" || true
  echo
  echo "Local uncommitted status:"
  git -C "$repo" status --short
  echo
  echo "Local changed files (unstaged + staged):"
  print_local_changed_files "$repo"
  echo
  echo "Local change summary:"
  git -C "$repo" diff --stat
  git -C "$repo" diff --cached --stat

  if [[ $SHOW_PATCH -eq 1 ]]; then
    echo
    echo "Local patch (unstaged):"
    git -C "$repo" diff
    echo
    echo "Local patch (staged):"
    git -C "$repo" diff --cached
  fi

  echo
}

print_repo_report "$SUBMODULE_REPO" "submodule" "$SUB_BASE" "$SUB_TARGET"
print_repo_report "$ROOT_REPO" "root" "$ROOT_BASE" "$ROOT_TARGET"
