#!/usr/bin/env bash
set -euo pipefail

ROOT_REPO="/data/jieyi/cuopt"
SUBMODULE_REPO="/data/jieyi/cuopt/basin_callback"
ROOT_BRANCH="landscape"
SUBMODULE_BRANCH="callback_basin"

usage() {
  cat <<'EOF'
Usage:
  ./commit_code_only.sh --root-msg "message for cuopt" --sub-msg "message for basin_callback"

Options:
  --root-msg   Commit message for /data/jieyi/cuopt on landscape branch.
  --sub-msg    Commit message for /data/jieyi/cuopt/basin_callback on callback_basin branch.
  --auto-rebase  If behind/diverged, run pull --rebase before commit.
  -h, --help   Show this help.
EOF
}

ROOT_MSG=""
SUB_MSG=""
AUTO_REBASE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-msg)
      ROOT_MSG="${2:-}"
      shift 2
      ;;
    --root-msg=*)
      ROOT_MSG="${1#*=}"
      shift
      ;;
    --sub-msg)
      SUB_MSG="${2:-}"
      shift 2
      ;;
    --sub-msg=*)
      SUB_MSG="${1#*=}"
      shift
      ;;
    --auto-rebase)
      AUTO_REBASE=1
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

if [[ -z "$ROOT_MSG" || -z "$SUB_MSG" ]]; then
  echo "Both --root-msg and --sub-msg are required." >&2
  usage
  exit 1
fi

ensure_repo() {
  local repo="$1"
  if [[ ! -d "$repo/.git" && ! -f "$repo/.git" ]]; then
    echo "Not a git repo: $repo" >&2
    exit 1
  fi
}

checkout_branch() {
  local repo="$1"
  local branch="$2"

  git -C "$repo" fetch origin
  if git -C "$repo" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$repo" checkout "$branch"
    return
  fi

  if git -C "$repo" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    git -C "$repo" checkout -b "$branch" --track "origin/$branch"
    return
  fi

  echo "Branch not found locally/remotely: $branch (repo: $repo)" >&2
  exit 1
}

maybe_rebase_branch() {
  local repo="$1"
  local branch="$2"
  local label="$3"
  local ahead behind

  if ! git -C "$repo" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    return 0
  fi

  read -r behind ahead < <(
    git -C "$repo" rev-list --left-right --count "origin/$branch...$branch"
  )
  echo "==> [$label] sync status: ahead=$ahead, behind=$behind"

  if [[ "$behind" -eq 0 ]]; then
    return 0
  fi

  if [[ $AUTO_REBASE -eq 1 ]]; then
    echo "==> [$label] pulling with rebase from origin/$branch"
    git -C "$repo" pull --rebase origin "$branch"
    return 0
  fi

  echo "==> [$label] warning: branch is behind/diverged with origin/$branch." >&2
  echo "==> [$label] hint: rerun with --auto-rebase, or run:" >&2
  echo "    git -C \"$repo\" pull --rebase origin \"$branch\"" >&2
}

is_code_path() {
  local path="$1"
  local name="${path##*/}"

  case "$path" in
    wandb/*|wandb_runtime_merged_backup_*|log/*|logs/*|out/*|\
    perturb_k1_collect/*|s3_data_cache/*|basin_datasets/*|basin_stat/*|plot/*|plots_*|plott_CaR/*)
      return 1
      ;;
  esac

  case "$name" in
    *.pkl|*.pickle)
      return 1
      ;;
  esac

  case "$name" in
    CMakeLists.txt|Makefile|Dockerfile|setup.py|setup.cfg|pyproject.toml|\
    .gitignore|.gitattributes|.gitmodules|.pre-commit-config.yaml)
      return 0
      ;;
  esac

  case "$name" in
    requirements*.txt|conda*.yml|conda*.yaml)
      return 0
      ;;
  esac

  case "$name" in
    *.py|*.pyi|*.ipynb|*.c|*.cc|*.cpp|*.cu|*.cuh|*.h|*.hpp|\
    *.sh|*.bash|*.zsh|*.ps1|*.js|*.jsx|*.ts|*.tsx|\
    *.md|*.rst|*.yaml|*.yml|*.toml|*.json|*.ini|*.cfg)
      return 0
      ;;
  esac

  return 1
}

stage_code_changes() {
  local repo="$1"
  local -a candidates
  local -a staged_files=()
  local path

  mapfile -t candidates < <(
    {
      git -C "$repo" diff --name-only
      git -C "$repo" diff --cached --name-only
      git -C "$repo" ls-files --others --exclude-standard
    } | awk 'NF' | sort -u
  )

  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 0
  fi

  for path in "${candidates[@]}"; do
    if is_code_path "$path"; then
      staged_files+=("$path")
    fi
  done

  if [[ ${#staged_files[@]} -eq 0 ]]; then
    return 0
  fi

  git -C "$repo" add -A -- "${staged_files[@]}"
}

run_commit() {
  local repo="$1"
  local branch="$2"
  local msg="$3"
  local label="$4"

  ensure_repo "$repo"

  echo "==> [$label] repo: $repo"
  checkout_branch "$repo" "$branch"
  maybe_rebase_branch "$repo" "$branch" "$label"
  stage_code_changes "$repo"

  if git -C "$repo" diff --cached --quiet; then
    echo "==> [$label] no code changes to commit."
    return 0
  fi

  git -C "$repo" commit -m "$msg"
  echo "==> [$label] commit done on branch $branch"
}

run_commit "$SUBMODULE_REPO" "$SUBMODULE_BRANCH" "$SUB_MSG" "submodule"
run_commit "$ROOT_REPO" "$ROOT_BRANCH" "$ROOT_MSG" "root"

echo
echo "All done."
