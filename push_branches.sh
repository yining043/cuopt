#!/usr/bin/env bash
set -euo pipefail

ROOT_REPO="/data/jieyi/cuopt"
SUBMODULE_REPO="/data/jieyi/cuopt/basin_callback"
ROOT_BRANCH="landscape"
SUBMODULE_BRANCH="callback_basin"

SET_UPSTREAM=0
AUTO_REBASE=1

usage() {
  cat <<'EOF'
Usage:
  ./push_branches.sh [--set-upstream] [--no-rebase]

Options:
  --set-upstream  Push with -u (useful for first push).
  --no-rebase     Skip pull --rebase before push.
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --set-upstream)
      SET_UPSTREAM=1
      shift
      ;;
    --no-rebase)
      AUTO_REBASE=0
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

sync_before_push() {
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
  echo "==> [$label] sync status before push: ahead=$ahead, behind=$behind"

  if [[ $AUTO_REBASE -eq 1 && "$behind" -gt 0 ]]; then
    echo "==> [$label] pulling with rebase from origin/$branch"
    git -C "$repo" pull --rebase origin "$branch"
  fi
}

push_one() {
  local repo="$1"
  local branch="$2"
  local label="$3"

  ensure_repo "$repo"
  echo "==> [$label] repo: $repo"
  checkout_branch "$repo" "$branch"
  sync_before_push "$repo" "$branch" "$label"

  if [[ $SET_UPSTREAM -eq 1 ]]; then
    git -C "$repo" push -u origin "$branch"
  else
    git -C "$repo" push origin "$branch"
  fi

  echo "==> [$label] pushed $branch"
}

push_one "$SUBMODULE_REPO" "$SUBMODULE_BRANCH" "submodule"
push_one "$ROOT_REPO" "$ROOT_BRANCH" "root"

echo
echo "All pushes done."
