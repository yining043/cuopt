#!/usr/bin/env bash
# 从系统恢复 /home/jieyi/cuopt/perturb_k1_collect 下被删的 jsonl（不合并）。
#
# 依赖: extundelete。分区: /，设备 /dev/nvme0n1p4，ext4。
#
# 用法:
#   ./restore_deleted_k1_results.sh [恢复输出目录] [分区] [after时间]
#   只恢复 perturb_k1_collect 目录；after 为 today 时只恢复今天删的。
#
# 只恢复今天在 perturb_k1_collect 里删掉的 jsonl:
#   ./restore_deleted_k1_results.sh "" "" today
#
# 注意: 分区挂载时也可运行 extundelete，但恢复率可能不如从 Live USB 挂载只读再跑。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 恢复输出目录：优先用 /data 避免写满根分区
RECOVERY_BASE="${1:-/data/perturb_k1_recovered}"
if [[ ! -d /data ]]; then
  RECOVERY_BASE="${SCRIPT_DIR}/perturb_k1_recovered"
fi
[[ -z "${RECOVERY_BASE}" ]] && RECOVERY_BASE="/data/perturb_k1_recovered"
[[ ! -d /data ]] && [[ "${RECOVERY_BASE}" == /data/* ]] && RECOVERY_BASE="${SCRIPT_DIR}/perturb_k1_recovered"

REL_PATH="home/jieyi/cuopt/perturb_k1_collect"
PARTITION="${2:-/dev/nvme0n1p4}"
[[ -z "${PARTITION}" ]] && PARTITION="/dev/nvme0n1p4"

# 只恢复该时间之后被删除的文件
AFTER_ARG="${3:-}"
if [[ "${AFTER_ARG}" == "today" ]] || [[ "${AFTER_ARG}" == "今日" ]]; then
  AFTER_TS=$(date -d "today 00:00:00" +%s 2>/dev/null || date -d "00:00" +%s 2>/dev/null || echo "0")
elif [[ -z "${AFTER_ARG}" ]]; then
  AFTER_TS=$(date -d "30 days ago" +%s 2>/dev/null || echo "0")
elif [[ "${AFTER_ARG}" =~ ^[0-9]+$ ]]; then
  # 纯数字：若小于 10000000000 视为“N 天前”，否则视为 Unix 时间戳
  if [[ "${AFTER_ARG}" -lt 10000000000 ]]; then
    AFTER_TS=$(date -d "${AFTER_ARG} days ago" +%s 2>/dev/null || echo "0")
  else
    AFTER_TS="${AFTER_ARG}"
  fi
else
  AFTER_TS=$(date -d "${AFTER_ARG}" +%s 2>/dev/null || echo "0")
fi

echo "=============================================="
echo "恢复被删的 k1_collection_results batch jsonl"
echo "=============================================="
echo "  分区:      ${PARTITION}"
echo "  相对路径:  ${REL_PATH}"
echo "  恢复目录:  ${RECOVERY_BASE}"
echo "  --after:   ${AFTER_TS}"
echo "=============================================="

if ! command -v extundelete &>/dev/null; then
  echo "未找到 extundelete。若已安装请确认 PATH；否则请管理员安装: apt-get install extundelete"
  exit 1
fi

# 1) 用 extundelete 恢复该目录下被删文件
mkdir -p "${RECOVERY_BASE}"
cd "${RECOVERY_BASE}"
# 会生成 RECOVERED_FILES/ 或按分区名生成子目录
echo "[1/3] 正在用 extundelete 扫描并恢复被删文件（可能较久）..."
if sudo extundelete "${PARTITION}" --restore-directory "${REL_PATH}" --after "${AFTER_TS}"; then
  echo "  extundelete 完成."
else
  echo "  extundelete 返回非 0，请检查权限/分区。继续尝试复制已恢复文件..."
fi

# 2) 找到恢复出来的 perturb_k1_collect 目录（下有 cvrp100_uniform.pkl#0 等）
RECOVERED_ROOT=""
if [[ -d "RECOVERED_FILES/${REL_PATH}" ]]; then
  RECOVERED_ROOT="RECOVERED_FILES/${REL_PATH}"
elif [[ -d "RECOVERED_FILES" ]]; then
  # 查找第一个含 # 的子目录的父目录
  first_inst=$(find RECOVERED_FILES -maxdepth 5 -type d -name "*#*" 2>/dev/null | head -1)
  if [[ -n "${first_inst}" ]]; then
    RECOVERED_ROOT="$(dirname "${first_inst}")"
  fi
fi
if [[ -z "${RECOVERED_ROOT}" ]]; then
  found=$(find . -maxdepth 6 -type d -name "cvrp100_uniform.pkl#0" 2>/dev/null | head -1)
  if [[ -n "${found}" ]]; then
    RECOVERED_ROOT="$(dirname "${found}")"
  fi
fi

if [[ -z "${RECOVERED_ROOT}" ]] || [[ ! -d "${RECOVERED_ROOT}" ]]; then
  echo "[2/3] 未找到恢复出的目录（${RECOVERY_BASE} 下应有 perturb_k1_collect 或 cvrp100_uniform.pkl#*）。"
  echo "  若已用 PhotoRec 等恢复到了其它目录，可单独运行:"
  echo "  python3 ${SCRIPT_DIR}/merge_recovered_k1_results.py --recovered_root <恢复目录> --target_base ${SCRIPT_DIR}/perturb_k1_collect --operator_type remove_and_insert --runs 30 --no_merge"
  exit 1
fi

echo "[2/3] 恢复目录: ${RECOVERY_BASE}/${RECOVERED_ROOT}"

# 3) 只把恢复出的 batch results jsonl 拷回 perturb_k1_collect，不合并
echo "[3/3] 复制 batch results jsonl 回 perturb_k1_collect（不合并）..."
python3 "${SCRIPT_DIR}/merge_recovered_k1_results.py" \
  --recovered_root "${RECOVERY_BASE}/${RECOVERED_ROOT}" \
  --target_base "${SCRIPT_DIR}/perturb_k1_collect" \
  --operator_type remove_and_insert \
  --runs 30 \
  --no_merge

echo "=============================================="
echo "完成。batch 文件已拷回 perturb_k1_collect/*/；需要合并时再运行 merge_k1_collect_batches.py"
echo "=============================================="
