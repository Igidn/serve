#!/usr/bin/env bash
# Download a Hugging Face snapshot and upload it as a private Kaggle Dataset.
#
# Kaggle's working disk is ~20 GB, which is too small for the 21 GB 27B AWQ
# checkpoint (plus the vLLM install). A dataset is mounted read-only under
# /kaggle/input and does not use the working disk, so the gateway can serve the
# weights in place.
#
# Usage:
#   ./scripts/upload_model_dataset.sh <hf-repo> [dataset-slug] [options]
#
# Examples:
#   ./scripts/upload_model_dataset.sh cyankiwi/Qwen3.8-27B-AWQ-INT4
#   ./scripts/upload_model_dataset.sh Qwen/Qwen3-8B-AWQ --title "Qwen3 8B AWQ weights"
#   ./scripts/upload_model_dataset.sh owner/model --workdir /data/staging --clean
#
# Requires the Kaggle CLI with credentials (~/.kaggle/kaggle.json or
# KAGGLE_USERNAME/KAGGLE_KEY), and huggingface_hub for the size preflight.
set -euo pipefail

REPO=""
SLUG=""
DATASET_ID=""
TITLE=""
LICENSE_NAME="other"
WORKDIR=""
ASSUME_YES=false
SKIP_SPACE_CHECK=false
CLEAN_AFTER=false
WAIT_TIMEOUT=1800

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --slug NAME          Dataset slug (default: derived from the repo name)
  --id OWNER/SLUG      Full dataset id (default: current Kaggle username)
  --title TEXT         Dataset title (default: "<repo name> weights")
  --license NAME       Kaggle license name (default: other)
  --workdir DIR        Staging directory (default: ./.model-datasets/<slug>)
  --yes                Do not ask before uploading
  --skip-space-check   Do not verify free disk before downloading
  --clean              Delete the staging directory after a successful upload
  --timeout SECONDS    How long to wait for dataset processing (default: 1800)
  -h, --help           Show this help
EOF
}

note() { printf '%s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

slugify() {
  printf '%s' "$1" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//' \
    | cut -c1-50
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --slug) [[ $# -ge 2 ]] || die "--slug needs a value"; SLUG="$2"; shift 2 ;;
    --id) [[ $# -ge 2 ]] || die "--id needs a value"; DATASET_ID="$2"; shift 2 ;;
    --title) [[ $# -ge 2 ]] || die "--title needs a value"; TITLE="$2"; shift 2 ;;
    --license) [[ $# -ge 2 ]] || die "--license needs a value"; LICENSE_NAME="$2"; shift 2 ;;
    --workdir) [[ $# -ge 2 ]] || die "--workdir needs a value"; WORKDIR="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || die "--timeout needs a value"; WAIT_TIMEOUT="$2"; shift 2 ;;
    --yes|-y) ASSUME_YES=true; shift ;;
    --skip-space-check) SKIP_SPACE_CHECK=true; shift ;;
    --clean) CLEAN_AFTER=true; shift ;;
    --) shift; break ;;
    -*) die "unknown option: $1 (try --help)" ;;
    *)
      if [[ -z "$REPO" ]]; then
        REPO="$1"
      elif [[ -z "$SLUG" ]]; then
        SLUG="$1"
      else
        die "unexpected argument: $1"
      fi
      shift
      ;;
  esac
done

[[ -n "$REPO" ]] || { usage >&2; exit 1; }
[[ "$REPO" == */* ]] || die "expected a Hugging Face repo like 'owner/name', got '$REPO'"
[[ "$REPO" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || die "invalid repo name: $REPO"

command -v kaggle >/dev/null 2>&1 || die "the Kaggle CLI is not installed (pip install kaggle)"
command -v python3 >/dev/null 2>&1 || die "python3 is required"

REPO_NAME="${REPO##*/}"
if [[ -z "$SLUG" ]]; then
  SLUG="$(slugify "$REPO_NAME")"
fi
[[ "$SLUG" =~ ^[a-z0-9][a-z0-9-]*$ ]] \
  || die "invalid dataset slug '$SLUG' (lowercase letters, numbers and dashes)"
[[ "$SLUG" != *"-" ]] || die "dataset slug must not end with a dash"

STAGE="${WORKDIR:-./.model-datasets/$SLUG}"
mkdir -p "$STAGE"
STAGE="$(cd "$STAGE" && pwd)"

TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
if [[ -n "$TOKEN" ]]; then
  export HUGGINGFACE_TOKEN="$TOKEN"
  export HF_TOKEN="$TOKEN"
fi

note "repo:  $REPO"
note "stage: $STAGE"

# ---------------------------------------------------------------------------
# Size preflight. Fail early instead of filling the local disk.
# ---------------------------------------------------------------------------
if [[ "$SKIP_SPACE_CHECK" == false ]]; then
  SIZE_GB="$(python3 - "$REPO" <<'PY' 2>/dev/null || true
import os
import sys

try:
    from huggingface_hub import HfApi
except ImportError:
    sys.exit(2)

token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or None
try:
    info = HfApi(token=token).model_info(sys.argv[1], files_metadata=True)
except Exception:
    sys.exit(2)

total = 0
for sibling in info.siblings or []:
    total += sibling.size or 0
if total <= 0:
    sys.exit(2)
print(total / 1_000_000_000)
PY
)" || true

  if [[ -n "$SIZE_GB" ]]; then
    AVAIL_GB="$(df -Pk "$STAGE" | awk 'NR==2 {printf "%.0f", $4/1000000}')"
    note "need:  ~${SIZE_GB%.*} GB (Hugging Face metadata)"
    note "free:  ${AVAIL_GB} GB in ${STAGE}"
    # 2 GB headroom for the dataset-metadata file and filesystem slack.
    if awk -v need="$SIZE_GB" -v free="$AVAIL_GB" 'BEGIN { exit !(need + 2 > free) }'; then
      die "not enough free disk; rerun with --workdir on a larger volume or --skip-space-check"
    fi
  else
    note "warning: could not read the model size from Hugging Face; skipping the disk check"
  fi
fi

# ---------------------------------------------------------------------------
# Download. `hf` is the modern CLI; fall back to the legacy entry point.
# ---------------------------------------------------------------------------
note "downloading snapshot (this can take a while)"
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --local-dir "$STAGE"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" --local-dir "$STAGE"
else
  python3 - "$REPO" "$STAGE" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
fi

# local_dir downloads keep a small metadata cache that is useless in a dataset.
rm -rf "$STAGE/.cache" "$STAGE/.git"

[[ -f "$STAGE/config.json" ]] \
  || die "download finished but $STAGE/config.json is missing; is this a model repo?"

# ---------------------------------------------------------------------------
# Dataset metadata. `kaggle datasets init` fills in the current username.
# ---------------------------------------------------------------------------
kaggle datasets init -p "$STAGE" >/dev/null

DEFAULT_ID="$(python3 - "$STAGE/dataset-metadata.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    print(json.load(handle).get("id", ""))
PY
)"

if [[ -n "$DATASET_ID" ]]; then
  ID="$DATASET_ID"
else
  OWNER="${DEFAULT_ID%%/*}"
  [[ -n "$OWNER" && "$OWNER" != "INSERT_SLUG_HERE" ]] \
    || die "could not determine your Kaggle username; rerun with --id OWNER/SLUG"
  ID="$OWNER/$SLUG"
fi
[[ "$ID" == */* ]] || die "--id must look like OWNER/SLUG"

if [[ -z "$TITLE" ]]; then
  TITLE="$REPO_NAME weights"
fi

python3 - "$STAGE/dataset-metadata.json" "$ID" "$TITLE" "$LICENSE_NAME" "$REPO" <<'PY'
import json
import sys

path, dataset_id, title, license_name, repo = sys.argv[1:6]
with open(path) as handle:
    metadata = json.load(handle)

metadata["id"] = dataset_id
metadata["title"] = title
metadata["licenses"] = [{"name": license_name}]
metadata["description"] = (
    f"Hugging Face snapshot of {repo}, uploaded for the Serve GPU Gateway. "
    "Mounted read-only under /kaggle/input so it does not use the kernel's working disk."
)

with open(path, "w") as handle:
    json.dump(metadata, handle, indent=2)
    handle.write("\n")
PY

note "dataset: $ID"
note "title:   $TITLE"

SIZE_HUMAN="$(du -sh "$STAGE" | awk '{print $1}')"
note "staged:  $SIZE_HUMAN"

if [[ "$ASSUME_YES" == false && -t 0 ]]; then
  printf 'Upload %s to Kaggle as a private dataset? [y/N] ' "$SIZE_HUMAN"
  read -r answer
  [[ "$answer" == "y" || "$answer" == "Y" || "$answer" == "yes" ]] \
    || die "upload cancelled (the staging directory was kept)"
fi

# ---------------------------------------------------------------------------
# Create or version the dataset. A 21 GB upload can take a long time.
# ---------------------------------------------------------------------------
if kaggle datasets status "$ID" >/dev/null 2>&1; then
  note "dataset exists; creating a new version"
  kaggle datasets version -p "$STAGE" -m "Update from $REPO" --dir-mode zip
else
  note "creating dataset"
  kaggle datasets create -p "$STAGE" --dir-mode zip
fi

# Wait for Kaggle to finish processing so the mount is usable immediately.
ATTEMPTS=$(( WAIT_TIMEOUT / 10 ))
STATUS=""
for ((i = 0; i < ATTEMPTS; i++)); do
  STATUS="$(kaggle datasets status "$ID" 2>/dev/null | tr -d '[:space:]' || true)"
  case "$STATUS" in
    ready|Ready|READY) break ;;
    failed|Failed|FAILED|error|Error|ERROR)
      printf '\n' >&2
      die "Kaggle reported dataset status: $STATUS"
      ;;
  esac
  sleep 10
done

printf '\n' >&2
note "dataset status: ${STATUS:-unknown}"
note "url: https://www.kaggle.com/datasets/$ID"
note ""
note "Next: attach the dataset to the kernel (Kaggle UI -> Add Data), then"
note "restart the kernel. The gateway auto-detects the weights under /kaggle/input."
note "To pin it in .kaggle/kernel-metadata.json, add:"
note "  \"dataset_sources\": [\"$ID\"]"

if [[ "$CLEAN_AFTER" == true ]]; then
  note "removing staging directory $STAGE"
  rm -rf "$STAGE"
  rmdir "$(dirname "$STAGE")" 2>/dev/null || true
fi
