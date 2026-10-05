#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH=""; TAG=""; ENGINE=docker; BASE=nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:4757e6477d94d56fd8230e15b7d59956750c983a47a42584f44dbe143090da82
RECEIPT=""; RESOLVED_LOCK=""; DEVELOPMENT_RESOLVE=0
while (($#)); do
    case "$1" in
      --source-patch) PATCH="$2"; shift 2 ;;
      --tag) TAG="$2"; shift 2 ;;
      --base-image) BASE="$2"; shift 2 ;;
      --engine) ENGINE="$2"; shift 2 ;;
      --receipt) RECEIPT="$2"; shift 2 ;;
      --resolved-lock) RESOLVED_LOCK="$2"; shift 2 ;;
      --development-resolve) DEVELOPMENT_RESOLVE=1; shift ;;
      --help|-h)
        printf 'build-image.sh --source-patch /path/reviewed.patch --resolved-lock /path/reviewed-hashed.lock --tag IMAGE[:TAG] [--base-image IMAGE@sha256:DIGEST] [--engine docker|podman] [--receipt PATH]\nExplicit candidate-only alternative: --development-resolve instead of --resolved-lock. Does not push.\n'
        exit 0 ;;
      *) printf 'Unknown build argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done
[[ -n "$PATCH" && -s "$PATCH" && -n "$TAG" ]] || { printf 'A reviewed nonempty --source-patch and --tag are required.\n' >&2; exit 64; }
if (( DEVELOPMENT_RESOLVE )); then
    [[ -z "$RESOLVED_LOCK" ]] || { printf 'Choose reviewed lock or explicit development resolution.\n' >&2; exit 64; }
else
    [[ -n "$RESOLVED_LOCK" && -s "$RESOLVED_LOCK" ]] || { printf 'A reviewed hashed --resolved-lock is required for the image build.\n' >&2; exit 64; }
fi
[[ "$ENGINE" == docker || "$ENGINE" == podman ]] || { printf 'Engine must be docker or podman.\n' >&2; exit 64; }
command -v "$ENGINE" >/dev/null || { printf 'Container engine unavailable: %s\n' "$ENGINE" >&2; exit 69; }
CONTEXT="$(mktemp -d -t fusion-build.XXXXXXXX)"
trap 'rm -rf -- "$CONTEXT"' EXIT
# The repository and media directories are never sent to the builder.
cp -a -- "$HERE/." "$CONTEXT/"
cp -- "$PATCH" "$CONTEXT/native.patch"
PATCH_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CONTEXT/native.patch")"
LOCK_SHA=""
if [[ -n "$RESOLVED_LOCK" ]]; then
    cp -- "$RESOLVED_LOCK" "$CONTEXT/requirements-linux.resolved.lock"
    LOCK_SHA="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$CONTEXT/requirements-linux.resolved.lock")"
fi
printf 'Building overlay %s on base %s\n' "$PATCH_SHA" "$BASE"
"$ENGINE" build --platform linux/amd64 --file "$CONTEXT/Dockerfile" --build-arg "CUDA_BASE=$BASE" \
    --build-arg "SOURCE_PATCH_SHA256=$PATCH_SHA" --build-arg "REQUIREMENTS_LOCK_SHA256=$LOCK_SHA" \
    --build-arg "DEVELOPMENT_RESOLVE=$DEVELOPMENT_RESOLVE" --tag "$TAG" "$CONTEXT"
if [[ -n "$RECEIPT" ]]; then
    FUSION_BUILD_ENGINE="$ENGINE" FUSION_BUILD_TAG="$TAG" FUSION_BUILD_BASE="$BASE" \
    FUSION_BUILD_PATCH_SHA="$PATCH_SHA" FUSION_BUILD_RECEIPT="$RECEIPT" \
    FUSION_BUILD_LOCK_SHA="$LOCK_SHA" FUSION_BUILD_DEVELOPMENT_RESOLVE="$DEVELOPMENT_RESOLVE" python3 - <<'PY'
import json, os, subprocess, time
from pathlib import Path
result = subprocess.run([os.environ['FUSION_BUILD_ENGINE'], 'image', 'inspect', os.environ['FUSION_BUILD_TAG']],
                        check=True, capture_output=True, text=True)
inspection = json.loads(result.stdout)
labels = inspection[0].get('Config', {}).get('Labels', {})
build_platform = labels.get('io.fusion.cloud-desktop.build-platform', 'unknown')
target_platform = labels.get('io.fusion.cloud-desktop.target-platform', 'unknown')
import_status = ('unknown-no-import-pass-claimed' if 'unknown' in (build_platform, target_platform)
                 else 'deferred-no-import-check' if build_platform != target_platform
                 else 'passed-setup-import-check')
data = {'created_unix': time.time(), 'status': 'built-locally-not-published-not-GPU-validated',
        'source_sha': 'f2d6f5ebe190b631ee5c8972332083cf7f8a3e72',
        'source_patch_sha256': os.environ['FUSION_BUILD_PATCH_SHA'],
        'reviewed_dependency_lock_sha256': os.environ['FUSION_BUILD_LOCK_SHA'],
        'development_resolution': os.environ['FUSION_BUILD_DEVELOPMENT_RESOLVE'] == '1',
        'candidate_or_verified_base': os.environ['FUSION_BUILD_BASE'], 'image': inspection,
        'actual_build_platform': build_platform, 'actual_image_target_platform': target_platform,
        'image_architecture': inspection[0].get('Architecture'),
        'setup_import_check_status': import_status,
        'import_status_evidence': 'recorded build/target platform labels and enforced Dockerfile policy; exact image setup receipt available at /opt/fusion-image/receipts/setup.json',
        'gpu_execution_verified': False, 'native_functionality': 'unvalidated'}
path = Path(os.environ['FUSION_BUILD_RECEIPT'])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2) + '\n')
path.chmod(0o600)
PY
fi
printf 'Image built locally; publication, fresh-pod validation and GPU gates remain separate steps.\n'
