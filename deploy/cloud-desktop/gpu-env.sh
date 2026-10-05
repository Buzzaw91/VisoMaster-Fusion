#!/usr/bin/env bash
# Source before launching Python. CUDA 13 leaf wheels use nvidia/cu13/lib;
# ORT 1.26's automatic preloader still searches the older directory layout.
# Prefer this runtime's libraries while retaining container-injected driver
# paths. No host files, devices, packages or loader configuration are changed.
set -Eeuo pipefail
[[ -x "${FUSION_PYTHON:-}" ]] || { printf 'FUSION_PYTHON is required for GPU library discovery.\n' >&2; return 69; }
FUSION_GPU_LIBRARY_PATH="$("$FUSION_PYTHON" - <<'PY'
import importlib.metadata
from pathlib import Path
folders = set()
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get('Name', '').lower().replace('_', '-')
    if not (name.startswith('nvidia-') or name.startswith('tensorrt')):
        continue
    for entry in distribution.files or ():
        if '.so' in Path(entry).name:
            path = Path(distribution.locate_file(entry)).resolve()
            if path.is_file() and 'stubs' not in path.parts:
                folders.add(str(path.parent))
print(':'.join(sorted(folders)))
PY
)"
if [[ -n "$FUSION_GPU_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH="$FUSION_GPU_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export FUSION_GPU_LIBRARY_PATH
