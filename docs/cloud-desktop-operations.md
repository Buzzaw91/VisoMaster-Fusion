# Native Fusion cloud desktop pilot

This deployment runs the native Qt application on a Linux NVIDIA pod. A Mac or
Windows browser receives the desktop; face processing stays on the pod. See
[validation](cloud-desktop-validation.md) for measured results and remaining
gates. Keep the existing ComfyUI and provider SSH services when using the additive
installer on an existing pod.

## Install on an existing pod

Use Ubuntu 24.04 x86_64, Python 3.12, a mounted persistent volume, and enough
ephemeral space for the application environment. The trial environment occupied
approximately 12 GB, plus 13 GB of disposable installer cache. Check the provider's
allocated volume quota: `df` on the shared filesystem can show capacity unrelated
to your allocation. Initial trial observations were 382 GB used of 400 GB.

Check out the fork's reviewed `codex/cloud-desktop` revision. Create the native
patch from that revision's changes to `app/` and `main.py` against upstream
`f2d6f5ebe190b631ee5c8972332083cf7f8a3e72`. The patch must include the new helpers;
an unstaged `git diff` alone omits them. After committing, this is sufficient:

```sh
git diff --binary f2d6f5ebe190b631ee5c8972332083cf7f8a3e72 HEAD -- app main.py > /tmp/native.patch
export FUSION_ROOT=/workspace/fusion
export FUSION_RUNTIME_ROOT=/opt/fusion-runtime
deploy/cloud-desktop/install.sh check
deploy/cloud-desktop/install.sh install \
  --source-patch /tmp/native.patch \
  --resolved-lock deploy/cloud-desktop/requirements-linux.resolved.lock \
  --models RetinaFace,Inswapper128ArcFace,Inswapper128,FaceLandmark5,FaceLandmark68,FaceLandmark106,GFPGANv1.4,CodeFormer
```

Use `--reuse-model-root /absolute/existing/model_assets` for hash-verified reuse.
Use `--skip-apt` only when the documented system packages are already installed.
The installer never starts Fusion. `--borrow-python /existing/venv/bin/python`
is a separate compatibility option: it never installs into that environment and
does not certify conformity to the reviewed lock. `--resolve-only` writes a
candidate lock without syncing application packages; it is a maintenance step,
not a substitute for the supplied lock.

Receipts under `receipts/` record the source overlay, dependencies and installed
system packages. The selection above supports the tested Inswapper profile and its GFPGAN and
CodeFormer comparisons. The image and desktop defaults select these eight
models; downloading the entire native catalog requires explicit `all`. Select
and verify further models before enabling other restorers, editors or masks.

## Start and connect

The pilot pod's stock NVENC probe failed. Its explicit SDR compatibility setting
is CPU HEVC encoding; CUDA still processes faces. The numeric GUI quality value
becomes x265 CRF, which is not interchangeable with NVENC CQ. HDR retains the
upstream encoder path and has not been validated here.

```sh
export FUSION_ROOT=/workspace/fusion
export FUSION_TRANSPORT=xpra
export VISOMASTER_SDR_ENCODER=libx265
export VISOMASTER_SDR_X265_PRESET=veryfast
export VISOMASTER_CUDA_UNIFIED_STREAM=1
export VISOMASTER_CUDA_ARENA_SHRINK=1
deploy/cloud-desktop/entrypoint.sh
```

The two CUDA options are explicit pilot settings. Native defaults remain off;
those defaults exhausted this pod's memory during repeated interaction. The
combined options passed five minutes and the short benchmark matrix. The user
postponed the 30-minute check to manual use, so longer stability is unverified.

The default full preflight requires actual CUDA work, verified native model
weights, visible Qt and a successful selected main10 encode. It reports the stock
NVENC result separately. `FUSION_PREFLIGHT_MODE=imports` is explicitly a GUI-only
compatibility trial. Do not use it to claim GPU readiness.

Xpra shadows the persistent Xfce/TigerVNC display and forwards a private
PulseAudio output sink to its HTML5 client. Forward the loopback desktop endpoint
through the provider's direct SSH connection:

```sh
ssh -N -i /path/to/private-key -p ASSIGNED_EXTERNAL_PORT \
  -L 127.0.0.1:16082:127.0.0.1:6082 root@ASSIGNED_IP
```

Open `http://127.0.0.1:16082/` in Chrome and enable speaker playback if prompted
by the browser. Enable native LiveSound as well as Play for preview audio.
Microphone forwarding is disabled. The noVNC fallback is
`FUSION_TRANSPORT=novnc` and a tunnel from local 16080 to remote 6080; open
`http://127.0.0.1:16080/vnc.html`. **noVNC previews are silent.** Xpra public mode
is refused; use its SSH tunnel. The optional authenticated public noVNC gateway
requires a runtime htpasswd file and a trusted HTTPS proxy.

Browser disconnection leaves Fusion and a render running. Reconnecting restores
the live desktop while that pod remains alive. Stopping a pod ends its processes;
only saved state can be restored, and an interrupted render has no promised
checkpoint. Mac Chrome was exercised in the pilot; Windows client usability still
needs verification.

## Projects, rendering and transfer

Keep inputs and saved projects at stable absolute paths under
`/workspace/fusion/projects/`. Upload through SCP/SFTP, select them in native
Fusion, select the tested CUDA provider with two processing threads and one
stream, find and assign faces, inspect processed previews, set markers/ranges and
use native Record. Exported results must be decoded and checked for expected
frame count, dimensions, duration and audio. Encoder completion alone is
insufficient. Download finished results through SCP/SFTP; keep partial files
separate from completed outputs.

The benchmark assigns one target face in a two-face clip. Additional processed
faces, a second restoration pass, ReF-LDM, frame enhancement, HDR and TensorRT
need their own capacity checks. The upstream default provider is TensorRT;
opening the desktop does not validate that provider. The current manual handover
has CUDA, two threads and one stream selected; the blank native window is ready
for your media. The preserved benchmark snapshots remain under
`/workspace/fusion/benchmark-runs/` if a matched comparison is needed.

Pinned code is staged under `source/<upstream-sha>/`. Native workspace, jobs and
presets map into `native/<upstream-sha>/`; models and external reference KV data
map into `assets/<upstream-sha>/model_assets/`. Profile and logs remain durable.
Compiled engines and temporary files use `cache/<compatibility-key>/`. The key
includes source/overlay, GPU identity, driver, toolchain and model manifest.
Switching GPU selects a new compatible cache without deleting user state.

To stop, finish or abort Record, explicitly save while idle, verify the save log
and workspace, and quit Fusion normally. A failed save keeps the window open.
Then stop the desktop:

```sh
supervisorctl -c /run/fusion-desktop/supervisord.conf shutdown
```

Terminating the supervisor while Fusion is alive can produce an unclean marker;
it is not a substitute for a verified native save and clean quit.

## Backup and recovery

Make a timestamped snapshot while all native writers are idle or stopped. Include
project JSON, input/source media, native state, presets/jobs, the external KV
registry and every payload it references. A workspace file alone is insufficient.
Weights can be recovered from the pinned catalog; reference KV payloads are user
data. Store a hash manifest with each snapshot and verify restoration into a copy
before relying on it. Retain previous known-good snapshots. A network volume is
persistence, not an independent backup.

The writer guard covers the entire storage root, with an additional lock held by
the native process. Direct managed `python main.py`, a second launcher and startup
after an unclean failure are refused. These protections coordinate cooperative
launchers; arbitrary code with root access can bypass them.

Inspect `.writer-state.json`, application/save logs and snapshots after a failed
launch or crash. Prove the recorded guard **and native child** are gone and that
no other pod uses this root. A dead PID on the current machine does not establish
that last condition. Only then recover the exact inspected token:

```sh
python3 deploy/cloud-desktop/runtime.py --root /workspace/fusion \
  recover-lock --expect-token EXACT_INSPECTED_TOKEN
```

Recovery refuses an occupied writer/native lock, live recorded PID or mismatched
token. New Linux markers include boot identity, PID namespace and process start
ticks, so a reused PID 1 in a replacement container is distinguishable from the
original owner. The exact token and inspection are still required; legacy
markers without identity remain conservative. It does not delete state or
silently clear stale markers. Workspace/KV
writes use temporary siblings and atomic replacement, resolving symlink targets.
Referenced assets publish before the workspace. A failed workspace publication
can leave a new unused payload; do not delete references merely because they look
unused. Successful publication plus directory-sync failure is reported as
uncertain durability.

## Image and replacement pod

Build a local candidate with the reviewed patch and hashed lock:

```sh
deploy/cloud-desktop/build-image.sh \
  --source-patch /tmp/native.patch \
  --resolved-lock deploy/cloud-desktop/requirements-linux.resolved.lock \
  --tag fusion-desktop:pilot --receipt /tmp/fusion-build.json
```

This command does not publish or provision a pod. The build context excludes the
repository, media, weights and keys. Runtime startup applies the same source
overlay to a fresh volume and rejects an incompatible existing source tree.

Custom images supervise a dedicated key-only SSH daemon. Supply public material
through `FUSION_SSH_PUBLIC_KEY_FILE`, `PUBLIC_KEY` or `SSH_PUBLIC_KEY`; never place
a private key in an image. Existing pods default to managed SSH disabled. For a
new custom RunPod image, expose **TCP 22**, select a public-IP-capable instance,
and use its assigned external connection from Connect. See [RunPod SSH](https://docs.runpod.io/pods/configuration/use-ssh)
and [port configuration](https://docs.runpod.io/pods/configuration/expose-ports).

Attach the same volume in its datacenter, retain identical project paths, inspect
source/dependency receipts and verify the saved workspace before editing. Network
volumes constrain GPU placement; moving regions requires a verified copy, as
described in [RunPod storage documentation](https://docs.runpod.io/storage/network-volumes).
Actual replacement-pod and different-GPU restoration remain pilot validation
gates. Do not describe them as proved by reopening a process on the same pod.

No storage cleanup has been performed. If the provider quota requires it, first
inventory obsolete LTX-2.3 files as requested by the user, calculate recoverable
space and obtain approval for the exact deletion list.
