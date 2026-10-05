# Cloud desktop validation — 5 October 2026

This is a pilot record, not a claim that every native feature or GPU combination
passes. The target is the native Fusion GUI at upstream
`f2d6f5ebe190b631ee5c8972332083cf7f8a3e72` (3.12.3), plus a reviewed compatibility
overlay. The fork is [Buzzaw91/VisoMaster-Fusion](https://github.com/Buzzaw91/VisoMaster-Fusion),
branch `codex/cloud-desktop`. See [operations](cloud-desktop-operations.md).

## Environment and measured profile

The supplied running pod has one RTX 4090, 24,564 MiB VRAM, NVIDIA driver
580.159.04, Ubuntu 24.04.4 and a 6.8-core container CPU quota in EU-RO-1. Existing
ComfyUI stayed running and used approximately 456 MiB GPU memory; figures below
include that background allocation. No host drivers, device nodes, borrowed
application environment or existing SSH service were changed.

Inputs are separate 10-second, 300-frame CFR30 derivatives with source audio, at
1280×720 and 1920×1080. One target face is assigned in a two-face clip; a source
image and native merged embedding are used. Original media was preserved. The measured profile uses
CUDA, native model precision, RetinaFace, Inswapper128ArcFace and Inswapper128,
two processing threads and one stream, native markers/ranges and SDR HEVC
main10 through explicit CPU libx265/veryfast. Both experimental CUDA options are
enabled. Full recordings compare restoration off, GFPGAN-v1.4 and CodeFormer,
with original alignment and blend 100; CodeFormer fidelity is 0.9. A second
restoration pass, face editors, frame enhancers, RefLDM and TensorFlow mouth
analysis are disabled.

The runner invokes native Qt actions and processing, observes actual native
processed-frame signals and paints, records through native Record, and checks
decoded exports. ONNX provider profiling and Torch CUDA tracing occur in a
separate untimed diagnostic pass. They establish backend execution there, not
profiling of every timed frame. Native viewport cadence is not browser-delivered
cadence, and audio metadata is not an audible synchronization test.

## Final short benchmark matrix

All eight profiles passed three native recordings each on overlay
`86cd25da933a53071b8311b9bfd4f029a03564c62652ba4bd65be88b9b75102b`.
Full recordings each decode to exactly 300 frames / 10 seconds with AAC audio;
two-range recordings each decode to exactly 180 frames / 6 seconds with AAC.
Every encoder exited successfully, with no tail stalls, fatal worker errors or
mutation of the original projects. The table reports full Record wall time,
including encoding and audio publication; no historical stall time is subtracted.

| Profile | Record seconds, three repeats | Verified render FPS | Native preview p95 | Sampled whole-GPU peak MiB |
| --- | --- | --- | --- | --- |
| 720p, restoration off | 7.043 / 8.337 / 8.793 | 34.12–42.60 | 0.471 s | 2,835 |
| 1080p, restoration off | 11.685 / 11.616 / 11.185 | 25.67–26.82 | 0.492 s | 2,867 |
| 720p, two ranges, off | 5.600 / 5.490 / 5.578 | 32.14–32.78 | 0.466 s | 3,013 |
| 1080p, two ranges, off | 8.083 / 7.913 / 7.667 | 22.27–23.48 | 0.480 s | 3,193 |
| 720p, GFPGAN | 14.751 / 15.071 / 14.886 | 19.91–20.34 | 0.492 s | 3,481 |
| 1080p, GFPGAN | 16.555 / 16.651 / 16.997 | 17.65–18.12 | 0.528 s | 3,613 |
| 720p, CodeFormer | 14.898 / 15.302 / 15.101 | 19.61–20.14 | 0.501 s | 4,319 |
| 1080p, CodeFormer | 16.508 / 16.765 / 17.064 | 17.58–18.17 | 0.501 s | 4,145 |

Cold native window creation took 6.8–7.5 seconds; project loading took another
3.7–4.8 seconds. Warm Find Faces medians were about 60–103 ms for one current
frame. These are not whole-video analysis measurements. Native Play observations
were generally 26–28 FPS without restoration and approximately 15–20 FPS with
restoration. The 720p basic segment profile also had one 16 FPS observation.
Restored playback therefore does not establish continuous 30 FPS preview.

The device and native-PID memory queries are sequential, independent sampled
lower bounds; their peaks are not simultaneous and must not be subtracted.
The table includes separate untimed backend diagnostics and background ComfyUI.
Short sampled peaks of approximately 2.8–4.3 GiB do not certify a minimum-capacity
GPU or long-session stability. CUDA provider traces observed actual core and
GFPGAN execution. CodeFormer executed 2,252 CUDA nodes and 598 CPU nodes in its
untimed diagnostic; it is not entirely GPU-executed.

| Reliability and deployment check | Observed result | Limit |
| --- | --- | --- |
| Native GUI | Mac Chrome displayed and operated Fusion through the SSH desktop tunnel; managed application stayed alive across idle browser reload | Windows, browser cadence and active render/audio reconnect untested |
| Native preview audio | Actual LiveSound/Play produced Qt PlayingState/NoError, own-PID uncorked Pulse input and nonzero monitor samples; Mac Chrome displayed the active native window and advancing unmuted HTML5 audio | Subjective audibility and AV sync need manual verification |
| Stock NVENC | Real encode failed with unsupported-device exit 187 | Explicit CPU encoding is used on this pod |
| Dependencies | Fresh 110-package hashed runtime, package check and CUDA baseline passed | Optional TensorFlow coexistence excluded |
| Save and failure checks | 83 cloud tests on Linux without skips, plus 15 existing audio tests; latest local suite 85 passed with four platform/dependency skips | Disposable copies; no live user-project mutation |
| Writer guard | All 21 Linux checks passed, including orphan-child, native lock and reused-PID protections | Cooperative launcher protection |
| Five-minute interaction | Combined CUDA options completed 67 cycles without OOM | Basic profile; 30-minute test postponed by user to manual use |
| Fresh-process restoration | All 14 native state fields and referenced media hashes matched | Replacement pod and different GPU untested |
| Local image | Final overlay and exact inherited runtime built; CPU desktop, SSH/tunnel, source, native PID 1 lock and clean recreation passed | Cross-build imports deferred; no fresh GPU-image deployment or published image |
| Tail correctness | Delayed final frames stayed ordered; early EOF and known segment skips published exact expected counts; permanent missing frame aborted and preserved prior outputs | Untimed diagnostic cases |

## Earlier failures and corrective evidence

Earlier source/runtime trials are retained below as diagnostic history. Their
rates and memory peaks are not the final matrix. The 30-minute trial was cancelled
before its interaction loop at the user's request on 5 October; it is neither a
pass nor a failure. A long render is unnecessary for that check: repeated editing,
seeking and playback can exercise memory stability during the manual session.

An initial two-range export had 178 frames instead of the expected 180. The
native range end is inclusive, while FFmpeg `-to` is exclusive; the compatibility
patch advances its mux endpoint by one frame. Three corrected 720p exports each
decode to exactly 180 frames / 6 s with AAC. Record wall times were 12.730,
12.888 and 12.757 s. A repeated 1080p seek appeared to time out in the original observer:
diagnostics proved its matching native paint preceded the queued completion
signal. The observer now associates that actual earlier paint; pipeline code and
rendering are unchanged, and reruns retain the original failed evidence.

The sustained 720p run failed with CUDA OOM and ORT BFCArena allocation errors.
At cycle 38 / 168 seconds the native process used approximately 22.4 GB device
memory, versus 742 MB Torch live / 1.24 GB reserved. An operator stop terminated
only this trial process; ComfyUI remained running and GPU memory returned to its
456 MiB baseline. Candidate `VISOMASTER_CUDA_UNIFIED_STREAM=1` selects ORT's
documented shared EP stream without changing native synchronization, model
precision or face algorithms. It is an opt-in hypothesis, not a proven memory
fix. See [ORT stream configuration](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#use_ep_level_unified_stream).

The unified-stream trial completed 69 seek/Play cycles over 301.9 seconds, but
whole-device usage continued increasing. Loaded session identities stayed fixed;
Torch live/reserved allocations were far smaller than the device total. The
next isolated option, `VISOMASTER_CUDA_ARENA_SHRINK=1`, passes ORT RunOptions
`memory.enable_memory_arena_shrinkage=gpu:0` during native CUDA inference. Default
behavior, native synchronization and model precision are unchanged. This is an
experimental allocator setting, not a GPU memory limit or a demonstrated fix.
See [ORT's RunOptions configuration contract](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h).

The combined stream/arena trial completed 67 cycles over 301.5 seconds without
OOM. At 300.1 seconds, measured native/whole-device usage was 2,864 / 3,335 MiB;
interactive peaks were 3,212 / 3,687 MiB. A separate backend diagnostic raised
the full-run peak to 3,755 MiB and observed all seven native binding calls using
`gpu:0` shrink RunOptions. Torch ended at 759 MB live / 1,409 MB reserved. This
is a five-minute improvement, not a 30-minute stability result. A Mac Xpra client
connected after timed exports; its background CPU load is a stress-run confound.

Exports before the queue correction also logged an eight-second natural tail-drain stall. The
native worker retrieved end-of-stream `None` queue entries but omitted their
`task_done()`, leaving the unfinished count positive after real frames finished.
The correction tracks successful acquisition separately and acknowledges the
sentinels in `finally`; genuine in-flight work remains pending. Five real-queue
regressions pass. Three subsequent 720p exports each contained exactly 300 frames,
10 seconds of video and AAC, with full wall times of 6.977, 8.460 and 8.217 seconds.
None logged the old stall. Their whole-device peak was 2,837 MiB. The earlier
720p baseline took 14.648, 14.582 and 14.602 seconds; earlier 1080p measurements
also included the stall; the final matrix above repeats them on corrected code. No time is
subtracted retrospectively. Forced cancellation queue clearing is a separate upstream
lifecycle issue and is not changed by this patch.

A separate untimed delayed-worker diagnostic exposed a second native defect:
once producers finished, the display consumer could take the smallest available
buffered frame rather than wait for its expected frame. The first probe encoded
299 before 298 even though output counts and durations passed. The correction
keeps the consumer cursor ordered, advances over explicitly known skipped frames,
and marks only unread frames unavailable when the decoder reaches early EOF.
It preserves buffered and in-flight work. On the corrected overlay
`77e876db366ed3715800a5ed721ac685d5f0c8684c4a720353ccfcf2070c297f`, the same
actual signal order produced encoder writes exactly 0 through 299. At natural
completion, the observed queue count was zero and the native drain check passed.
These injected-delay results carry no throughput claim.

Four unit regressions cover late queued arrival, known interior/final skips,
unavailable EOF tails and permanent missing frames. The tail timeout is still
eight seconds, but its failure behavior changes: a required frame that never
arrives now aborts the recording rather than publishing a partial result. A
legitimate final worker taking longer than eight seconds can therefore abort.
The bounded native diagnostics below check output removal, prior-output
preservation and the rebuilt audio endpoint.

The missing-frame native probe passed: frame 298 was never emitted or marked
skipped, writes stopped after 297, the eight-second required-frame timeout
aborted, the temporary recording was removed and prior output hashes were
preserved. An early-EOF probe preserved the ordered 298 encoded frames but exposed
loss during audio publication: the temporary video contained 298 frames while
the completed file contained only 294. A segment-end skip probe similarly wrote
178 frames but published 171. A container duration alone concealed these losses.

A bounded CPU reproduction on the pod's FFmpeg 6.1.1 confirmed that copying the
same 298-frame x265 main10 video and bounded AAC with `-shortest` lost two frames.
Without that cutoff, all 298 video packets retained their PTS/DTS identities and
every decoded frame hash matched the temporary video. An ordinary AAC-reencode
control with `-shortest` preserved all 298 frames. The correction therefore
removes the cutoff only from the two rebuilt-audio copy remux commands; their
audio already follows finite inclusive kept-frame ranges. It does not reencode
video or change ordinary no-skip audio arguments. Native EOF and segment-skip
reruns on overlay
`86cd25da933a53071b8311b9bfd4f029a03564c62652ba4bd65be88b9b75102b`
passed the native reruns: early EOF published exactly 298 frames / 9.933333 s
with AAC 9.933 s; segment-end skips published exactly 178 frames / 5.933333 s
with AAC 5.954 s. The latter carries 20.7 ms of AAC padding, which is reported
rather than claimed as sample-exact synchronization. Previous outputs remained
unchanged. The final performance matrix uses this overlay.

## Reproducibility and compatibility decisions

The initial borrowed runtime was not modified. A separate ephemeral environment
was built from a Linux hashed lock. Upstream CUDA toolkit 13.2.1 conflicts with
Torch 2.11.0+cu130 metadata requiring 13.0.2; the Linux candidate follows actual
package constraints instead of forcing incompatible NVIDIA leaf pins. Torch,
vision/audio, ORT, TensorRT, Qt, NumPy, OpenCV and TensorFlow native-facing versions
remain pinned.

Upstream pyqtdarktheme 2.1.0 declares Python below 3.12. The compatible drop-in
`pyqtdarktheme-fork==2.3.6` retains the `qdarktheme` API and supports Python 3.12;
see [the maintained fork's release](https://pypi.org/project/PyQtDarkTheme-fork/2.3.6/).
Visual theme compatibility is checked separately from face output quality.

ORT's exact CUDA13 cp312 Linux wheel is pinned by URL and SHA256. Its CUDA provider
binary matches the working borrowed runtime. Process-start GPU library paths
include the installed NVIDIA wheel directories: ORT 1.26's automatic preloader
still searches an older layout. See [ORT loader source](https://raw.githubusercontent.com/microsoft/onnxruntime/rel-1.26.0/onnxruntime/__init__.py).
No host loader/device workaround is installed.

Preloading optional TensorFlow before ORT caused a reproducible native segfault in
session construction. Torch plus ORT without that preload execute CUDA correctly,
and native baseline inference works. Fusion normally imports TensorFlow lazily for
mouth detection. Preflight checks its import in a separate child and records that
scope; **mixed-framework mouth analysis remains unvalidated and has a known
co-import failure**. A passing baseline preflight does not certify that feature.

Native changes also add atomic workspace/reference publication, visible failed-save
handling, application-held startup exclusion, strict FFmpeg completion checks,
and an explicit SDR CPU encoder setting with a visible GUI label. The stock
NVENC/default arguments, HDR encoder path and face processing remain unchanged.

Astra was consulted for architecture, failure handling and new runtime diagnostics
with `gpt-6-astra`, high reasoning and no inherited turns requested. Accepted
recommendations include additive deployment, explicit encoder fallback, whole-root
and orphan-child exclusion, atomic symlink-aware state saves, exact image overlays
and dependency locks, and testing actual executed providers. The tools did not
expose execution metadata verifying the advisor's actual model/settings.

An additional focused Astra review supported the natural-EOF acknowledgment fix
and recommended checking delayed/out-of-order final frames with actual Qt signal
delivery. The initial full repository pre-commit run passed every hook except 14
existing macOS typing errors in the unchanged Windows-only
`app/helpers/lsfg_bridge.py`. Targeted changed-file hooks pass; this unrelated
platform failure is retained rather than described as a clean full check.

## GPU assessment and remaining acceptance

The RTX 4090 is a measured alternative to RTX Pro 6000 for both resolutions,
including the tested GFPGAN and CodeFormer settings. Full native recordings
finished within 1.71 times clip duration; restoration-off 720p was faster than
real time. CPU quota and x265 encoding are part of these results. The benchmark
establishes the supplied 4090 as usable, not the minimum efficient GPU.

For this one-assigned-face Inswapper profile, 12 GB is a plausible memory-fit
candidate and 16 GB is a reasonable first smaller-GPU benchmark candidate. This is
an inference from sampled memory and the five-minute basic-profile check, not
validation of either capacity. Even 8 GB may fit the observed allocations, but
cold peaks, longer operation and throughput are untested. Multiple assigned
faces, heavier restorers, frame enhancement, HDR and TensorRT may change the
requirement substantially. GPU compute performance and the pod's CPU allocation
matter independently of memory capacity.

For an availability/cost comparison, test a [24 GB RTX 3090 or RTX A5000](https://www.runpod.io/gpu-models)
with the same clip and CPU encoding, or a [16 GB RTX A4000](https://www.runpod.io/gpu-models/rtx-a4000)
as the smaller candidate. These links establish offered models/capacities, not
current availability in the attached volume's datacenter or Fusion performance.
No smaller GPU was rented or benchmarked. The practical choice today is the
4090 measured here with both experimental CUDA options enabled; Pro 6000 capacity is unnecessary for the measured short
profiles. A strict minimum recommendation needs the matched smaller-GPU run.

Manual follow-up covers the postponed 30-minute interaction test, client audio
and AV sync, browser playback cadence and Windows usability. Separate deployment
qualification covers a published immutable image on a fresh GPU pod,
replacement-pod/different-GPU restoration, TensorRT and the optional features.
A short browser reconnect can establish process independence; it does not certify
a render lasting beyond 100 seconds. No additional paid pods were provisioned.
No models were deleted; if quota becomes a constraint, obsolete LTX-2.3 files are
the user's preferred first inventory, with an exact deletion proposal required.
