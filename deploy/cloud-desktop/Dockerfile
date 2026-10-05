# Registry-pinned Linux base; local emulated CPU service/recreation trial passed.
# Fresh GPU image execution and publication remain unvalidated.
# Build context: deploy/cloud-desktop only (never media, weights, SSH keys).
ARG CUDA_BASE=nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:4757e6477d94d56fd8230e15b7d59956750c983a47a42584f44dbe143090da82
FROM ${CUDA_BASE}
ARG CUDA_BASE
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG SOURCE_SHA=f2d6f5ebe190b631ee5c8972332083cf7f8a3e72
ARG SOURCE_PATCH_SHA256
ARG REQUIREMENTS_LOCK_SHA256
ARG DEVELOPMENT_RESOLVE=0
LABEL org.opencontainers.image.source="https://github.com/VisoMasterFusion/VisoMaster-Fusion" \
      org.opencontainers.image.revision="f2d6f5ebe190b631ee5c8972332083cf7f8a3e72" \
      io.fusion.cloud-desktop.source-overlay="${SOURCE_PATCH_SHA256}" \
      io.fusion.cloud-desktop.dependency-lock="${REQUIREMENTS_LOCK_SHA256}" \
      io.fusion.cloud-desktop.development-resolve="${DEVELOPMENT_RESOLVE}" \
      io.fusion.cloud-desktop.build-platform="${BUILDPLATFORM}" \
      io.fusion.cloud-desktop.target-platform="${TARGETPLATFORM}" \
      io.fusion.cloud-desktop.import-policy="required-on-native-build-deferred-on-cross-build-see-setup-receipt" \
      io.fusion.cloud-desktop.validation="provisional"
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics,display \
    FUSION_ROOT=/workspace/fusion \
    FUSION_ACCESS_MODE=tunnel \
    FUSION_TRANSPORT=xpra \
    FUSION_SOURCE_REPO=/opt/fusion-source.git \
    FUSION_SOURCE_PATCH=/opt/fusion-deploy/native.patch \
    FUSION_PYTHON=/opt/fusion-image/runtime/venv/bin/python \
    FUSION_DOWNLOAD_MODELS=1 \
    FUSION_MODEL_SELECTION=RetinaFace,Inswapper128ArcFace,Inswapper128,FaceLandmark5,FaceLandmark68,FaceLandmark106,GFPGANv1.4,CodeFormer \
    FUSION_MANAGED_SSH=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY . /opt/fusion-deploy/
RUN chmod +x /opt/fusion-deploy/*.sh /opt/fusion-deploy/*.py \
 && apt-get update \
 && apt-get install -y --no-install-recommends git python3 python3.12 python3.12-venv ca-certificates \
 && test "$SOURCE_SHA" = f2d6f5ebe190b631ee5c8972332083cf7f8a3e72 \
 && test -n "$SOURCE_PATCH_SHA256" \
 && test "$(sha256sum /opt/fusion-deploy/native.patch | cut -d ' ' -f 1)" = "$SOURCE_PATCH_SHA256" \
 && git init --bare /opt/fusion-source.git \
 && git -C /opt/fusion-source.git fetch --no-tags https://github.com/VisoMasterFusion/VisoMaster-Fusion.git "$SOURCE_SHA"
RUN set -eu; defer_flag=""; \
    test -n "$BUILDPLATFORM" && test -n "$TARGETPLATFORM"; \
    if [ "$BUILDPLATFORM" != "$TARGETPLATFORM" ]; then defer_flag="--defer-import-check"; fi; \
    export FUSION_BUILD_PLATFORM="$BUILDPLATFORM" FUSION_TARGET_PLATFORM="$TARGETPLATFORM" FUSION_INSTALL_UV_LINK_MODE=hardlink; \
    if [ "$DEVELOPMENT_RESOLVE" = 1 ]; then \
      FUSION_ROOT=/opt/fusion-image FUSION_BASE_DIGEST="$CUDA_BASE" /opt/fusion-deploy/install.sh install --source-repo /opt/fusion-source.git --source-patch /opt/fusion-deploy/native.patch --skip-models --resolve-lock $defer_flag; \
    else \
      test -n "$REQUIREMENTS_LOCK_SHA256" \
      && test "$(sha256sum /opt/fusion-deploy/requirements-linux.resolved.lock | cut -d ' ' -f 1)" = "$REQUIREMENTS_LOCK_SHA256" \
      && FUSION_ROOT=/opt/fusion-image FUSION_BASE_DIGEST="$CUDA_BASE" /opt/fusion-deploy/install.sh install --source-repo /opt/fusion-source.git --source-patch /opt/fusion-deploy/native.patch --skip-models --resolved-lock /opt/fusion-deploy/requirements-linux.resolved.lock $defer_flag; \
    fi \
 && rm -rf /opt/fusion-image/runtime/uv-cache /opt/fusion-image/runtime/bootstrap /var/lib/apt/lists/*
# No EXPOSE: SSH tunnel is the default. Optional public mode requires credentials.
VOLUME ["/workspace"]
WORKDIR /opt/fusion-deploy
ENTRYPOINT ["/opt/fusion-deploy/entrypoint.sh"]
