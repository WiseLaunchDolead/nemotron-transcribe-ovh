#!/usr/bin/env bash
set -euo pipefail

# Checks linking and CLI availability without downloading weights or using a GPU.
if [[ "${1:-}" == "--check" && "$#" == 1 ]]; then
    nemo-speech --version
    nemo-speech serve --help >/dev/null
    exit 0
fi
if [[ "$#" != 0 ]]; then
    echo 'Usage: start.sh [--check]' >&2
    exit 2
fi
case "${AUTH_MODE:-api-key}" in
    api-key)
        if [[ -z "${NEMO_SPEECH_HTTP_API_KEY:-}" ]]; then
            echo 'Set the secret NEMO_SPEECH_HTTP_API_KEY before starting the service.' >&2
            exit 2
        fi
        ;;
    ovh)
        # OVH's restricted ingress validates Authorization itself. Avoid two independent
        # Bearer secrets competing for the same header (or relying on header forwarding).
        if [[ -n "${NEMO_SPEECH_HTTP_API_KEY:-}" ]]; then
            echo 'AUTH_MODE=ovh uses restricted OVH ingress; remove the native API key.' >&2
            exit 2
        fi
        echo 'AUTH_MODE=ovh: restricted OVHcloud access MUST be enabled.' >&2
        ;;
    *) echo 'AUTH_MODE must be api-key or ovh.' >&2; exit 2 ;;
esac
# Fail before a 21 GB download if the NVIDIA driver/GPU is absent.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
MODEL_DIR="${MODEL_DIR:-/workspace/models}"
export MODEL_DIR
python3 /opt/app/download_models.py --directory "$MODEL_DIR"
export NEMO_SPEECH_ASR_MODEL_PATH="$MODEL_DIR/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-IQ4_NL.gguf"
export NEMO_SPEECH_ASR_SALM_ENCODER_PATH="$MODEL_DIR/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf"
exec nemo-speech serve --config /opt/app/config.yaml
