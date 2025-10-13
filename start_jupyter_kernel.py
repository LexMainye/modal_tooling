# Start a jupyter kernel on Modal as a sandbox
# See: https://modal.com/docs/examples/jupyter_sandbox
#
# Right now the image is composed of necessary dependencies for running Whisper models
# via FasterWhisper and huggingface transformers. Add other libraries to the kernel as needed
# or install them from the running notebook.
#
# run with:
# python start_jupyter_kernel.py
#
# then see sandbox dashboard:
# https://modal.com/sandboxes/personalizedmodels/main

###########################
# Adjust these
#
JUPYTER_PORT = 8888
# TIMEOUT = 3600 # seconds
TIMEOUT = 86400  # 24 hours maximum for Modal sandbox -- if training longer, consider using a Modal function!
# -> when you use that, don't forget to stop after you're done!
GPU_TYPE = 'L4' # choose according to: https://modal.com/pricing
NUM_CPUS = 8 # for training want more than 1 (4 is good)
MEM = 32768 # for training you need more (16384 is a good default)ccording to: https://modal.com/pricing
###########################


import json
import secrets
import time
import urllib.request

import modal

STORAGE_VOLUME_NAME = "jupyter_kernel"

app = modal.App.lookup(STORAGE_VOLUME_NAME, create_if_missing=True)

volume = modal.Volume.from_name(STORAGE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install(
        # Complete audio codec support
        "wget", "git", "pkg-config", "curl", "aria2",
        "libsndfile1", "libsndfile1-dev", "libflac-dev", "libvorbis-dev",
        "libopus-dev", "libmp3lame-dev", "libfdk-aac-dev", "libspeex-dev",
        
        # FFmpeg with all codecs - ensure latest version
        "ffmpeg", "libavcodec-dev", "libavformat-dev", "libavutil-dev",
        "libswresample-dev", "libavfilter-dev", "libavdevice-dev",
        
        # Audio processing tools
        "libsamplerate0-dev", "libsox-dev", "sox", "rubberband-cli",
        "pulseaudio", "alsa-utils",
        
        # Additional packages for better audio support
        "libportaudio2", "libportaudiocpp0", "portaudio19-dev",
    )
    # Update FFmpeg to latest version and install additional audio tools
    .run_commands([
        "apt update",
        # Remove old FFmpeg and install from multimedia repository for latest version
        "apt remove -y ffmpeg",
        "apt install -y software-properties-common",
        "add-apt-repository -y ppa:savoury1/ffmpeg4",
        "apt update",
        "apt install -y ffmpeg",
        # Verify FFmpeg version
        "ffmpeg -version | head -1",
    ])
    .pip_install(
        "jupyter~=1.1.0",
        "numpy",
        # Install specific compatible versions
        "datasets[audio]==2.16.1",  # Pin to stable version
        "audioread",
        "itables",
        "huggingface_hub[hf_transfer]==0.26.2",        
        "torch",
        "torchaudio",  # Add torchaudio explicitly
        "sounddevice",
        "resampy",
        "mutagen",
        "ctranslate2",
        "faster_whisper",
        "transformers",
        "transformers[torch]",  # Ensure torch dependencies
        "tensorboardX",
        #plotting
        "matplotlib",
        # Audio processing alternatives
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        # Avoid torchcodec issues by using stable audio backends
        "av>=10.0.0",  # PyAV as alternative to torchcodec
        # Evaluation metrics
        "evaluate>=0.4.0",
        "jiwer",  # WER calculation backend
        "sacrebleu",
    )
)


token = secrets.token_urlsafe(13)
token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})





print("🏖️  Creating sandbox")

with modal.enable_output():
    sandbox = modal.Sandbox.create(
        "jupyter",
        "notebook",
        "--no-browser",
        "--allow-root",
        "--ip=0.0.0.0",
        f"--port={JUPYTER_PORT}",
        "--NotebookApp.allow_origin='*'",
        "--NotebookApp.allow_remote_access=1",
        encrypted_ports=[JUPYTER_PORT],
        secrets=[token_secret],
        timeout=TIMEOUT,
        image=image,
        app=app,
        gpu=GPU_TYPE,
        cpu=NUM_CPUS,
        memory=MEM,
        volumes={f"/{STORAGE_VOLUME_NAME}": volume} 
    )

print(f"🏖️  Sandbox ID: {sandbox.object_id}")

tunnel = sandbox.tunnels()[JUPYTER_PORT]
url = f"{tunnel.url}/?token={token}"
print(f"🏖️  Jupyter notebook is running at: {url}")


def is_jupyter_up():
    try:
        response = urllib.request.urlopen(f"{tunnel.url}/api/status?token={token}")
        if response.getcode() == 200:
            data = json.loads(response.read().decode())
            return data.get("started", False)
    except Exception:
        return False
    return False


# timeout for startup
startup_timeout = 60  # seconds
start_time = time.time()
while time.time() - start_time < startup_timeout:
    if is_jupyter_up():
        print("🏖️  Jupyter is up and running!")
        break
    time.sleep(1)
else:
    print("🏖️  Timed out waiting for Jupyter to start.")    