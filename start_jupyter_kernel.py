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
GPU_TYPE = 'A10'  # choose according to: https://modal.com/pricing
NUM_CPUS = 8      # for training want more than 1 (4 is good)
MEM = 32768       # for training you need more (16384 is a good default) -- see: https://modal.com/pricing

# How long to wait for Jupyter to become reachable after the sandbox is created.
# Cold starts (first run, image not cached) can take 3–5 min; warm starts are ~30s.
STARTUP_TIMEOUT = 600  # seconds
###########################


import json
import secrets
import ssl
import sys
import time
import urllib.error
import urllib.request

import modal

# Modal tunnel certificates may not be in Python's local CA bundle.
# This context is used only for the /api/status health-check poll — we're
# not sending any credentials over it, so skipping verification is safe here.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

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
        # Plotting
        "matplotlib",
        # Audio processing alternatives
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        # Avoid torchcodec issues by using stable audio backends
        "av>=10.0.0",  # PyAV as alternative to torchcodec
        # Evaluation metrics
        "evaluate>=0.4.0",
        "jiwer",    # WER calculation backend
        "sacrebleu",
    )
)


token = secrets.token_urlsafe(13)
token_secret = modal.Secret.from_dict({"JUPYTER_TOKEN": token})


print("🏖️  Creating sandbox...")

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
        volumes={f"/{STORAGE_VOLUME_NAME}": volume},
    )

print(f"🏖️  Sandbox ID: {sandbox.object_id}")

tunnel = sandbox.tunnels()[JUPYTER_PORT]
jupyter_url = f"{tunnel.url}/?token={token}"
status_url = f"{tunnel.url}/api/status?token={token}"

print(f"🏖️  Tunnel established. Waiting for Jupyter to start (timeout: {STARTUP_TIMEOUT}s)...")
print(f"    Cold starts (uncached image) typically take 3–5 min.")
print(f"    Warm starts typically take ~30s.")


def is_jupyter_up(verbose=False):
    """
    Poll the Jupyter /api/status endpoint.
    Returns True when Jupyter reports it has started.
    Prints the specific failure reason when verbose=True, so you can
    distinguish 'still booting' from 'something is actually wrong'.
    """
    try:
        response = urllib.request.urlopen(status_url, timeout=5, context=_SSL_CTX)
        if response.getcode() == 200:
            data = json.loads(response.read().decode())
            return data.get("started", False)
        if verbose:
            print(f"    [poll] Unexpected HTTP status: {response.getcode()}")
    except urllib.error.HTTPError as e:
        if verbose:
            print(f"    [poll] HTTP error: {e.code} {e.reason}")
    except urllib.error.URLError as e:
        # This is normal while the server is still booting — suppress unless verbose
        if verbose:
            print(f"    [poll] URL error (server not yet reachable): {e.reason}")
    except Exception as e:
        if verbose:
            print(f"    [poll] Unexpected error: {e}")
    return False


# Poll with a short initial delay, then slow down slightly to reduce noise.
# Print a progress dot every 10 seconds so you know it's still working.
POLL_INTERVAL = 3      # seconds between each check
VERBOSE_AFTER = 120    # start printing poll errors after this many seconds (helps debug stalls)

start_time = time.time()
last_dot_time = start_time
dot_interval = 10  # print a dot every N seconds

print("    ", end="", flush=True)

while True:
    elapsed = time.time() - start_time

    if elapsed >= STARTUP_TIMEOUT:
        print()  # newline after dots
        print(f"🏖️  Timed out after {STARTUP_TIMEOUT}s waiting for Jupyter to start.")
        print(f"    Sandbox ID: {sandbox.object_id}")
        print(f"    Check the sandbox logs at: https://modal.com/sandboxes/")
        print(f"    You can also try connecting manually once it's up: {jupyter_url}")
        sys.exit(1)

    verbose = elapsed > VERBOSE_AFTER
    if is_jupyter_up(verbose=verbose):
        print()  # newline after dots
        print(f"🏖️  Jupyter is up and running! ({elapsed:.0f}s)")
        print(f"🏖️  Open your notebook: {jupyter_url}")
        break

    # Progress indicator
    now = time.time()
    if now - last_dot_time >= dot_interval:
        print(f".", end="", flush=True)
        last_dot_time = now

    time.sleep(POLL_INTERVAL)