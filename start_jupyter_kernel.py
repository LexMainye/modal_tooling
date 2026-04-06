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
#
# ── IMAGE LAYER STRATEGY (read before editing) ───────────────────────────────
# Modal caches each builder step as its own layer, exactly like Docker.
# A change to any step invalidates that step AND every step after it.
# So the order below is intentional:
#
#   Layer 1 – base OS + apt system packages   (almost never changes)
#   Layer 2 – FFmpeg PPA upgrade              (rarely changes)
#   Layer 3 – heavy, slow pip packages        (PyTorch, transformers — pin versions!)
#   Layer 4 – lighter / more volatile pip     (evaluation libs, plotting, etc.)
#
# If you need to add a new pip package, append it to Layer 4.
# Only touch Layers 1–3 if you truly need to; each change forces a full rebuild
# of that layer and everything below it, turning a ~30s warm start into 5+ min.
# ─────────────────────────────────────────────────────────────────────────────

###########################
# Adjust these
#
JUPYTER_PORT = 8888
# TIMEOUT = 3600 # seconds
TIMEOUT = 86400   # 24 hours maximum for Modal sandbox
                  # -> don't forget to stop the sandbox when done!
GPU_TYPE = 'L4'   # choose according to: https://modal.com/pricing
NUM_CPUS = 4      # for training want more than 1 (4 is good)
MEM = 32768       # for training you need more (16384 is a good default)
                  # see: https://modal.com/pricing

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

# ── Layer 1: Base OS + system packages ───────────────────────────────────────
# Only change this if you need a new apt package. Changes here invalidate
# ALL subsequent layers and cause a full image rebuild (~5–10 min).
_base = (
    modal.Image.from_registry("nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04", add_python="3.11")
    .apt_install(
        # Build essentials
        "wget", "git", "pkg-config", "curl", "aria2",
        "software-properties-common",   # needed for add-apt-repository in Layer 2

        # Audio codec libraries
        "libsndfile1", "libsndfile1-dev", "libflac-dev", "libvorbis-dev",
        "libopus-dev", "libmp3lame-dev", "libfdk-aac-dev", "libspeex-dev",

        # FFmpeg dev headers (the ffmpeg binary itself is upgraded in Layer 2)
        "libavcodec-dev", "libavformat-dev", "libavutil-dev",
        "libswresample-dev", "libavfilter-dev", "libavdevice-dev",

        # Audio processing tools
        "libsamplerate0-dev", "libsox-dev", "sox", "rubberband-cli",
        "pulseaudio", "alsa-utils",

        # PortAudio (needed by sounddevice)
        "libportaudio2", "libportaudiocpp0", "portaudio19-dev",
    )
)

# ── Layer 2: FFmpeg upgrade ───────────────────────────────────────────────────
# Separated from Layer 1 so you can bump the PPA without reinstalling all
# the apt packages above. Still slow when it misses cache (~2–3 min).
_with_ffmpeg = (
    _base
    .run_commands([
        # Add the FFmpeg4 PPA and upgrade — combined to minimise cache layers
        "add-apt-repository -y ppa:savoury1/ffmpeg4 && apt-get update -qq",
        "apt-get install -y --no-install-recommends ffmpeg",
        # Sanity-check the installed version
        "ffmpeg -version | head -1",
    ])
)

# ── Layer 3: Heavy pip packages (slow to install, rarely change) ──────────────
# PyTorch + HuggingFace ecosystem. Pin versions so Modal's cache key is stable.
# Changing a version here busts this layer and Layer 4.
_with_torch = (
    _with_ffmpeg
    .pip_install(
        # PyTorch — install first so everything below links against the same build
        "torch",
        "torchaudio",

        # HuggingFace core
        "transformers[torch]==4.47.0",   # pin for reproducibility
        "datasets[audio]==2.16.1",
        "huggingface_hub[hf_transfer]==0.26.2",

        # Whisper inference
        "ctranslate2",
        "faster_whisper",
    )
)

# ── Layer 4: Lighter / more volatile pip packages ─────────────────────────────
# Add new packages HERE first — it only rebuilds this layer, not Layer 3.
image = (
    _with_torch
    .pip_install(
        # Jupyter
        "jupyter~=1.1.0",

        # Numeric / data
        "numpy",
        "itables",

        # Audio processing
        "audioread",
        "sounddevice",
        "resampy",
        "mutagen",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "av>=10.0.0",           # PyAV — stable alternative to torchcodec

        # Training utilities
        "tensorboardX",

        # Plotting
        "matplotlib",

        # Evaluation metrics
        "evaluate>=0.4.0",
        "jiwer",                # WER calculation backend
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
status_url  = f"{tunnel.url}/api/status?token={token}"

print(f"🏖️  Tunnel established. Waiting for Jupyter to start (timeout: {STARTUP_TIMEOUT}s)...")
print(f"    Cold starts (uncached image) typically take 3–5 min.")
print(f"    Warm starts typically take ~30s.")
print()
print(f"    💡 IDE tip: wait for the '🏖️  Ready' message below,")
print(f"       then connect from VS Code / PyCharm using the full URL including ?token=...")
print(f"       The kernel manager needs a moment after the HTTP endpoint responds.")


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
        # Normal while the server is still booting — suppress unless verbose
        if verbose:
            print(f"    [poll] URL error (server not yet reachable): {e.reason}")
    except Exception as e:
        if verbose:
            print(f"    [poll] Unexpected error: {e}")
    return False


# Poll with a short initial delay, then slow down slightly to reduce noise.
# Print a progress dot every 10 seconds so you know it's still working.
POLL_INTERVAL = 1      # seconds between each check (1s catches readiness almost instantly)
VERBOSE_AFTER = 120    # start printing poll errors after this many seconds

# Extra grace period after Jupyter reports started, before printing the URL.
# Reduced from 15s → 3s; double-confirm logic below replaces the blind sleep.
POST_READY_GRACE = 3   # seconds

start_time    = time.time()
last_dot_time = start_time
dot_interval  = 5      # print a dot every N seconds (halved for better feedback)

print("    ", end="", flush=True)

while True:
    elapsed = time.time() - start_time

    if elapsed >= STARTUP_TIMEOUT:
        print()
        print(f"🏖️  Timed out after {STARTUP_TIMEOUT}s waiting for Jupyter to start.")
        print(f"    Sandbox ID: {sandbox.object_id}")
        print(f"    Check the sandbox logs at: https://modal.com/sandboxes/")
        print(f"    You can also try connecting manually once it's up: {jupyter_url}")
        sys.exit(1)

    verbose = elapsed > VERBOSE_AFTER
    if is_jupyter_up(verbose=verbose):
        # Double-confirm: poll once more after a short pause to ensure the kernel
        # manager is fully initialised, rather than sleeping blindly.
        print()
        print(f"🏖️  Jupyter responded! ({elapsed:.0f}s) — confirming kernel readiness...")
        time.sleep(POST_READY_GRACE)
        if is_jupyter_up(verbose=True):
            print(f"🏖️  Ready. Open your notebook:")
            print(f"    {jupyter_url}")
            break
        else:
            # Briefly not ready — continue polling; counts towards STARTUP_TIMEOUT
            print(f"    [confirm] Not quite ready yet, continuing to poll...")
            continue

    # Progress indicator
    now = time.time()
    if now - last_dot_time >= dot_interval:
        print(".", end="", flush=True)
        last_dot_time = now

    time.sleep(POLL_INTERVAL)