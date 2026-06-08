"""Run sanity_check.py on a Modal A100 40GB — same hardware the autograder
uses. Lets you iterate on your kernels without burning graded submissions.

One-time setup:
    pip install modal
    modal token new            # browser auth, links your Modal account

Run (from the repo root, after editing kernels.py):
    modal run modal_sanity.py

NOTE: First run is slow (~30 min) because Triton 3.7 compiles each unique
kernel spec from scratch. A persistent volume caches the compiled kernels,
so subsequent runs take ~1 minute.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

# Compiled Triton kernels are cached in /root/.triton on the container.
# Persisting this directory to a Modal Volume avoids recompiling on every run.
triton_cache = modal.Volume.from_name("sparse-attn-triton-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("triton")
    # add_local_* must be the LAST build steps.
    .add_local_file(str(HERE / "kernels.py"),        "/app/kernels.py")
    .add_local_file(str(HERE / "kernels_golden.py"), "/app/kernels_golden.py")
    .add_local_file(str(HERE / "bcsr.py"),           "/app/bcsr.py")
    .add_local_file(str(HERE / "harness.py"),        "/app/harness.py")
    .add_local_file(str(HERE / "sanity_check.py"),   "/app/sanity_check.py")
    .add_local_file(str(HERE / "golden.pt"),         "/app/golden.pt")
)

app = modal.App("sparse-attn-student-sanity", image=image)


@app.function(gpu="A100-40GB", timeout=3600,
              volumes={"/root/.triton": triton_cache})
def run_sanity():
    import sys
    sys.path.insert(0, "/app")
    import sanity_check
    sanity_check.main()


@app.local_entrypoint()
def main():
    run_sanity.remote()
