"""
deploy_qaoa_maxcut.py — Prefect 3.x Cloud Deployment
=====================================================
Registers the QAOA MaxCut pipeline to Prefect Cloud.

BEFORE RUNNING:
  1. Set WORK_POOL_NAME below to your Prefect Cloud work pool name.
  2. Push pipeline code to GitHub:
       git push origin main
  3. Ensure your IQM token is stored as a Prefect Secret:
       from prefect.blocks.system import Secret
       Secret(value="your-iqm-key").save("iqm-resonance-token")

Usage:
    python deploy_qaoa_maxcut.py
"""

from prefect.runner.storage import GitRepository
from qaoa_maxcut_pipeline import qaoa_maxcut_pipeline

# ── CONFIGURE THIS ──────────────────────────────────────────────────────
WORK_POOL_NAME = "my-managed-pool"   # ← replace with your work pool name
GITHUB_URL     = "https://github.com/PlayfulDevBit/qaoa-qec.git"  # ← your repo
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    deployment = qaoa_maxcut_pipeline.from_source(
        source=GitRepository(
            url=GITHUB_URL,
            branch="main",
        ),
        entrypoint="qaoa_maxcut_pipeline.py:qaoa_maxcut_pipeline",
    )

    deployment_id = deployment.deploy(
        name="qaoa-maxcut-iqm-garnet",
        work_pool_name=WORK_POOL_NAME,
        version="1.0.0",
        description=(
            "QAOA MaxCut on IQM Garnet. "
            "Toggleable REM and QEC (Surface Code d=3 on RTH). "
            "QEC always prepares RTH before job submission."
        ),
        tags=["quantum", "qaoa", "maxcut", "iqm-garnet", "rem", "qec", "surface-code"],

        # ── Default parameters (all overridable from Prefect Cloud UI) ──
        parameters={
            "n_iterations": 5,
            "shots": 1024,
            "enable_rem": False,
            "enable_qec": False,
        },

        job_variables={
            "pip_packages": [
                "qiskit==2.1.2",
                "iqm-client[qiskit]==33.0.5",
                "numpy>=1.24",
            ],
        },
    )

    print(f"\nDeployment registered: {deployment_id}")
    print("\nParameter toggles available in Prefect Cloud UI:")
    print("  n_iterations  int   — QAOA layers p (default 5)")
    print("  shots         int   — QPU shots per iteration (default 1024)")
    print("  enable_rem    bool  — Readout Error Mitigation")
    print("  enable_qec    bool  — QEC Surface Code d=3 on RTH")
    print("\nPrerequisites:")
    print("  - Prefect Cloud account with PREFECT_API_URL and PREFECT_API_KEY set")
    print("  - Prefect Secret block 'iqm-resonance-token' containing IQM Resonance API key")
    print(f"  - Code pushed to: {GITHUB_URL}")
