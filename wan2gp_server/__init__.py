"""
wan2gp_server - REST API server for WanGP media generation.

Exposes WanGP's in-process Python API (shared/api.py) as a clean FastAPI
service with three generation endpoints:

    POST /v1/generations/text-to-image
    POST /v1/generations/text-to-video
    POST /v1/generations/image-to-video

Jobs are queued and processed one at a time (the GPU runs a single
generation at once); clients poll GET /v1/jobs/{job_id} for progress and
download outputs from GET /v1/jobs/{job_id}/files/{index}.

This integration uses WanGP by DeepBeepMeep. Use of the WanGP API is
subject to the WanGP Terms and Conditions.
"""

__version__ = "0.1.0"

from .app import create_app  # noqa: F401
