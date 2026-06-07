"""Minimal Python client for wan2gp_server.

Zero dependencies beyond `requests`. Usable from notebooks, scripts, and
agent tools:

    from wan2gp_server.client import Wan2GPServerClient

    client = Wan2GPServerClient("http://localhost:8000")
    client.wait_until_ready()

    job = client.text_to_image("a red bicycle in front of a bakery", width=768, height=768)
    job = client.wait_for(job["id"])                 # poll until terminal
    path = client.download(job, "bicycle.png")       # fetch the first output

    job = client.image_to_video("the bicycle rides away", image="bicycle.png")
    job = client.wait_for(job["id"])
"""

import base64
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import requests


class Wan2GPServerError(RuntimeError):
    pass


class Wan2GPServerClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None, timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["X-API-Key"] = api_key

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        resp = self._session.request(method, f"{self.base_url}{path}", **kwargs)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except ValueError:
                detail = resp.text
            raise Wan2GPServerError(f"{method} {path} -> {resp.status_code}: {detail}")
        return resp.json()

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def wait_until_ready(self, timeout: float = 300, poll: float = 2.0) -> Dict[str, Any]:
        """Wait until the server answers /health (NOT until models are loaded)."""
        deadline = time.time() + timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                return self.health()
            except (requests.ConnectionError, Wan2GPServerError) as exc:
                last_err = exc
                time.sleep(poll)
        raise Wan2GPServerError(f"Server at {self.base_url} not ready after {timeout}s: {last_err}")

    def models(self, task: Optional[str] = None) -> list:
        params = {"task": task} if task else None
        return self._request("GET", "/v1/models", params=params)["models"]

    def preload(self, models: Optional[list] = None, *, wait: bool = True,
                timeout: float = 7200, poll: float = 5.0) -> list:
        """Warm up models (download checkpoints + load weights into VRAM).

        `models` is a list of preset ids; None/empty = the server's default
        t2i/t2v/i2v presets. With wait=True (default) this polls each warmup
        job so progress is printed; returns the final job dicts.
        """
        resp = self._request("POST", "/v1/models/preload", json={"models": models or []})
        jobs = resp["jobs"]
        if not wait:
            return jobs
        done = []
        for job in jobs:
            print(f"--- preloading '{job['model']}' (job {job['id']}) ---")
            done.append(self.wait_for(job["id"], timeout=timeout, poll=poll))
        return done

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def text_to_image(self, prompt: str, *, model: Optional[str] = None,
                      width: int = 1024, height: int = 1024, **kwargs) -> Dict[str, Any]:
        body = {"prompt": prompt, "model": model, "width": width, "height": height, **kwargs}
        return self._request("POST", "/v1/generations/text-to-image", json=body)

    def text_to_video(self, prompt: str, *, model: Optional[str] = None,
                      width: int = 832, height: int = 480,
                      duration_seconds: float = 5.0, **kwargs) -> Dict[str, Any]:
        body = {"prompt": prompt, "model": model, "width": width, "height": height,
                "duration_seconds": duration_seconds, **kwargs}
        return self._request("POST", "/v1/generations/text-to-video", json=body)

    def image_to_video(self, prompt: str, *, image: Union[str, Path], model: Optional[str] = None,
                       duration_seconds: float = 5.0, **kwargs) -> Dict[str, Any]:
        """`image` may be a local file path (uploaded as base64), an http(s)
        URL, a server-side path, or an `asset:<id>` reference."""
        body: Dict[str, Any] = {"prompt": prompt, "model": model,
                                "duration_seconds": duration_seconds, **kwargs}
        image = str(image)
        if image.startswith(("http://", "https://")):
            body["image_url"] = image
        elif image.startswith("asset:"):
            body["image_asset_id"] = image.split(":", 1)[1]
        elif Path(image).is_file():
            body["image_b64"] = base64.b64encode(Path(image).read_bytes()).decode()
        else:
            body["image_path"] = image  # path that exists on the server
        return self._request("POST", "/v1/generations/image-to-video", json=body)

    def upload_asset(self, path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(path)
        with path.open("rb") as fh:
            resp = self._session.post(
                f"{self.base_url}/v1/assets",
                files={"file": (path.name, fh)},
                timeout=self.timeout,
            )
        if resp.status_code >= 400:
            raise Wan2GPServerError(f"POST /v1/assets -> {resp.status_code}: {resp.text}")
        return resp.json()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def job(self, job_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/v1/jobs/{job_id}")

    def jobs(self, limit: int = 50) -> list:
        return self._request("GET", "/v1/jobs", params={"limit": limit})["jobs"]

    def cancel(self, job_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/v1/jobs/{job_id}/cancel")

    def wait_for(
        self,
        job_id: str,
        *,
        timeout: float = 3600,
        poll: float = 3.0,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Poll a job until it reaches a terminal state; raises on failure."""
        deadline = time.time() + timeout
        last_line = ""
        while True:
            job = self.job(job_id)
            if on_progress:
                on_progress(job)
            else:
                p = job.get("progress") or {}
                line = f"[{job['status']}] {p.get('phase') or ''} {p.get('percent', 0)}% {p.get('status') or ''}"
                if line != last_line:
                    print(line)
                    last_line = line
            if job["status"] in ("succeeded", "failed", "cancelled"):
                if job["status"] == "failed":
                    raise Wan2GPServerError(f"Job {job_id} failed: {job.get('error')}")
                return job
            if time.time() > deadline:
                raise Wan2GPServerError(f"Job {job_id} still {job['status']} after {timeout}s")
            time.sleep(poll)

    def download(self, job: Dict[str, Any], dest: Union[str, Path], index: int = 0) -> Path:
        """Download an output file of a finished job to `dest` (file or directory)."""
        files = job.get("files") or []
        if index >= len(files):
            raise Wan2GPServerError(f"Job {job.get('id')} has {len(files)} file(s); index {index} out of range")
        url = f"{self.base_url}{files[index]['url']}"
        dest = Path(dest)
        if dest.is_dir():
            dest = dest / files[index]["filename"]
        with self._session.get(url, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return dest
