from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
import subprocess
import re
from typing import List, Tuple, Dict, Any

app = FastAPI()

# Allow Next.js dev server to call this API
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# In-memory job store (v0). Later replace with Postgres.
JOBS: Dict[str, Dict[str, Any]] = {}


# ---------- FFmpeg helpers ----------

def get_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def run_silencedetect(
    video_path: str,
    noise_db: str = "-30dB",
    min_silence: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Uses ffmpeg silencedetect to find silent intervals.
    Returns list of (silence_start, silence_end).
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-af", f"silencedetect=noise={noise_db}:d={min_silence}",
        "-f", "null",
        "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr = proc.stderr

    silence_starts: List[float] = []
    silence_ends: List[float] = []

    for line in stderr.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([0-9.]+)", line)
            if m:
                silence_starts.append(float(m.group(1)))
        elif "silence_end:" in line:
            m = re.search(r"silence_end:\s*([0-9.]+)", line)
            if m:
                silence_ends.append(float(m.group(1)))

    n = min(len(silence_starts), len(silence_ends))
    return [(silence_starts[i], silence_ends[i]) for i in range(n)]


def silences_to_keep_intervals(
    duration: float,
    silences: List[Tuple[float, float]],
    pad: float = 0.12,
    min_keep: float = 0.25
) -> List[Dict[str, float]]:
    """
    Convert silent intervals into keep intervals (everything outside silence),
    with small padding to avoid clipping speech boundaries.
    """
    if duration <= 0:
        return []

    # Expand silence by pad (turn silences into cut intervals)
    cuts: List[Tuple[float, float]] = []
    for s, e in silences:
        s2 = max(0.0, s - pad)
        e2 = min(duration, e + pad)
        if e2 > s2:
            cuts.append((s2, e2))

    # Merge overlapping cuts
    cuts.sort()
    merged: List[List[float]] = []
    for s, e in cuts:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # Keep intervals are gaps between merged cuts
    keep: List[Dict[str, float]] = []
    cur = 0.0
    for s, e in merged:
        if s - cur >= min_keep:
            keep.append({"start": round(cur, 3), "end": round(s, 3)})
        cur = max(cur, e)

    if duration - cur >= min_keep:
        keep.append({"start": round(cur, 3), "end": round(duration, 3)})

    return keep


# ---------- Job processing ----------

def process_job(job_id: str):
    """
    Background job:
    - read input video path
    - compute duration
    - detect silent intervals
    - compute keep intervals
    - update JOBS[job_id]
    """
    try:
        JOBS[job_id]["status"] = "processing"
        in_path = JOBS[job_id]["input_path"]

        duration = get_duration(in_path)
        silences = run_silencedetect(in_path, noise_db="-30dB", min_silence=0.5)
        keep_intervals = silences_to_keep_intervals(duration, silences, pad=0.12, min_keep=0.25)
        params = JOBS[job_id].get("params", {"noise_db": "-30dB", "min_silence": 0.5, "pad": 0.12})
        noise_db = params["noise_db"]
        min_silence = params["min_silence"]
        pad = params["pad"]

        JOBS[job_id].update({
            "status": "done",
            "duration": duration,
            "silences": run_silencedetect(in_path, noise_db=noise_db, min_silence=min_silence),
            "keep_intervals": silences_to_keep_intervals(duration, silences, pad=pad, min_keep=0.25),
        })

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


# ---------- API endpoints ----------

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """
    Upload a video to backend/uploads and return file_id.
    """
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(video.filename)[1] or ".mp4"
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    return {"file_id": file_id}


@app.post("/jobs")
async def create_job(
        file_id: str,
        background_tasks: BackgroundTasks,
        noise_db: str = "-30dB",
        min_silence: float = 0.5,
        pad: float = 0.12,
    ):
    """
    Create a processing job for an uploaded file_id.
    """

    matches = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(file_id)]
    if not matches:
        raise HTTPException(status_code=404, detail="file_id not found")

    input_path = os.path.join(UPLOAD_DIR, matches[0])
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "status": "queued",
        "input_path": input_path,
        "params": {
            "noise_db": noise_db,
            "min_silence": min_silence,
            "pad": pad,
        },
        "duration": None,
        "silences": None,
        "keep_intervals": None,
        "error": None,
    }


    background_tasks.add_task(process_job, job_id)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """
    Get job status only (queued/processing/done/error).
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    return {"job_id": job_id, "status": job["status"], "error": job.get("error")}


@app.get("/jobs/{job_id}/cuts")
async def get_cuts(job_id: str):
    """
    Get the computed keep intervals (and optional silences) once done.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    return {
        "job_id": job_id,
        "status": job["status"],
        "duration": job.get("duration"),
        "keep_intervals": job.get("keep_intervals"),
        "silences": job.get("silences"),
        "error": job.get("error"),
    }
