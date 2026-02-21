from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
import subprocess
import re
from typing import List, Tuple, Dict, Any
from faster_whisper import WhisperModel
import tempfile
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()
WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


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

# audio extraction + filler detection + interval subtraction
FILLER_WORDS = {
    "um", "uh", "erm", "ah", "like", "okay", "ok", "so", "yeah",
    "you know", "i mean", "basically", "actually", "literally"
}

def extract_audio_wav(video_path: str, wav_path: str):
    """
    Extract mono 16kHz WAV for ASR.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        wav_path
    ]
    subprocess.check_call(cmd)



def transcribe_with_word_timestamps(wav_path: str):
    """
    Returns:
      - transcript_segments: [{start, end, text}]
      - words: [{start, end, word}]
    """
    segments, _info = WHISPER_MODEL.transcribe(
        wav_path,
        word_timestamps=True,
        vad_filter=True,  # helps ignore long silence/noise
    )

    transcript_segments = []
    words = []

    for seg in segments:
        transcript_segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })
        if seg.words:
            for w in seg.words:
                # w.word may contain punctuation, normalize lightly
                word = (w.word or "").strip()
                if word:
                    words.append({
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": word
                    })

    return transcript_segments, words


def normalize_token(token: str) -> str:
    # lowercase and strip punctuation-ish chars
    t = token.lower().strip()
    t = re.sub(r"[^\w']+", "", t)
    return t


def detect_filler_intervals(words, pad: float = 0.05):
    """
    Detect filler words; returns cut intervals [{start, end, label}].
    For multiword fillers (e.g., "you know"), we detect via bigrams.
    """
    cut_intervals = []

    # single-word fillers
    single = {w for w in FILLER_WORDS if " " not in w}
    multi = [w for w in FILLER_WORDS if " " in w]

    norm_words = [normalize_token(w["word"]) for w in words]

    for i, w in enumerate(words):
        tok = norm_words[i]
        if tok in single and w["end"] > w["start"]:
            cut_intervals.append({
                "start": max(0.0, w["start"] - pad),
                "end": w["end"] + pad,
                "label": tok
            })

    # multi-word fillers (bigrams/trigrams simple)
    for phrase in multi:
        parts = [normalize_token(p) for p in phrase.split()]
        n = len(parts)
        for i in range(0, len(words) - n + 1):
            window = norm_words[i:i+n]
            if window == parts:
                start = words[i]["start"]
                end = words[i+n-1]["end"]
                cut_intervals.append({
                    "start": max(0.0, start - pad),
                    "end": end + pad,
                    "label": phrase
                })

    # merge overlaps
    cut_intervals.sort(key=lambda x: x["start"])
    merged = []
    for c in cut_intervals:
        if not merged or c["start"] > merged[-1]["end"]:
            merged.append(c)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
            merged[-1]["label"] = merged[-1]["label"]  # keep first label
    return merged


def subtract_cut_intervals(keep_intervals, cut_intervals, min_keep: float = 0.20):
    """
    keep_intervals: [{start,end}]
    cut_intervals: [{start,end,...}]
    returns refined keep intervals
    """
    if not keep_intervals:
        return []

    cuts = [(c["start"], c["end"]) for c in cut_intervals if c["end"] > c["start"]]
    cuts.sort()

    refined = []
    for k in keep_intervals:
        ks, ke = k["start"], k["end"]
        cur = ks
        for cs, ce in cuts:
            if ce <= cur:
                continue
            if cs >= ke:
                break
            if cs > cur and cs - cur >= min_keep:
                refined.append({"start": round(cur, 3), "end": round(cs, 3)})
            cur = max(cur, ce)
        if ke - cur >= min_keep:
            refined.append({"start": round(cur, 3), "end": round(ke, 3)})

    return refined



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

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def detect_repetition_cuts(
    transcript_segments,
    threshold: float = 0.88,
    min_chars: int = 30,
    pad: float = 0.08,
    lookback: int = 12,
):
    """
    Detect near-duplicate segments using embeddings.
    Returns cut intervals: [{start,end,label,score,match_index}]
    Strategy: compare each segment to a rolling window of previously kept segments.
    """
    # filter segments with enough content
    texts = [s["text"].strip() for s in transcript_segments]
    keep_idx = []
    cut_intervals = []

    # embed all upfront for speed
    embeddings = EMBED_MODEL.encode(texts, normalize_embeddings=True)

    for i, seg in enumerate(transcript_segments):
        text = seg["text"].strip()
        if len(text) < min_chars:
            keep_idx.append(i)
            continue

        # compare to previous kept segments in window
        candidates = keep_idx[-lookback:] if lookback > 0 else keep_idx
        best_score = -1.0
        best_j = None

        for j in candidates:
            score = float(np.dot(embeddings[i], embeddings[j]))  # normalized => dot = cosine
            if score > best_score:
                best_score = score
                best_j = j

        if best_score >= threshold and best_j is not None:
            # Mark THIS segment as redundant (cut it)
            cut_intervals.append({
                "start": max(0.0, float(seg["start"]) - pad),
                "end": float(seg["end"]) + pad,
                "label": "repetition",
                "score": round(best_score, 3),
                "match_index": best_j,
            })
            # do NOT add to keep_idx
        else:
            keep_idx.append(i)

    # merge overlaps
    cut_intervals.sort(key=lambda x: x["start"])
    merged = []
    for c in cut_intervals:
        if not merged or c["start"] > merged[-1]["end"]:
            merged.append(c)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
            merged[-1]["score"] = max(merged[-1].get("score", 0), c.get("score", 0))
    return merged

def apply_time_budget(keep_intervals, target_seconds: float, min_keep: float = 0.20):
    """
    Keep content from the start until we reach target_seconds.
    (This is the simple MVP version; later we'll choose best segments.)
    """
    out = []
    remaining = float(target_seconds)

    for k in keep_intervals or []:
        s, e = float(k["start"]), float(k["end"])
        dur = max(0.0, e - s)
        if dur <= 0:
            continue

        if dur <= remaining:
            out.append({"start": round(s, 3), "end": round(e, 3)})
            remaining -= dur
        else:
            # partial segment
            if remaining >= min_keep:
                out.append({"start": round(s, 3), "end": round(s + remaining, 3)})
            break

        if remaining <= 0:
            break

    return out

def make_time_chunks_from_words(words, chunk_seconds=3.0, hop_seconds=1.5, min_words=5):
    """
    Sliding windows: 3.0s chunks every 1.5s (overlapping).
    Returns: [{start,end,text}]
    """
    chunks = []
    if not words:
        return chunks

    i = 0
    while i < len(words):
        start_t = float(words[i]["start"])
        end_t = start_t + chunk_seconds

        j = i
        while j < len(words) and float(words[j]["end"]) <= end_t:
            j += 1

        if j - i >= min_words:
            text = " ".join(w["word"].strip() for w in words[i:j]).strip()
            chunks.append({
                "start": float(words[i]["start"]),
                "end": float(words[j-1]["end"]),
                "text": text
            })

        # move i forward by hop_seconds (time-based)
        next_i = i + 1
        while next_i < len(words) and float(words[next_i]["start"]) < start_t + hop_seconds:
            next_i += 1
        i = next_i

    return chunks

INFO_PATTERNS = [
    r"\bkey\b", r"\bimportant\b", r"\bmain point\b", r"\bidea\b",
    r"\bstep\b", r"\bfirst\b", r"\bsecond\b", r"\bthird\b",
    r"\bbecause\b", r"\btherefore\b", r"\bso that\b", r"\bmeans\b",
    r"\bfor example\b", r"\be\.g\.\b", r"\bin summary\b", r"\bconclusion\b",
]
INFO_RE = re.compile("|".join(INFO_PATTERNS), re.IGNORECASE)

def info_density_score(text: str) -> float:
    """
    Simple, interpretable heuristics.
    """
    t = (text or "").strip()
    if not t:
        return 0.0

    score = 0.0

    # numbers often signal concrete info
    if re.search(r"\d", t):
        score += 0.25

    # “key/important/step/because...” cues
    if INFO_RE.search(t):
        score += 0.35

    # longer (but not too long) segments tend to be more informative
    n = len(t)
    if n >= 60:
        score += 0.15
    if n >= 120:
        score += 0.05  # diminishing returns

    return score


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # If you used normalize_embeddings=True, dot is cosine.
    return float(np.dot(a, b))


def select_best_segments_under_budget(chunks, embeddings, budget_seconds: float,
                                     w_central=0.55, w_novel=0.35, w_info=0.25,
                                     novelty_threshold=0.88):
    """
    Greedy selection maximizing value per second with novelty constraint.
    - centrality: similarity to overall embedding
    - novelty: penalize similarity to already selected chunks
    - info: heuristic boost
    """
    if not chunks:
        return []

    # Overall topic embedding = mean of chunk embeddings (already normalized)
    overall = np.mean(embeddings, axis=0)
    overall = overall / (np.linalg.norm(overall) + 1e-12)

    # Precompute centrality + info + duration
    items = []
    for idx, c in enumerate(chunks):
        dur = max(0.001, float(c["end"]) - float(c["start"]))
        text = c["text"]
        central = cosine(embeddings[idx], overall)  # 0..1-ish
        info = info_density_score(text)
        items.append({
            "idx": idx,
            "start": float(c["start"]),
            "end": float(c["end"]),
            "dur": dur,
            "text": text,
            "central": central,
            "info": info,
        })

    # Greedy: repeatedly pick best remaining chunk that fits time + is novel enough
    selected = []
    selected_embs = []
    remaining = float(budget_seconds)

    # To avoid picking tons of overlapping windows, we will later merge, but also lightly discourage overlap
    while remaining > 0.2:
        best = None
        best_score_per_sec = -1e9

        for it in items:
            if it.get("picked"):
                continue
            if it["dur"] > remaining + 1e-6:
                continue

            # novelty vs selected
            if selected_embs:
                sims = [cosine(embeddings[it["idx"]], e) for e in selected_embs]
                max_sim = max(sims)
            else:
                max_sim = 0.0

            # If it's too similar to what we've kept, skip (hard constraint)
            if max_sim >= novelty_threshold:
                continue

            novelty = 1.0 - max_sim  # higher is better

            value = (w_central * it["central"]) + (w_novel * novelty) + (w_info * it["info"])
            score_per_sec = value / it["dur"]

            if score_per_sec > best_score_per_sec:
                best_score_per_sec = score_per_sec
                best = it

        if best is None:
            break

        best["picked"] = True
        selected.append({"start": best["start"], "end": best["end"], "label": "smart_keep"})
        selected_embs.append(embeddings[best["idx"]])
        remaining -= best["dur"]

    # Sort by time and merge overlaps/adjacent into clean keep intervals
    selected.sort(key=lambda x: x["start"])
    merged = []
    for s in selected:
        if not merged:
            merged.append({"start": round(s["start"], 3), "end": round(s["end"], 3)})
            continue
        if s["start"] <= merged[-1]["end"] + 0.10:  # merge if overlapping or very close
            merged[-1]["end"] = round(max(merged[-1]["end"], s["end"]), 3)
        else:
            merged.append({"start": round(s["start"], 3), "end": round(s["end"], 3)})

    return merged

# ---------- Job processing ----------

def process_job(job_id: str):
    try:
        JOBS[job_id]["status"] = "processing"
        in_path = JOBS[job_id]["input_path"]

        # --- Step A: silence detection ---
        params = JOBS[job_id].get("params", {"noise_db": "-30dB", "min_silence": 0.5, "pad": 0.12})
        noise_db = params["noise_db"]
        min_silence = float(params["min_silence"])
        pad = float(params["pad"])

        duration = get_duration(in_path)
        silences = run_silencedetect(in_path, noise_db=noise_db, min_silence=min_silence)
        keep_intervals = silences_to_keep_intervals(duration, silences, pad=pad, min_keep=0.25)

        # --- Step B: transcription + filler cuts ---
        with tempfile.TemporaryDirectory() as td:
            wav_path = os.path.join(td, "audio.wav")
            extract_audio_wav(in_path, wav_path)

            transcript_segments, words = transcribe_with_word_timestamps(wav_path)
            filler_cuts = detect_filler_intervals(words, pad=0.05)

            rep_chunks = make_time_chunks_from_words(words, chunk_seconds=3.0, hop_seconds=1.5, min_words=5)
            repetition_cuts = detect_repetition_cuts(
                rep_chunks,
                threshold=0.88,
                min_chars=25,
                pad=0.08,
                lookback=12,
            )

            # Combine filler + repetition cuts and sort
            all_cuts = filler_cuts + repetition_cuts
            all_cuts.sort(key=lambda x: x["start"])

        # Subtract all cuts from silence-based keep intervals
        final_keep = subtract_cut_intervals(keep_intervals, all_cuts, min_keep=0.20)

        # --- Keep% budget ---
        target_pct = float(params.get("target_pct", 1.0))
        target_pct = max(0.4, min(1.0, target_pct))
        budget_seconds = duration * target_pct

        # Build candidate chunks
        chunks = make_time_chunks_from_words(
            words,
            chunk_seconds=3.0,
            hop_seconds=1.5,
            min_words=5,
        )

        # --- Smart compression guard ---
        if not chunks:
            # Fallback: if chunking failed, just use cleaned intervals
            smart_keep = final_keep
        else:
            texts = [c["text"] for c in chunks]
            embs = EMBED_MODEL.encode(texts, normalize_embeddings=True)

            smart_keep = select_best_segments_under_budget(
                chunks,
                embs,
                budget_seconds=budget_seconds,
                w_central=0.55,
                w_novel=0.35,
                w_info=0.25,
                novelty_threshold=0.88,
            )

        # IMPORTANT: smart_keep is a keep plan, but it may include filler/repetition regions.
        # So subtract all_cuts again to ensure removed content stays removed.
        smart_keep = subtract_cut_intervals(smart_keep, all_cuts, min_keep=0.20)

        JOBS[job_id].update({
            "status": "done",
            "duration": duration,
            "silences": [{"start": s, "end": e} for (s, e) in silences],
            "keep_intervals": keep_intervals,            # silence-only
            "transcript": transcript_segments,           # text + timestamps
            "filler_cuts": filler_cuts,                  # what we removed (filler)
            "repetition_cuts": repetition_cuts,          # what we removed (repetition)
            "final_keep_intervals": final_keep,          # silence minus all cuts
            "budget_keep_intervals": smart_keep,
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
        target_pct: float = 1.0,
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
            "target_pct": target_pct,
        },
        "duration": None,
        "silences": None,
        "keep_intervals": None,
        "error": None,
        "transcript": None,
        "filler_cuts": None,
        "repetition_cuts": None,
        "final_keep_intervals": None,
        "budget_keep_intervals": None,
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
        "final_keep_intervals": job.get("final_keep_intervals"),
        "filler_cuts": job.get("filler_cuts"),
        "repetition_cuts": job.get("repetition_cuts"),
        "transcript": job.get("transcript"),
        "budget_keep_intervals": job.get("budget_keep_intervals"),
    }
