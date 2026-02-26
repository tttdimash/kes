from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import uuid
import shutil
import subprocess
import re
import math
from typing import List, Tuple, Dict, Any
from faster_whisper import WhisperModel
import tempfile
from sentence_transformers import SentenceTransformer
import numpy as np
from fastapi import HTTPException

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
    Transcribe a 16kHz mono WAV file using the global Whisper model.

    Returns:
        transcript_segments:
            List of segment-level speech blocks.
            Each segment contains:
                - start (float): time in seconds where speech begins
                - end (float): time in seconds where speech ends
                - text (str): recognized speech text

        words:
            Flat list of word-level timestamps.
            Each entry contains:
                - start (float): word start time in seconds
                - end (float): word end time in seconds
                - word (str): spoken word text

    These timestamps are later used for:
        - filler word removal
        - semantic chunking
        - precise video cutting
    """

    # Run Whisper transcription with:
    # - word-level timestamps enabled
    # - voice activity detection to skip long silence
    segments, _info = WHISPER_MODEL.transcribe(
        wav_path,
        word_timestamps=True,
        vad_filter=True,
    )

    transcript_segments = []
    words = []

    # Iterate over each recognized speech segment
    for seg in segments:
        # Store segment-level timing and text
        transcript_segments.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip()
        })

        # If word-level timestamps are available, collect them
        if seg.words:
            for w in seg.words:
                word_text = (w.word or "").strip()
                if word_text:
                    words.append({
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": word_text
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
    return merged

def subtract_cut_intervals(keep_intervals, cut_intervals, min_keep: float = 0.20):
    """
    Subtract cut intervals from keep intervals.

    Args:
        keep_intervals:
            List of time segments currently marked to be kept.
            Format: [{"start": float, "end": float}, ...]

        cut_intervals:
            List of time segments that should be removed.
            Format: [{"start": float, "end": float, ...}, ...]

        min_keep:
            Minimum duration (in seconds) required for a segment
            to remain in the final result. Very small fragments
            below this threshold are discarded.

    Returns:
        List of refined keep intervals after subtracting cuts.
    """

    # If there is nothing to keep, return empty immediately.
    if not keep_intervals:
        return []

    # Convert cut dicts to simple (start, end) tuples
    # and discard invalid intervals where end <= start.
    cuts = [
        (c["start"], c["end"])
        for c in cut_intervals
        if c["end"] > c["start"]
    ]

    # Sort cuts by start time so we can sweep forward chronologically.
    cuts.sort()

    refined = []

    # Process each keep interval independently.
    for k in keep_intervals:
        ks, ke = k["start"], k["end"]

        # `cur` tracks our current position inside the keep interval.
        cur = ks

        # Compare this keep interval against each cut interval.
        for cs, ce in cuts:

            # If the cut ends before our current pointer,
            # it has no effect — skip it.
            if ce <= cur:
                continue

            # If the cut starts after the keep interval ends,
            # no further overlap is possible — stop checking.
            if cs >= ke:
                break

            # If there is a gap between `cur` and the start of this cut,
            # and it is large enough to keep, preserve that portion.
            if cs > cur and (cs - cur) >= min_keep:
                refined.append({
                    "start": round(cur, 3),
                    "end": round(cs, 3)
                })

            # Move the current pointer forward past the cut.
            cur = max(cur, ce)

        # After processing all relevant cuts,
        # check if there is remaining tail segment to keep.
        if (ke - cur) >= min_keep:
            refined.append({
                "start": round(cur, 3),
                "end": round(ke, 3)
            })

    return refined


# ---------- FFmpeg helpers ----------

def _sec(s: float) -> str:
    """Format seconds as an ffmpeg-friendly timestamp string."""
    return f"{s:.3f}"

def render_video_from_intervals(input_path: str, intervals, output_path: str):
    """
    Render an edited video by keeping only the provided time intervals.

    Strategy:
      1) Cut each keep-interval into a temporary clip (re-encode for accuracy).
      2) Concatenate the clips into a single MP4 using ffmpeg's concat demuxer.

    Args:
        input_path: Path to the original video.
        intervals: List of {"start": float, "end": float} keep segments (seconds).
        output_path: Path to write the final MP4.
    """
    if not intervals:
        raise ValueError("No intervals to render")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Temporary directory is automatically deleted (clips + concat file).
    with tempfile.TemporaryDirectory() as td:
        clip_paths = []

        for idx, it in enumerate(intervals):
            start = float(it["start"])
            end = float(it["end"])
            dur = end - start

            # Skip tiny fragments that are likely noise and can break ffmpeg.
            if dur <= 0.05:
                continue

            clip_path = os.path.join(td, f"clip_{idx:04d}.mp4")
            clip_paths.append(clip_path)

            # Cut this interval out of the source video.
            # We re-encode to avoid keyframe/accuracy issues from stream-copy cutting.
            cmd = [
                "ffmpeg", "-y",
                "-ss", _sec(start),
                "-to", _sec(end),
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                clip_path,
            ]
            subprocess.check_call(cmd)

        if not clip_paths:
            raise ValueError("All intervals were too short to render")

        # ffmpeg concat demuxer expects a text file with: file 'path'
        list_path = os.path.join(td, "concat.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        # Concatenate without re-encoding (fast) because clips share codecs/settings.
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ]
        subprocess.check_call(cmd)

def get_duration(video_path: str) -> float:
    """
    Return total video duration (in seconds) using ffprobe.

    Uses ffprobe to read container metadata without decoding video.
    Raises CalledProcessError if ffprobe fails.
    """

    cmd = [
        "ffprobe", "-v", "error",                 # suppress non-error logs
        "-show_entries", "format=duration",       # request only duration field
        "-of", "default=noprint_wrappers=1:nokey=1",  # output raw number only
        video_path
    ]

    # Execute ffprobe and capture stdout as text
    out = subprocess.check_output(cmd, text=True).strip()

    # Convert string output (e.g. "83.427000") to float
    return float(out)

def run_silencedetect(
    video_path: str,
    noise_db: str = "-30dB",
    min_silence: float = 0.5
) -> List[Tuple[float, float]]:
    """
    Detect silent regions in a video using ffmpeg's silencedetect filter.

    Args:
        video_path: Path to input video.
        noise_db: Audio threshold below which signal is treated as silence.
        min_silence: Minimum silence duration (seconds) to count.

    Returns:
        List of (silence_start, silence_end) time tuples in seconds.
    """

    # Run ffmpeg with silencedetect filter.
    # Output is discarded; we only parse stderr logs.
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-af", f"silencedetect=noise={noise_db}:d={min_silence}",
        "-f", "null",
        "-"
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stderr = proc.stderr

    silence_starts = []
    silence_ends = []

    # Parse ffmpeg log lines to extract silence timestamps.
    for line in stderr.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*([0-9.]+)", line)
            if m:
                silence_starts.append(float(m.group(1)))

        elif "silence_end:" in line:
            m = re.search(r"silence_end:\s*([0-9.]+)", line)
            if m:
                silence_ends.append(float(m.group(1)))

    # Pair starts and ends safely (handle incomplete matches).
    n = min(len(silence_starts), len(silence_ends))
    return [(silence_starts[i], silence_ends[i]) for i in range(n)]


def silences_to_keep_intervals(
    duration: float,
    silences: List[Tuple[float, float]],
    pad: float = 0.12,
    min_keep: float = 0.25
) -> List[Dict[str, float]]:
    """
    Convert silent regions into keep intervals (non-silent regions).

    Steps:
      1) Expand each silence interval by `pad` seconds on both sides.
      2) Merge overlapping expanded silences.
      3) Keep the gaps between the merged silence regions.
      4) Drop tiny keep fragments smaller than `min_keep`.
    """
    if duration <= 0:
        return []

    # Expand silences into "cut intervals" with padding and clamp to [0, duration].
    cuts: List[Tuple[float, float]] = []
    for s, e in silences:
        s2 = max(0.0, s - pad)
        e2 = min(duration, e + pad)
        if e2 > s2:
            cuts.append((s2, e2))

    # Merge overlapping/adjacent cut intervals.
    cuts.sort()
    merged: List[List[float]] = []
    for s, e in cuts:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # Keep intervals are the gaps between merged cuts.
    keep: List[Dict[str, float]] = []
    cur = 0.0
    for s, e in merged:
        if (s - cur) >= min_keep:
            keep.append({"start": round(cur, 3), "end": round(s, 3)})
        cur = max(cur, e)

    # Keep any remaining tail after the last cut.
    if (duration - cur) >= min_keep:
        keep.append({"start": round(cur, 3), "end": round(duration, 3)})

    return keep

def detect_repetition_cuts(
    transcript_segments,
    threshold: float = 0.88,
    min_chars: int = 30,
    pad: float = 0.08,
    lookback: int = 12,
):
    """
    Identify transcript segments that repeat earlier content.

    For each segment, compare its embedding against a rolling window of
    previously kept segments. If the best similarity exceeds `threshold`,
    mark the current segment as redundant and return its time span as a cut.
    """
    texts = [s["text"].strip() for s in transcript_segments]
    embeddings = EMBED_MODEL.encode(texts, normalize_embeddings=True)

    keep_idx = []
    cut_intervals = []

    for i, seg in enumerate(transcript_segments):
        text = seg["text"].strip()

        # Skip very short segments to reduce false positives.
        if len(text) < min_chars:
            keep_idx.append(i)
            continue

        # Only compare against recent "kept" segments for speed and locality.
        candidates = keep_idx[-lookback:] if lookback > 0 else keep_idx

        best_score = -1.0
        best_j = None

        for j in candidates:
            # embeddings are normalized => dot product equals cosine similarity
            score = float(np.dot(embeddings[i], embeddings[j]))
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None and best_score >= threshold:
            cut_intervals.append({
                "start": max(0.0, float(seg["start"]) - pad),
                "end": float(seg["end"]) + pad,
                "label": "repetition",
                "score": round(best_score, 3),
                "match_index": best_j,
            })
        else:
            keep_idx.append(i)

    # Merge overlapping cut intervals (due to padding).
    cut_intervals.sort(key=lambda x: x["start"])
    merged = []
    for c in cut_intervals:
        if not merged or c["start"] > merged[-1]["end"]:
            merged.append(c)
        else:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
            merged[-1]["score"] = max(merged[-1].get("score", 0), c.get("score", 0))

    return merged

def make_time_chunks_from_words(words, chunk_seconds=3.0, hop_seconds=1.5, min_words=5):
    """
    Build overlapping, time-based transcript chunks from word-level timestamps.

    - Each chunk starts at a word start time and spans `chunk_seconds`.
    - We advance the start time by `hop_seconds` (sliding window).
    - Only chunks with at least `min_words` are emitted.

    Returns:
        List of {"start": float, "end": float, "text": str}
    """
    chunks = []
    if not words:
        return chunks

    i = 0
    while i < len(words):
        start_t = float(words[i]["start"])
        end_t = start_t + chunk_seconds

        # Find the largest j such that words[i:j] end within the window.
        j = i
        while j < len(words) and float(words[j]["end"]) <= end_t:
            j += 1

        # Only keep chunks with enough words to be meaningful.
        if (j - i) >= min_words:
            text = " ".join(w["word"].strip() for w in words[i:j]).strip()
            chunks.append({
                "start": float(words[i]["start"]),
                "end": float(words[j - 1]["end"]),
                "text": text
            })

        # Advance i to the first word starting at or after (start_t + hop_seconds).
        next_i = i + 1
        while next_i < len(words) and float(words[next_i]["start"]) < (start_t + hop_seconds):
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
            "duration": duration,
            "silences": [{"start": s, "end": e} for (s, e) in silences],
            "keep_intervals": keep_intervals,            # silence-only
            "transcript": transcript_segments,           # text + timestamps
            "filler_cuts": filler_cuts,                  # what we removed (filler)
            "repetition_cuts": repetition_cuts,          # what we removed (repetition)
            "final_keep_intervals": final_keep,          # silence minus all cuts
            "budget_keep_intervals": smart_keep,
        })

        render_intervals = (
            JOBS[job_id].get("budget_keep_intervals")
            or JOBS[job_id].get("final_keep_intervals")
            or keep_intervals
        )

        out_path = os.path.join("outputs", f"{job_id}.mp4")

        render_video_from_intervals(
            in_path,               # original video path
            render_intervals,      # chosen keep plan
            out_path
        )

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["output_path"] = out_path


    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)


# ---------- API endpoints ----------

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """Save an uploaded video to disk and return a generated file_id."""
    file_id = str(uuid.uuid4())

    # Preserve the original extension if present; default to .mp4
    ext = os.path.splitext(video.filename)[1] or ".mp4"

    # Example: uploads/<uuid>.mp4
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

    # Stream upload bytes to disk
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
        "output_path": None,
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

@app.get("/jobs/{job_id}/download")
def download_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail="job not done")

    out_path = job.get("output_path")
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="output not found")

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )