"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

type JobStatus = {
  job_id: string;
  status: string;
  error?: string | null;
};

type Interval = { start: number; end: number };
type TranscriptSeg = { start: number; end: number; text: string };
type FillerCut = { start: number; end: number; label?: string };

type Cuts = {
  duration: number | null;
  keep_intervals: Interval[] | null;
  final_keep_intervals?: Interval[] | null;
  budget_keep_intervals?: { start: number; end: number }[] | null;
  silences: Interval[] | null;

  transcript?: TranscriptSeg[] | null;     // <-- add
  filler_cuts?: FillerCut[] | null;        // <-- add
  repetition_cuts?: FillerCut[] | null;    // <-- add

  status: string;
  error?: string | null;
};

function sumCutDurations(cuts: FillerCut[] | null | undefined) {
  if (!cuts) return 0;
  return cuts.reduce((acc, c) => acc + Math.max(0, (c.end ?? 0) - (c.start ?? 0)), 0);
}

function sumIntervals(intervals: Interval[] | null | undefined) {
  if (!intervals) return 0;
  return intervals.reduce((acc, it) => acc + Math.max(0, it.end - it.start), 0);
}

function fmtSeconds(s: number) {
  if (!isFinite(s)) return "-";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

export default function JobPage() {
  const params = useParams();
  const jobId = params?.jobId as string;

  const [job, setJob] = useState<JobStatus | null>(null);
  const [cuts, setCuts] = useState<Cuts | null>(null);

  useEffect(() => {
    if (!jobId) return;

    let timer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(`http://localhost:8000/jobs/${jobId}`);
        const data = await res.json();

        if (cancelled) return;
        setJob(data);

        if (data.status === "done") {
          const cutsRes = await fetch(`http://localhost:8000/jobs/${jobId}/cuts`);
          const cutsData = await cutsRes.json();
          if (!cancelled) setCuts(cutsData);
          return;
        }

        if (data.status !== "error") {
          timer = setTimeout(poll, 1000);
        }
      } catch (e: any) {
        if (!cancelled) setJob({ job_id: jobId, status: "error", error: e?.message || "fetch failed" });
      }
    };

    poll();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  const intervals = useMemo(() => {
    if (!cuts) return null;
    // Prefer budgeted intervals (time-budgeted) if available, otherwise final_keep, otherwise silence-based keep
    if (cuts.budget_keep_intervals && Array.isArray(cuts.budget_keep_intervals) && cuts.budget_keep_intervals.length) return cuts.budget_keep_intervals;
    if (cuts.final_keep_intervals && Array.isArray(cuts.final_keep_intervals) && cuts.final_keep_intervals.length) return cuts.final_keep_intervals;
    return cuts.keep_intervals;
  }, [cuts]);

  const summary = useMemo(() => {
    const total = cuts?.duration ?? 0;
    const kept = sumIntervals(intervals);
    const removed = Math.max(0, total - kept);
    const keptPct = total > 0 ? (kept / total) * 100 : 0;
    const removedPct = total > 0 ? (removed / total) * 100 : 0;
    const fillerCount = cuts?.filler_cuts?.length ?? 0;
    const fillerRemovedSec = sumCutDurations(cuts?.filler_cuts);
    const repCount = cuts?.repetition_cuts?.length ?? 0;
    const repRemovedSec = sumCutDurations(cuts?.repetition_cuts);

    // Count filler cuts by label
    const fillerByLabel: { [key: string]: number } = {};
    if (cuts?.filler_cuts) {
      cuts.filler_cuts.forEach((cut) => {
        const label = cut.label || "unknown";
        fillerByLabel[label] = (fillerByLabel[label] || 0) + 1;
      });
    }

    return {
      total,
      kept,
      removed,
      keptPct,
      removedPct,
      segments: intervals?.length ?? 0,
      fillerCount,
      fillerRemovedSec,
      fillerByLabel,
      repCount,
      repRemovedSec,
    };
  }, [cuts, intervals]);

  // Determine which transcript segments were removed and why
  const removedSegments = useMemo(() => {
    if (!cuts?.transcript) return [];

    const allCuts = [
      ...(cuts.filler_cuts?.map(c => ({ ...c, reason: "filler", label: c.label || "filler" })) || []),
      ...(cuts.repetition_cuts?.map(c => ({ ...c, reason: "repetition", label: "repetition" })) || []),
      ...(cuts.silences?.map(c => ({ ...c, reason: "silence", label: "silence" })) || []),
    ];

    return cuts.transcript
      .filter(seg => {
        // Check if segment overlaps with any cut
        return allCuts.some(cut => seg.start < cut.end && seg.end > cut.start);
      })
      .map(seg => {
        // Find which cut(s) removed this segment
        const reasons = allCuts
          .filter(cut => seg.start < cut.end && seg.end > cut.start)
          .map(cut => cut.reason === "filler" ? `filler (${cut.label})` : cut.reason);
        return { ...seg, reasons: [...new Set(reasons)] };
      })
      .sort((a, b) => a.start - b.start);
  }, [cuts, cuts?.transcript]);

  if (!jobId) return <div className="p-8 text-white">Loading route…</div>;

  return (
    <div className="min-h-screen bg-black text-white p-10">
      <h1 className="text-3xl font-bold mb-6">KES Cut Plan</h1>

      {!job ? (
        <p>Loading job…</p>
      ) : (
        <>
          <p className="mb-4">
            Status: <span className="font-mono">{job.status}</span>
          </p>

          {job.status === "error" && <p className="text-red-400">Error: {job.error}</p>}
        </>
      )}

      {cuts && (
        <div className="w-full max-w-2xl bg-zinc-900 p-4 rounded mb-6">
          <h2 className="text-xl font-semibold mb-3">Summary</h2>

          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="text-zinc-400">Total</div>
              <div className="font-mono">{fmtSeconds(summary.total)}</div>
            </div>

            <div>
              <div className="text-zinc-400">Kept</div>
              <div className="font-mono">
                {fmtSeconds(summary.kept)} ({summary.keptPct.toFixed(1)}%)
              </div>
            </div>

            <div>
              <div className="text-zinc-400">Removed</div>
              <div className="font-mono">
                {fmtSeconds(summary.removed)} ({summary.removedPct.toFixed(1)}%)
              </div>
            </div>

            <div>
              <div className="text-zinc-400">Keep segments</div>
              <div className="font-mono">{summary.segments}</div>
            </div>
          </div>

          <div className="mt-3 text-xs text-zinc-400">
            Showing:{" "}
            <span className="font-mono">
              {cuts?.budget_keep_intervals?.length ? "budget_keep_intervals" : cuts?.final_keep_intervals?.length ? "final_keep_intervals" : "keep_intervals"}
            </span>
          </div>
        </div>
      )}

      {intervals && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-3">Keep Intervals ({intervals.length})</h2>

          <div className="bg-zinc-900 p-4 rounded font-mono text-sm space-y-1">
            {intervals.map((k, idx) => (
              <div key={idx}>
                {idx + 1}. {k.start}s → {k.end}s
              </div>
            ))}
          </div>
        </div>
      )}

      {cuts && (
  <div className="w-full max-w-2xl bg-zinc-900 p-4 rounded mb-6">
    <h2 className="text-xl font-semibold mb-3">Filler removal</h2>

    <div className="grid grid-cols-2 gap-3 text-sm">
      <div>
        <div className="text-zinc-400">Filler cuts</div>
        <div className="font-mono">{summary.fillerCount}</div>
      </div>

      <div>
        <div className="text-zinc-400">Time removed (est.)</div>
        <div className="font-mono">{fmtSeconds(summary.fillerRemovedSec)}</div>
      </div>
    </div>

        {summary.fillerCount > 0 && (
        <div className="mt-3 text-xs text-zinc-300">
            <div className="text-zinc-400 mb-1">Counts by label</div>
            <div className="flex flex-wrap gap-2">
            {Object.entries(summary.fillerByLabel).map(([label, n]) => (
                <span key={label} className="bg-zinc-800 px-2 py-1 rounded font-mono">
                {label}: {n}
                </span>
            ))}
            </div>
        </div>
        )}
    </div>
    )}

    {cuts && summary.repCount > 0 && (
  <div className="w-full max-w-2xl bg-zinc-900 p-4 rounded mb-6">
    <h2 className="text-xl font-semibold mb-3">Repetition removal</h2>

    <div className="grid grid-cols-2 gap-3 text-sm">
      <div>
        <div className="text-zinc-400">Repetition cuts</div>
        <div className="font-mono">{summary.repCount}</div>
      </div>

      <div>
        <div className="text-zinc-400">Time removed (est.)</div>
        <div className="font-mono">{fmtSeconds(summary.repRemovedSec)}</div>
      </div>
    </div>
    </div>
    )}

    {removedSegments.length > 0 && (
  <div className="w-full max-w-3xl bg-zinc-900 p-4 rounded mb-6">
    <h2 className="text-xl font-semibold mb-3">Removed segments ({removedSegments.length})</h2>

    <div className="text-sm space-y-2 max-h-64 overflow-y-auto">
      {removedSegments.map((seg, idx) => (
        <div key={idx} className="border border-red-900 rounded p-3 bg-red-950 bg-opacity-30">
          <div className="flex justify-between items-start mb-2">
            <div className="text-xs text-zinc-400 font-mono">
              {seg.start.toFixed(2)}s → {seg.end.toFixed(2)}s
            </div>
            <div className="flex gap-1">
              {seg.reasons.map((r, i) => (
                <span
                  key={i}
                  className="text-xs px-2 py-1 rounded bg-red-900 text-red-200"
                >
                  {r}
                </span>
              ))}
            </div>
          </div>
          <div className="text-red-100">{seg.text}</div>
        </div>
      ))}
    </div>
  </div>
    )}

    {cuts?.transcript && (
  <div className="w-full max-w-2xl bg-zinc-900 p-4 rounded mb-6">
    <h2 className="text-xl font-semibold mb-3">Transcript</h2>

    <div className="text-sm space-y-2">
      {cuts.transcript.map((seg, idx) => (
        <div key={idx} className="border border-zinc-800 rounded p-3">
          <div className="text-xs text-zinc-400 font-mono mb-1">
            {seg.start.toFixed(2)}s → {seg.end.toFixed(2)}s
          </div>
          <div>{seg.text}</div>
        </div>
      ))}
    </div>
  </div>
)}
    </div>
  );
}