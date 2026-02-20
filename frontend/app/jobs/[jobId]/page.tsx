"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

type JobStatus = {
  job_id: string;
  status: string;
  error?: string | null;
};

type Cuts = {
  duration: number | null;
  keep_intervals: { start: number; end: number }[] | null;
  silences: { start: number; end: number }[] | null;
  status: string;
  error?: string | null;
};

export default function JobPage() {
  const params = useParams(); // { jobId: "..." }
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

  if (!jobId) {
    return <div className="p-8 text-white">Loading route…</div>;
  }

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

          {job.status === "error" && (
            <p className="text-red-400">Error: {job.error}</p>
          )}
        </>
      )}

      {cuts?.keep_intervals && (
        <div className="mt-6">
          <h2 className="text-xl font-semibold mb-3">
            Keep Intervals ({cuts.keep_intervals.length})
          </h2>

          <div className="bg-zinc-900 p-4 rounded font-mono text-sm space-y-1">
            {cuts.keep_intervals.map((k, idx) => (
              <div key={idx}>
                {idx + 1}. {k.start}s → {k.end}s
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
