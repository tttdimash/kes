"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function UploadPage() {
    // Configuration parameters for silence detection 
    const [noise,  setNoise] = useState(-30);
    const [minSilence, setMinSilence] = useState(0.5);
    const [pad, setPad] = useState(0.12);

    const router = useRouter();
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);

    const handleUpload = async () => {
        if (!file || uploading) return;
        setUploading(true);

        const formData = new FormData();
        formData.append("video", file);

        try {
        // 1) upload
        const upRes = await fetch("http://localhost:8000/upload", {
            method: "POST",
            body: formData,
        });

        if (!upRes.ok) {
            const err = await upRes.json().catch(() => ({}));
            throw new Error(err.detail || "Upload failed");
        }

        const upData: { file_id: string } = await upRes.json();
        if (!upData.file_id) throw new Error("Upload response missing file_id");

        // 2) create job with parameters
        const jobRes = await fetch(
            `http://localhost:8000/jobs?file_id=${encodeURIComponent(upData.file_id)}&noise_db=${noise}&min_silence=${minSilence}&pad=${pad}`,
            { method: "POST" }
        );

        if (!jobRes.ok) {
            const err = await jobRes.json().catch(() => ({}));
            throw new Error(err.detail || "Job creation failed");
        }

        const jobData: { job_id: string } = await jobRes.json();
        if (!jobData.job_id) throw new Error("Job response missing job_id");

        // 3) go to job status page
        router.push(`/jobs/${jobData.job_id}`);
        } catch (err: any) {
        console.error(err);
        alert(err?.message || "Upload failed");
        } finally {
        setUploading(false);
        }
    };

  return (
    <div className="min-h-screen bg-black text-white p-10">
      <h1 className="text-3xl font-bold mb-8">Upload Video</h1>

      {/* File input section */}
      <div className="mb-8 w-full max-w-md">
        <label className="block mb-2 font-semibold">Select Video File:</label>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          className="w-full px-3 py-2 bg-zinc-800 rounded border border-zinc-600"
        />
        {file && <p className="text-sm text-green-400 mt-2">Selected: {file.name}</p>}
      </div>

      {/* Controls (only show if file is selected) */}
      {file && (
        <>
          {/* Control 1: Silence sensitivity (noise_db) */}
      <div className="mb-6 w-full max-w-md">
        <label className="block mb-2 font-semibold">
          Silence sensitivity (noise_db): <span className="font-mono">{noise} dB</span>
        </label>

        <input
          type="range"
          min={-40}
          max={-20}
          step={1}
          value={noise}
          onChange={(e) => setNoise(Number(e.target.value))}
          className="w-full"
        />
        <p className="text-sm text-zinc-400 mt-1">
          More negative = stricter silence (quieter required).
        </p>
      </div>

      {/* Control 2: Minimum pause length (min_silence) */}
      <div className="mb-6 w-full max-w-md">
        <label className="block mb-2 font-semibold">
          Minimum pause length: <span className="font-mono">{minSilence.toFixed(2)} s</span>
        </label>

        <input
          type="range"
          min={0.2}
          max={1.5}
          step={0.05}
          value={minSilence}
          onChange={(e) => setMinSilence(Number(e.target.value))}
          className="w-full"
        />
        <p className="text-sm text-zinc-400 mt-1">
          Smaller = cuts short pauses too.
        </p>
      </div>

      {/* Control 3: Padding around cuts (pad) */}
      <div className="mb-6 w-full max-w-md">
        <label className="block mb-2 font-semibold">
          Padding (pad): <span className="font-mono">{pad.toFixed(2)} s</span>
        </label>

        <input
          type="range"
          min={0.0}
          max={0.3}
          step={0.01}
          value={pad}
          onChange={(e) => setPad(Number(e.target.value))}
          className="w-full"
        />
        <p className="text-sm text-zinc-400 mt-1">
          Adds safety buffer so cuts don’t clip words.
        </p>
      </div>

      {/* Current params summary */}
          <div className="mt-8 mb-8 p-4 bg-zinc-800 rounded w-full max-w-md text-sm text-zinc-300">
            <p className="font-semibold mb-2">Parameters to send:</p>
            <div className="font-mono space-y-1">
              <p>noise_db: {noise} dB</p>
              <p>min_silence: {minSilence.toFixed(2)} s</p>
              <p>pad: {pad.toFixed(2)} s</p>
            </div>
          </div>

          {/* Upload button */}
          <button
            onClick={handleUpload}
            disabled={uploading}
            className="bg-blue-600 px-6 py-2 rounded hover:bg-blue-700 disabled:opacity-50 font-semibold"
          >
            {uploading ? "Uploading..." : "Upload & Process"}
          </button>
        </>
      )}
    </div>
  );
}
