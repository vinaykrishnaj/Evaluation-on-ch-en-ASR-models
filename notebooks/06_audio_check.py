# Notebook 6: sanity-check all TTS clips (duration, loudness, silence, speech rate) and flag defective ones.
# CPU only. Attach zh-tts-audio as input. Paste each "# %%" block into its own cell.

# %% Cell 1: index the clips
import os, re, json, glob, zipfile
import numpy as np
import pandas as pd
import soundfile as sf

MANI = glob.glob("/kaggle/input/**/manifest_tts.jsonl", recursive=True)[0]
IN_DIR = os.path.dirname(MANI)
rows = [json.loads(l) for l in open(MANI, encoding="utf-8")]
index = {os.path.basename(p): p for p in glob.glob(f"{IN_DIR}/**/*.wav", recursive=True)}
if not index:    # Kaggle normally unzips uploads; fall back to the zip
    zipfile.ZipFile(f"{IN_DIR}/tts_audio.zip").extractall("/tmp/tts_wav")
    index = {os.path.basename(p): p for p in glob.glob("/tmp/tts_wav/**/*.wav", recursive=True)}
print(len(rows), "rows;", len(index), "wav files found")

# %% Cell 2: per-clip measurements
FRAME = 0.025          # 25 ms analysis frames
SIL_DB = -50.0         # a frame below this level counts as silence

def frame_db(x, sr):
    n = int(FRAME * sr)
    m = len(x) // n
    f = x[:m * n].reshape(m, n)
    return 20 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-9)

def longest_run(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best

out = []
for r in rows:
    x, sr = sf.read(index[os.path.basename(r["tts_audio"])], dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    db = frame_db(x, sr)
    sil = db < SIL_DB
    voiced = np.where(~sil)[0]
    lead = voiced[0] * FRAME if len(voiced) else len(x) / sr
    trail = (len(db) - 1 - voiced[-1]) * FRAME if len(voiced) else len(x) / sr
    dur = len(x) / sr
    chars = len(re.findall(r"[一-鿿A-Za-z0-9]", r["zh_ref"]))
    speech = max(dur - lead - trail, 1e-3)
    out.append({"id": r["id"], "source": r["source"], "voice": r["tts_voice"], "sr": sr,
                "dur_s": round(dur, 2), "chars": chars, "chars_per_s": round(chars / speech, 2),
                "rms_db": round(float(20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)), 1),
                "peak": round(float(np.abs(x).max()), 3),
                "silent_frac": round(float(sil.mean()), 3), "lead_s": round(lead, 2), "trail_s": round(trail, 2),
                "longest_gap_s": round(longest_run(sil[voiced[0]:voiced[-1] + 1]) * FRAME if len(voiced) else 0, 2)})
df = pd.DataFrame(out)
print(df.describe().round(2).to_string())

# %% Cell 3: flag defective clips
med = df.groupby("voice")["chars_per_s"].transform("median")
mad = df.groupby("voice")["chars_per_s"].transform(lambda s: (s - s.median()).abs().median()) + 1e-6
df["rate_z"] = ((df["chars_per_s"] - med) / (1.4826 * mad)).round(1)

flags = {
    "longer than 30 s (Whisper window)": df.dur_s > 30,
    "shorter than 0.7 s": df.dur_s < 0.7,
    "very quiet (rms < -40 dB)": df.rms_db < -40,
    "clipping (peak >= 0.99)": df.peak >= 0.99,
    "mostly silent (> 60% frames)": df.silent_frac > 0.6,
    "long internal gap (> 2 s)": df.longest_gap_s > 2.0,
    "speech rate outlier (|z| > 5)": df.rate_z.abs() > 5,
}
df["problems"] = ""      # not "flags": DataFrame.flags is a built-in pandas attribute
for name, m in flags.items():
    df.loc[m, "problems"] += name + "; "
    print(f"{name}: {int(m.sum())}")
bad = df[df["problems"] != ""].sort_values("dur_s", ascending=False)
print("\nflagged clips:", len(bad))
print(bad[["id", "dur_s", "chars", "chars_per_s", "rate_z", "rms_db", "silent_frac", "longest_gap_s", "problems"]].head(40).to_string(index=False))
print("\nfleurs_00121:")
print(df[df.id == "fleurs_00121"].T.to_string())
print("\nlongest 10 clips:")
print(df.nlargest(10, "dur_s")[["id", "dur_s", "chars", "chars_per_s"]].to_string(index=False))

# %% Cell 4: save
df.to_csv("/kaggle/working/audio_check.csv", index=False)
bad.to_csv("/kaggle/working/audio_flagged.csv", index=False)
print(os.listdir("/kaggle/working"))
