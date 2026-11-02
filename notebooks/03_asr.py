# Notebook 3: Chinese ASR on the TTS audio (Whisper large-v3 + Paraformer-zh) and CER.
# Needs: GPU accelerator, Internet on, zh-tts-audio attached as input.
# Paste each "# %%" block into its own cell.

# %% Cell 1: install
%pip install -q funasr modelscope opencc-python-reimplemented jiwer

# %% Cell 2: load manifest and unpack the TTS audio (to /tmp, not the output folder)
import os, json, glob, zipfile, re, time
import numpy as np
import soundfile as sf
import torch

MANI = glob.glob("/kaggle/input/**/manifest_tts.jsonl", recursive=True)[0]
IN_DIR = os.path.dirname(MANI)
rows = [json.loads(l) for l in open(MANI, encoding="utf-8")]
# Kaggle usually unzips uploaded archives, so look for the wavs first and only unzip if none are there
index = {os.path.basename(p): p for p in glob.glob(f"{IN_DIR}/**/*.wav", recursive=True)}
if not index:
    zipfile.ZipFile(f"{IN_DIR}/tts_audio.zip").extractall("/tmp/tts_wav")
    index = {os.path.basename(p): p for p in glob.glob("/tmp/tts_wav/**/*.wav", recursive=True)}
paths = [index[os.path.basename(r["tts_audio"])] for r in rows]
print(len(rows), "rows;", len(paths), "audio files found; gpu:", torch.cuda.is_available())

# %% Cell 3: Whisper large-v3
import gc
from transformers import WhisperProcessor, WhisperForConditionalGeneration
proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
wm = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3", dtype=torch.float16).to("cuda").eval()

SR, WIN = 16000, 30 * 16000
audio = [sf.read(p, dtype="float32")[0] for p in paths]
order = sorted(range(len(rows)), key=lambda i: len(audio[i]))     # similar lengths share a batch
short = [i for i in order if len(audio[i]) <= WIN]
long_ = [i for i in order if len(audio[i]) > WIN]                  # over 30 s: decoded one at a time, long-form
print(len(short), "short clips,", len(long_), "clips over 30 s")

def transcribe(idx, long_form=False):
    arrs = [audio[i] for i in idx]
    kw = dict(language="zh", task="transcribe", max_new_tokens=200)
    if long_form:
        inp = proc(arrs, sampling_rate=SR, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True)
        kw.update(attention_mask=inp.attention_mask.to("cuda"), return_timestamps=True, condition_on_prev_tokens=False)
    else:
        inp = proc(arrs, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        ids = wm.generate(inp.input_features.to("cuda", torch.float16), **kw)
    for i, t in zip(idx, proc.batch_decode(ids, skip_special_tokens=True)):
        rows[i]["zh_asr_whisper"] = t.strip()

t0, done = time.time(), 0
for b in range(0, len(short), 16):
    transcribe(short[b:b + 16])
    done += len(short[b:b + 16])
    if (b // 16) % 5 == 0:
        print(f"{done}/{len(rows)} clips, {round(time.time() - t0)} s", flush=True)
for i in long_:
    transcribe([i], long_form=True)
    done += 1
    print(f"{done}/{len(rows)} clips (long), {round(time.time() - t0)} s", flush=True)
print("whisper done in", round(time.time() - t0), "s")
print(rows[0]["zh_ref"], "|", rows[0]["zh_asr_whisper"])
del wm, audio
gc.collect(); torch.cuda.empty_cache()

# %% Cell 4: Paraformer-zh (FunASR)
from funasr import AutoModel
pf = AutoModel(model="paraformer-zh", device="cuda:0", disable_update=True)
t0 = time.time()
for i in range(0, len(rows), 32):
    res = pf.generate(input=paths[i:i + 32], batch_size_s=300)
    for r, o in zip(rows[i:i + 32], res):
        r["zh_asr_paraformer"] = o["text"].replace(" ", "").strip()
print("paraformer done in", round(time.time() - t0), "s")
print(rows[0]["zh_ref"], "|", rows[0]["zh_asr_paraformer"])

# %% Cell 5: CER (script and punctuation normalised first)
import jiwer
from opencc import OpenCC
t2s = OpenCC("t2s")

def norm(s):
    s = t2s.convert(s)
    return "".join(re.findall(r"[一-鿿A-Za-z0-9]", s)).lower()

def cer(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    return jiwer.cer(ref, hyp) if ref else None

for m in ("whisper", "paraformer"):
    for r in rows:
        r[f"cer_{m}"] = cer(r["zh_ref"], r[f"zh_asr_{m}"])
    for src in ("covost2", "fleurs", None):
        sel = [r for r in rows if src is None or r["source"] == src]
        pairs = [(norm(r["zh_ref"]), norm(r[f"zh_asr_{m}"])) for r in sel if norm(r["zh_ref"])]
        c = jiwer.cer([a for a, _ in pairs], [b for _, b in pairs])
        print(f"{m:11s} {src or 'all':8s} corpus CER = {c:.4f}  (n={len(pairs)})")

# %% Cell 6: save results
open("/kaggle/working/manifest_asr.jsonl", "w", encoding="utf-8").write(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
print(os.listdir("/kaggle/working"))
