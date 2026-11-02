# Notebook 1: build the <audio, zh transcript, en translation> manifest.
# Paste each "# %%" block into its own Kaggle cell.

# %% Cell 1: login + config
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(UserSecretsClient().get_secret("HF_TOKEN"))

import os, json, random
import soundfile as sf
import numpy as np
from datasets import load_dataset

N = 500            # pilot size per source; raise to 2000+ for the full run
SEED = 42
OUT = "/kaggle/working/data"
os.makedirs(f"{OUT}/covost2", exist_ok=True)
os.makedirs(f"{OUT}/fleurs", exist_ok=True)
manifest = []

def save_wav(audio, path):
    arr = np.asarray(audio["array"], dtype="float32")
    sf.write(path, arr, audio["sampling_rate"])

# %% Cell 2: CoVoST 2 zh->en (primary)
ds = load_dataset("fixie-ai/covost2", "zh-CN_en", split="test", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=2000)
count = 0
for ex in ds:
    zh, en = (ex["sentence"] or "").strip(), (ex["translation"] or "").strip()
    if not zh or not en:
        continue
    uid = f"covost2_{count:05d}"
    path = f"{OUT}/covost2/{uid}.wav"
    save_wav(ex["audio"], path)
    manifest.append({"id": uid, "source": "covost2", "audio_path": path,
                     "zh_ref": zh, "en_ref": en})
    count += 1
    if count >= N:
        break
print("covost2:", count)

# %% Cell 3: FLEURS zh test, English reference joined on sentence id (secondary)
zh_ds = load_dataset("google/fleurs", "cmn_hans_cn", split="test", streaming=True)
en_ds = load_dataset("google/fleurs", "en_us", split="test", streaming=True)

# one English transcription per id (FLEURS can repeat an id across speakers)
en_by_id = {}
for ex in en_ds:
    en_by_id.setdefault(ex["id"], ex["raw_transcription"].strip())
print("english sentences:", len(en_by_id))

count, seen = 0, set()
for ex in zh_ds:
    if ex["id"] in seen or ex["id"] not in en_by_id:
        continue
    seen.add(ex["id"])
    uid = f"fleurs_{count:05d}"
    path = f"{OUT}/fleurs/{uid}.wav"
    save_wav(ex["audio"], path)
    manifest.append({"id": uid, "source": "fleurs", "audio_path": path,
                     "zh_ref": ex["raw_transcription"].strip(),
                     "en_ref": en_by_id[ex["id"]]})
    count += 1
    if count >= N:
        break
print("fleurs:", count)

# %% Cell 4: write manifest + sanity check
with open(f"{OUT}/manifest.jsonl", "w", encoding="utf-8") as f:
    for r in manifest:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("total rows:", len(manifest))
for r in random.Random(SEED).sample(manifest, 5):
    print(r["id"], "|", r["zh_ref"], "|", r["en_ref"])

# %% Cell 5: package output as manifest.jsonl + audio.zip (Kaggle lists at most 500 output files)
import shutil
rel = [dict(r, audio_path=os.path.relpath(r["audio_path"], OUT)) for r in manifest]
open("/kaggle/working/manifest.jsonl", "w", encoding="utf-8").write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rel))
os.remove(f"{OUT}/manifest.jsonl")
shutil.make_archive("/kaggle/working/audio", "zip", OUT)
shutil.rmtree(OUT)
print(os.listdir("/kaggle/working"))
