# Notebook 8: translate the same 848 sentences with MADLAD-400-3B-MT (replaces TowerInstruct at step 4).
# Needs: GPU T4 x2, Internet on, zh-asr-transcripts attached as input. No Hugging Face token needed (model is not gated).
# Paste each "# %%" block into its own Kaggle cell.

# %% Cell 1: load manifest and collect the unique Chinese texts (same as notebook 4)
import os, json, glob, time, gc
import torch

MODEL_ID = "google/madlad400-3b-mt"
MANI = glob.glob("/kaggle/input/**/manifest_asr.jsonl", recursive=True)[0]
rows = [json.loads(l) for l in open(MANI, encoding="utf-8")]
KEYS = {"ref": "zh_ref", "whisper": "zh_asr_whisper", "paraformer": "zh_asr_paraformer"}
texts = sorted({r[k].strip() for r in rows for k in KEYS.values() if r[k].strip()}, key=len)
print(len(rows), "rows;", len(texts), "unique Chinese texts; gpus:", torch.cuda.device_count())

# %% Cell 2: load MADLAD (T5 encoder-decoder, 3B). fp32 on purpose: T5 models can overflow in fp16 and T4s have no native bf16.
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained(MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float32).eval()
print("loaded; devices:", sorted({str(v) for v in getattr(model, "hf_device_map", {}).values()}))

# %% Cell 3: smoke test (the target language is chosen by the <2en> prefix)
def translate(batch):
    enc = tok(["<2en> " + t for t in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=200, num_beams=1, do_sample=False)
    return [s.strip() for s in tok.batch_decode(out, skip_special_tokens=True)]

for zh, en in zip(texts[-3:], translate(texts[-3:])):
    print(zh, "->", en)

# %% Cell 4: translate every unique text (sorted by length, progress printed)
BS = 32
cache = {}
t0 = time.time()
for b in range(0, len(texts), BS):
    batch = texts[b:b + BS]
    for zh, en in zip(batch, translate(batch)):
        cache[zh] = en
    if (b // BS) % 10 == 0:
        print(f"{len(cache)}/{len(texts)} texts, {round(time.time() - t0)} s", flush=True)
print("translation done in", round(time.time() - t0), "s")

# %% Cell 5: attach translations (same column names as notebook 4, so the scoring code is reused) and save
for r in rows:
    for name, k in KEYS.items():
        r[f"en_mt_{name}"] = cache.get(r[k].strip(), "")
open("/kaggle/working/manifest_mt_madlad.jsonl", "w", encoding="utf-8").write(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
for r in rows[:5]:
    print(r["en_ref"], "|", r["en_mt_ref"], "|", r["en_mt_whisper"], "|", r["en_mt_paraformer"])
print(os.listdir("/kaggle/working"))
