# Notebook 4: translate zh_ref and both ASR transcripts to English with Tower.
# Needs: GPU T4 x2, Internet on, zh-asr-transcripts attached as input.
# Paste each "# %%" block into its own cell.

# %% Cell 1: load manifest and collect the unique Chinese texts
import os, json, glob, time, gc
import torch

MODEL_ID = "Unbabel/TowerInstruct-7B-v0.2"   # "TowerChat" is not a model name; swap here to try another Tower model
MANI = glob.glob("/kaggle/input/**/manifest_asr.jsonl", recursive=True)[0]
rows = [json.loads(l) for l in open(MANI, encoding="utf-8")]
KEYS = {"ref": "zh_ref", "whisper": "zh_asr_whisper", "paraformer": "zh_asr_paraformer"}
# identical strings (ASR matched the reference) are translated once
texts = sorted({r[k].strip() for r in rows for k in KEYS.values() if r[k].strip()}, key=len)
print(len(rows), "rows;", len(texts), "unique Chinese texts to translate; gpus:", torch.cuda.device_count())

# %% Cell 2: load Tower in fp16, split across the two T4s
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto").eval()
print("loaded;", {str(k): v for k, v in model.hf_device_map.items()} if hasattr(model, "hf_device_map") else "")

# %% Cell 3: smoke test on 3 sentences
def build(t):
    msg = [{"role": "user", "content": f"Translate the following text from Chinese into English.\nChinese: {t}\nEnglish:"}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

def translate(batch):
    inp = tok([build(t) for t in batch], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
    return [s.strip() for s in tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)]

for zh, en in zip(texts[-3:], translate(texts[-3:])):
    print(zh, "->", en)

# %% Cell 4: translate every unique text (sorted by length, progress printed)
BS = 16
cache = {}
t0 = time.time()
for b in range(0, len(texts), BS):
    batch = texts[b:b + BS]
    for zh, en in zip(batch, translate(batch)):
        cache[zh] = en
    if (b // BS) % 10 == 0:
        print(f"{len(cache)}/{len(texts)} texts, {round(time.time() - t0)} s", flush=True)
print("translation done in", round(time.time() - t0), "s")

# %% Cell 5: attach translations to the rows and save
for r in rows:
    for name, k in KEYS.items():
        r[f"en_mt_{name}"] = cache.get(r[k].strip(), "")
open("/kaggle/working/manifest_mt.jsonl", "w", encoding="utf-8").write(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
for r in rows[:5]:
    print(r["en_ref"], "|", r["en_mt_ref"], "|", r["en_mt_whisper"], "|", r["en_mt_paraformer"])
print(os.listdir("/kaggle/working"))
