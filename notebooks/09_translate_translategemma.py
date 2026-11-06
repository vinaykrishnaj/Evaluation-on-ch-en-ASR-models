# Notebook 9: translate the same 848 sentences with TranslateGemma-4B-IT (replaces TowerInstruct at step 4).
# Needs: GPU T4 x2, Internet on, zh-asr-transcripts attached as input, the HF_TOKEN Kaggle secret attached,
# and the Gemma terms accepted on https://huggingface.co/google/translategemma-4b-it with the same Hugging Face account.
# Paste each "# %%" block into its own Kaggle cell.

# %% Cell 1: log in to Hugging Face (the model is gated) and load the manifest
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
login(UserSecretsClient().get_secret("HF_TOKEN"))

import os, json, glob, time, gc
import torch

MODEL_ID = "google/translategemma-4b-it"
MANI = glob.glob("/kaggle/input/**/manifest_asr.jsonl", recursive=True)[0]
rows = [json.loads(l) for l in open(MANI, encoding="utf-8")]
KEYS = {"ref": "zh_ref", "whisper": "zh_asr_whisper", "paraformer": "zh_asr_paraformer"}
texts = sorted({r[k].strip() for r in rows for k in KEYS.values() if r[k].strip()}, key=len)
print(len(rows), "rows;", len(texts), "unique Chinese texts; gpus:", torch.cuda.device_count())

# %% Cell 2: load the model. Weights are bf16; T4 GPUs have no native bf16, so the smoke test below decides bf16 or fp16.
from transformers import AutoModelForImageTextToText, AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "left"

def load(dtype):
    m = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto", dtype=dtype).eval()
    return m

def messages_for(text, src):
    return [{"role": "user", "content": [{"type": "text", "source_lang_code": src, "target_lang_code": "en", "text": text}]}]

# the card does not list the Chinese code; the chat template raises an error for an unsupported code, so try the candidates
SRC = None
for cand in ["zh", "zh-CN", "zh-Hans", "zh_CN"]:
    try:
        processor.apply_chat_template(messages_for("你好", cand), tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        SRC = cand
        break
    except Exception as e:
        print("code", cand, "rejected:", str(e)[:100])
print("Chinese source code accepted:", SRC)

# %% Cell 3: smoke test in bf16, fall back to fp16 if it fails or returns empty text
def translate(batch):
    msgs = [messages_for(t, SRC) for t in batch]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                        return_tensors="pt", padding=True).to(model.device)
    n = inp["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=256, do_sample=False)
    return [s.strip() for s in processor.batch_decode(out[:, n:], skip_special_tokens=True)]

sample = texts[-3:]
for dtype in (torch.bfloat16, torch.float16):
    try:
        model = load(dtype)
        res = translate(sample)
        if all(len(x) > 3 for x in res):
            print("dtype OK:", dtype)
            break
        raise RuntimeError("empty output")
    except Exception as e:
        print("dtype", dtype, "failed:", str(e)[:150])
        model = None; gc.collect(); torch.cuda.empty_cache()
for zh, en in zip(sample, res):
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

# %% Cell 5: attach translations (same column names as notebook 4) and save
for r in rows:
    for name, k in KEYS.items():
        r[f"en_mt_{name}"] = cache.get(r[k].strip(), "")
open("/kaggle/working/manifest_mt_translategemma.jsonl", "w", encoding="utf-8").write(
    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
for r in rows[:5]:
    print(r["en_ref"], "|", r["en_mt_ref"], "|", r["en_mt_whisper"], "|", r["en_mt_paraformer"])
print(os.listdir("/kaggle/working"))
