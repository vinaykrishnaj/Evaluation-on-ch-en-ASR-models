# Notebook 11: re-translate the same 848 sentences with beam search (and a better prompt for Tower). No training.
# One notebook for all three models: change MODEL_KEY in Cell 1 ("tower", "madlad" or "translategemma"), then Save & Run All.
# Needs: GPU T4 x2, Internet on, zh-asr-transcripts attached as input; for translategemma also the HF_TOKEN secret.
# Output files: manifest_mt_<model>_<variant>.jsonl (same columns as notebooks 4, 8 and 9).
# Paste each "# %%" block into its own Kaggle cell.

# %% Cell 1: settings, manifest, unique Chinese texts
MODEL_KEY = "tower"          # "tower" | "madlad" | "translategemma"
NUM_BEAMS = 4
BS = 8                       # smaller batches than the greedy runs because beams multiply memory

import os, json, glob, time, gc
import torch

CFG = {
    "tower": {"id": "Unbabel/TowerInstruct-7B-v0.2", "variants": ["beam4", "beam4_fewshot"]},
    "madlad": {"id": "google/madlad400-3b-mt", "variants": ["beam4"]},
    "translategemma": {"id": "google/translategemma-4b-it", "variants": ["beam4"]},
}[MODEL_KEY]
MODEL_ID = CFG["id"]
MANI = glob.glob("/kaggle/input/**/manifest_asr.jsonl", recursive=True)[0]
rows = [json.loads(l) for l in open(MANI, encoding="utf-8")]
KEYS = {"ref": "zh_ref", "whisper": "zh_asr_whisper", "paraformer": "zh_asr_paraformer"}
texts = sorted({r[k].strip() for r in rows for k in KEYS.values() if r[k].strip()}, key=len)
print(MODEL_KEY, MODEL_ID, "|", len(rows), "rows;", len(texts), "unique texts; gpus:", torch.cuda.device_count())

# %% Cell 2: load the model and define translate(batch, variant)
GEN = dict(num_beams=NUM_BEAMS, do_sample=False, early_stopping=True)

if MODEL_KEY == "tower":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map="auto").eval()

    # hand-written generic examples (not taken from the datasets)
    SHOTS = [("我明天要去北京出差。", "I have to go to Beijing on a business trip tomorrow."),
             ("请把这份文件发给张伟先生。", "Please send this document to Mr. Zhang Wei."),
             ("这家餐厅的价格是每人三百元。", "The price at this restaurant is 300 yuan per person.")]

    def build(t, variant):
        if variant == "beam4_fewshot":
            shots = "\n\n".join(f"Chinese: {z}\nEnglish: {e}" for z, e in SHOTS)
            content = ("Translate the following text from Chinese into English. Keep names, numbers and technical "
                       "terms accurate and output only the translation.\n\n" + shots + f"\n\nChinese: {t}\nEnglish:")
        else:
            content = f"Translate the following text from Chinese into English.\nChinese: {t}\nEnglish:"
        return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    def translate(batch, variant):
        inp = tok([build(t, variant) for t in batch], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=200, pad_token_id=tok.pad_token_id, **GEN)
        return [s.strip() for s in tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)]

elif MODEL_KEY == "madlad":
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tok = T5Tokenizer.from_pretrained(MODEL_ID)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto", torch_dtype=torch.float32).eval()

    def translate(batch, variant):
        enc = tok(["<2en> " + t for t in batch], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=200, **GEN)
        return [s.strip() for s in tok.batch_decode(out, skip_special_tokens=True)]

else:  # translategemma
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import login
    login(UserSecretsClient().get_secret("HF_TOKEN"))
    from transformers import AutoModelForImageTextToText, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto", dtype=torch.bfloat16).eval()

    def translate(batch, variant):
        msgs = [[{"role": "user", "content": [{"type": "text", "source_lang_code": "zh", "target_lang_code": "en", "text": t}]}] for t in batch]
        inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                            return_tensors="pt", padding=True).to(model.device)
        n = inp["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=256, **GEN)
        return [s.strip() for s in processor.batch_decode(out[:, n:], skip_special_tokens=True)]

for zh, en in zip(texts[-3:], translate(texts[-3:], CFG["variants"][0])):
    print(zh, "->", en)

# %% Cell 3: translate every unique text for each variant and save one file per variant
for variant in CFG["variants"]:
    cache, t0 = {}, time.time()
    for b in range(0, len(texts), BS):
        batch = texts[b:b + BS]
        for zh, en in zip(batch, translate(batch, variant)):
            cache[zh] = en
        if (b // BS) % 25 == 0:
            print(f"[{variant}] {len(cache)}/{len(texts)} texts, {round(time.time() - t0)} s", flush=True)
    print(variant, "done in", round(time.time() - t0), "s")
    out_rows = [dict(r, **{f"en_mt_{n}": cache.get(r[k].strip(), "") for n, k in KEYS.items()}) for r in rows]
    open(f"/kaggle/working/manifest_mt_{MODEL_KEY}_{variant}.jsonl", "w", encoding="utf-8").write(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
print(os.listdir("/kaggle/working"))
