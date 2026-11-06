# Notebook 10: score every translation model on the same 848 clips and put the results in one table.
# Needs: GPU (for COMET), Internet on. Attach as input the datasets that hold the translation files:
#   zh-mt-outputs (manifest_mt.jsonl = TowerInstruct), and the datasets with manifest_mt_madlad.jsonl and
#   manifest_mt_translategemma.jsonl. Any file named manifest_mt*.jsonl found under /kaggle/input is scored.
# Paste each "# %%" block into its own cell.

# %% Cell 1: install (COMET without its dependencies; numpy stays untouched)
%pip install -q sacrebleu entmax "jsonargparse[signatures]" sentencepiece pytorch-lightning torchmetrics
%pip install -q --no-deps unbabel-comet

# %% Cell 2: load every model's translations
import os, json, glob
os.environ["USE_TF"] = "0"; os.environ["USE_JAX"] = "0"; os.environ["TRANSFORMERS_NO_TF"] = "1"
import numpy as np
import pandas as pd

LABELS = {"manifest_mt.jsonl": "TowerInstruct-7B-v0.2", "manifest_mt_madlad.jsonl": "MADLAD-400-3B-MT",
          "manifest_mt_translategemma.jsonl": "TranslateGemma-4B-IT"}
SYS = ["ref", "whisper", "paraformer"]          # "ref" = translation of the correct Chinese text (clean)
EXCLUDE = ["fleurs_00121"]                      # defective TTS clip (see 06_audio_check)

frames = {}
for p in sorted(glob.glob("/kaggle/input/**/manifest_mt*.jsonl", recursive=True)):
    name = LABELS.get(os.path.basename(p), os.path.basename(p))
    d = pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])
    d = d[~d["id"].isin(EXCLUDE)].reset_index(drop=True)
    frames[name] = d
    print(name, d.shape)
base = next(iter(frames.values()))
for name, d in frames.items():              # the same sentences and the same transcripts for every model
    assert d["id"].tolist() == base["id"].tolist(), name
    assert d["zh_asr_whisper"].tolist() == base["zh_asr_whisper"].tolist(), name
print("all models share the same", len(base), "clips and transcripts")

# %% Cell 3: BLEU and chrF++
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
bleu, chrf = BLEU(), CHRF(word_order=2)
rows = []
for name, d in frames.items():
    for s in SYS:
        for src in ["all", "covost2", "fleurs"]:
            x = d if src == "all" else d[d.source == src]
            hyp, ref = x[f"en_mt_{s}"].tolist(), x["en_ref"].tolist()
            rows.append({"model": name, "system": s, "set": src, "n": len(x),
                         "BLEU": round(bleu.corpus_score(hyp, [ref]).score, 2),
                         "chrF++": round(chrf.corpus_score(hyp, [ref]).score, 2)})
res = pd.DataFrame(rows)
print(res[res.set == "all"].to_string(index=False))

# %% Cell 4: COMET-22 (patched encoder, as in notebook 7)
from comet import download_model, load_from_checkpoint
from comet.encoders import xlmr as _xlmr

def _encoder_forward(self, input_ids, attention_mask, **kwargs):
    o = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    return {"sentemb": o.pooler_output, "wordemb": o.last_hidden_state,
            "all_layers": o.hidden_states, "attention_mask": attention_mask}
for _c in vars(_xlmr).values():
    if isinstance(_c, type) and _c.__module__ == _xlmr.__name__ and "forward" in vars(_c):
        _c.forward = _encoder_forward

comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
for name, d in frames.items():
    for s in SYS:
        data = [{"src": a, "mt": b, "ref": c} for a, b, c in zip(d["zh_ref"], d[f"en_mt_{s}"], d["en_ref"])]
        d[f"comet_{s}"] = comet.predict(data, batch_size=32, gpus=1, progress_bar=False).scores
    print(name, {s: round(float(d[f"comet_{s}"].mean()), 4) for s in SYS})

# %% Cell 5: one comparison table (each model's loss is measured against its own clean translation)
out = []
for name, d in frames.items():
    for s in SYS:
        r = res[(res.model == name) & (res.system == s) & (res.set == "all")].iloc[0]
        out.append({"model": name, "input": {"ref": "clean text", "whisper": "Whisper", "paraformer": "Paraformer"}[s],
                    "BLEU": r["BLEU"], "chrF++": r["chrF++"], "COMET": round(float(d[f"comet_{s}"].mean()), 4)})
table = pd.DataFrame(out)
for m in ["BLEU", "chrF++", "COMET"]:
    clean = table[table.input == "clean text"].set_index("model")[m]
    table[f"{m}_loss"] = [round(row[m] - clean[row["model"]], 4) for _, row in table.iterrows()]
print(table.to_string(index=False))
for src in ["covost2", "fleurs"]:
    print("\n", src)
    print(res[res.set == src][["model", "system", "BLEU", "chrF++"]].to_string(index=False))

# %% Cell 6: save
table.to_csv("/kaggle/working/model_comparison.csv", index=False)
res.to_csv("/kaggle/working/model_comparison_by_source.csv", index=False)
pd.concat([d.assign(model=n) for n, d in frames.items()]).to_json("/kaggle/working/comparison_scored.jsonl",
                                                                  orient="records", lines=True, force_ascii=False)
print(os.listdir("/kaggle/working"))
