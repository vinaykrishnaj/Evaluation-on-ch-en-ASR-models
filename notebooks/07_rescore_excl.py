# Notebook 7 (= notebook 5 with fleurs_00121 excluded): score the translations against the English references and analyse ASR-error impact.
# Needs: GPU (for COMET), Internet on, zh-mt-outputs (output of 04_translate) attached as input.
# Paste each "# %%" block into its own cell.

# %% Cell 1: install
# unbabel-comet pins numpy<2, and letting pip change numpy breaks transformers/scipy here,
# so install comet without its dependencies plus only what it needs (numpy stays untouched)
%pip install -q sacrebleu entmax "jsonargparse[signatures]" sentencepiece pytorch-lightning torchmetrics
%pip install -q --no-deps unbabel-comet

# %% Cell 2: load
import os, json, glob
# unbabel-comet pins numpy<2, which breaks Kaggle's preinstalled TensorFlow; transformers only needs PyTorch here
os.environ["USE_TF"] = "0"; os.environ["USE_JAX"] = "0"; os.environ["TRANSFORMERS_NO_TF"] = "1"
import numpy as np
import pandas as pd

MANI = glob.glob("/kaggle/input/**/manifest_mt.jsonl", recursive=True)[0]
df = pd.DataFrame([json.loads(l) for l in open(MANI, encoding="utf-8")])
SYS = ["ref", "whisper", "paraformer"]     # "ref" = translation of the clean Chinese reference (upper bound)
print(df.shape, df.columns.tolist())

# Clips to leave out. fleurs_00121: Edge-TTS produced 0.24 s of audio for a 55-character sentence (see 06_audio_check).
EXCLUDE = ["fleurs_00121"]

def corpus_cer(d, s):
    """Corpus CER from the per-clip CERs: total edits / total reference characters (same as jiwer on the whole set)."""
    n = d["zh_ref"].str.count(r"[一-鿿A-Za-z0-9]")
    return float(((d[f"cer_{s}"] * n).round().sum()) / n.sum())

print("corpus CER on all clips:", {s: round(corpus_cer(df, s), 4) for s in ["whisper", "paraformer"]})
if EXCLUDE:
    df = df[~df["id"].isin(EXCLUDE)].reset_index(drop=True)
    print("excluded", EXCLUDE, "->", len(df), "clips")
    for src in ["covost2", "fleurs"]:
        print("corpus CER", src, {s: round(corpus_cer(df[df.source == src], s), 4) for s in ["whisper", "paraformer"]})
    print("corpus CER after exclusion:", {s: round(corpus_cer(df, s), 4) for s in ["whisper", "paraformer"]})

# %% Cell 3: BLEU and chrF++ (corpus and per sentence)
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF
bleu, chrf = BLEU(), CHRF(word_order=2)          # word_order=2 is chrF++
rows = []
for s in SYS:
    for src in ["covost2", "fleurs", "all"]:
        d = df if src == "all" else df[df.source == src]
        hyp, ref = d[f"en_mt_{s}"].tolist(), d["en_ref"].tolist()
        rows.append({"system": s, "set": src, "n": len(d),
                     "BLEU": round(bleu.corpus_score(hyp, [ref]).score, 2),
                     "chrF++": round(chrf.corpus_score(hyp, [ref]).score, 2)})
    df[f"chrf_{s}"] = [chrf.sentence_score(h, [r]).score for h, r in zip(df[f"en_mt_{s}"], df["en_ref"])]
res = pd.DataFrame(rows)
print(res.to_string(index=False))

# %% Cell 4: COMET (wmt22-comet-da needs source, hypothesis, reference)
from comet import download_model, load_from_checkpoint
from comet.encoders import xlmr as _xlmr

# unbabel-comet unpacks a 3-tuple from the encoder, but the installed transformers returns a different tuple;
# read the named outputs instead (same values: last states, pooled output, all hidden layers)
def _encoder_forward(self, input_ids, attention_mask, **kwargs):
    o = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    return {"sentemb": o.pooler_output, "wordemb": o.last_hidden_state,
            "all_layers": o.hidden_states, "attention_mask": attention_mask}
for _c in vars(_xlmr).values():
    if isinstance(_c, type) and _c.__module__ == _xlmr.__name__ and "forward" in vars(_c):
        _c.forward = _encoder_forward

comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
for s in SYS:
    data = [{"src": a, "mt": b, "ref": c} for a, b, c in zip(df["zh_ref"], df[f"en_mt_{s}"], df["en_ref"])]
    out = comet.predict(data, batch_size=32, gpus=1)
    df[f"comet_{s}"] = out.scores
    print(s, "COMET (all) =", round(float(np.mean(out.scores)), 4))
for src in ["covost2", "fleurs"]:
    print(src, {s: round(float(df[df.source == src][f"comet_{s}"].mean()), 4) for s in SYS})

# %% Cell 5: effect of ASR errors, bucketed by per-clip CER (ASR translation vs translation of the clean transcript)
bins = [-0.001, 0, 0.05, 0.15, 1e9]
labels = ["CER 0 (exact)", "0-5%", "5-15%", ">15%"]
out = []
for s in ["whisper", "paraformer"]:
    d = df.copy()
    d["bucket"] = pd.cut(d[f"cer_{s}"], bins=bins, labels=labels)
    g = d.groupby("bucket", observed=True).agg(n=("id", "count"),
                                               chrf_asr=(f"chrf_{s}", "mean"), chrf_clean=("chrf_ref", "mean"),
                                               comet_asr=(f"comet_{s}", "mean"), comet_clean=("comet_ref", "mean")).round(3)
    g["comet_drop"] = (g.comet_clean - g.comet_asr).round(3)
    g.insert(0, "system", s)
    out.append(g.reset_index())
bucket_tbl = pd.concat(out)
print(bucket_tbl.to_string(index=False))
for s in ["whisper", "paraformer"]:
    m = df[df[f"cer_{s}"].notna()]
    print(s, "Pearson(CER, COMET drop) =", round(float(np.corrcoef(m[f"cer_{s}"], m["comet_ref"] - m[f"comet_{s}"])[0, 1]), 3))

# %% Cell 6: plot and save everything
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, s in zip(ax, ["whisper", "paraformer"]):
    a.scatter(df[f"cer_{s}"], df["comet_ref"] - df[f"comet_{s}"], s=8, alpha=0.5)
    a.set_xlabel("per-clip CER"); a.set_ylabel("COMET drop vs clean-transcript translation"); a.set_title(s)
plt.tight_layout(); plt.savefig("/kaggle/working/cer_vs_comet_drop.png", dpi=130)
res.to_csv("/kaggle/working/scores_overall.csv", index=False)
bucket_tbl.to_csv("/kaggle/working/scores_by_cer_bucket.csv", index=False)
df.to_json("/kaggle/working/manifest_scored.jsonl", orient="records", lines=True, force_ascii=False)
# the 20 worst ASR-induced translation drops, for manual error analysis
worst = []
for s in ["whisper", "paraformer"]:
    w = df.assign(drop=df["comet_ref"] - df[f"comet_{s}"]).nlargest(20, "drop")
    worst.append(w[["id", "zh_ref", f"zh_asr_{s}", "en_ref", "en_mt_ref", f"en_mt_{s}", f"cer_{s}", "drop"]].assign(system=s)
                 .rename(columns={f"zh_asr_{s}": "zh_asr", f"en_mt_{s}": "en_mt_asr", f"cer_{s}": "cer"}))
pd.concat(worst).to_csv("/kaggle/working/worst_cases.csv", index=False)
print(os.listdir("/kaggle/working"))
