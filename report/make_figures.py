"""Generate every figure used in the final report (PNG, 200 dpi) into report/fig/.

All numbers are the ones measured in the project (see results_summary.md); the clip-duration
histogram is computed from the 849 TTS wav files in listening/zh-tts-audio.
"""
import os, glob, wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)

# validated reference palette (dataviz skill): slot 1 blue, slot 2 orange; neutral for the clean baseline
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"]          # ordinal blue: exact, 0-5%, 5-15%, >15%

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 100, "savefig.dpi": 200, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def label_bars(ax, bars, fmt="{:.1f}", dy=0.0, size=8.5, color=INK):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", xytext=(0, 2 + dy), textcoords="offset points", fontsize=size, color=color)


# ---- Fig 2: clip duration distribution --------------------------------------------------------
wavs = glob.glob(os.path.join(HERE, "..", "listening", "zh-tts-audio", "tts_audio", "tts", "*.wav"))
dur = {}
for p in wavs:
    with wave.open(p) as w:
        dur[os.path.basename(p)[:-4]] = w.getnframes() / w.getframerate()
if dur:
    cov = [v for k, v in dur.items() if k.startswith("covost2")]
    flr = [v for k, v in dur.items() if k.startswith("fleurs")]
    bins = np.arange(0, 22, 1)
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    h1, _ = np.histogram(cov, bins); h2, _ = np.histogram(flr, bins)
    ax.bar(bins[:-1] + 0.5, h1, width=0.92, color=BLUE, label=f"CoVoST 2 ({len(cov)} clips, median {np.median(cov):.1f} s)")
    ax.bar(bins[:-1] + 0.5, h2, width=0.92, bottom=h1, color=ORANGE, label=f"FLEURS ({len(flr)} clips, median {np.median(flr):.1f} s)")
    ax.set_xlabel("Clip duration (seconds)"); ax.set_ylabel("Number of clips"); ax.set_xlim(0, 21)
    ax.legend(loc="upper right")
    ax.annotate("fleurs_00121\n0.24 s", xy=(0.4, 2), xytext=(0.05, 52), fontsize=7.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    save(fig, "f2_durations.png")

# ---- Fig 3: character error rate ---------------------------------------------------------------
groups = ["CoVoST 2\n(500 clips)", "FLEURS\n(348 clips)", "All\n(848 clips)"]
whisper, para = [9.72, 4.59, 6.57], [6.69, 6.01, 6.27]
fig, ax = plt.subplots(figsize=(5.6, 2.9))
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, whisper, w, color=BLUE, label="Whisper large-v3")
b2 = ax.bar(x + w / 2 + 0.01, para, w, color=ORANGE, label="Paraformer-zh")
label_bars(ax, b1, "{:.2f}%"); label_bars(ax, b2, "{:.2f}%")
ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("Corpus CER (%)"); ax.set_ylim(0, 11.5)
ax.legend(loc="upper right", ncol=1)
save(fig, "f3_cer.png")

# ---- Fig 4: translation quality, clean transcript vs ASR transcripts -----------------------------
systems = ["Clean\ntext", "Whisper", "Paraformer"]
cols = [GRAY, BLUE, ORANGE]
panels = [("BLEU", [27.81, 24.24, 23.50], 33, "{:.2f}"), ("chrF++", [52.46, 49.21, 48.89], 62, "{:.2f}"),
          ("COMET-22 (x100)", [81.95, 78.27, 78.99], 98, "{:.1f}")]
fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.7))
for ax, (t, vals, top, fmt) in zip(axs, panels):
    bars = ax.bar(range(3), vals, 0.7, color=cols)
    label_bars(ax, bars, fmt)
    ax.set_xticks(range(3)); ax.set_xticklabels(systems, fontsize=7.5); ax.set_title(t, fontsize=9.5, color=INK, pad=6)
    ax.set_ylim(0, top); ax.grid(axis="x", visible=False)
fig.tight_layout(w_pad=1.2)
save(fig, "f4_translation.png")

# ---- Fig 5: COMET by source -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 2.9))
src = ["CoVoST 2 (500 clips)", "FLEURS (348 clips)"]
vals = {"Clean text": [78.56, 86.83], "Whisper": [73.89, 84.56], "Paraformer": [75.22, 84.40]}
x = np.arange(2); w = 0.26
for i, (k, c) in enumerate(zip(vals, cols)):
    bars = ax.bar(x + (i - 1) * (w + 0.015), vals[k], w, color=c, label=k)
    label_bars(ax, bars, "{:.1f}", size=8)
ax.set_xticks(x); ax.set_xticklabels(src); ax.set_ylabel("COMET-22 (x100)"); ax.set_ylim(0, 100)
ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.32))
save(fig, "f5_comet_source.png")

# ---- Fig 6: COMET drop by per-clip CER bucket ----------------------------------------------------
buckets = ["CER = 0\n(exact)", "0-5%", "5-15%", "> 15%"]
n_w, n_p = [411, 98, 167, 172], [427, 94, 189, 138]
d_w, d_p = [0.007, 0.015, 0.044, 0.113], [0.009, 0.016, 0.051, 0.074]
fig, ax = plt.subplots(figsize=(6.0, 3.0))
x = np.arange(4); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, d_w, w, color=BLUE, label="Whisper large-v3")
b2 = ax.bar(x + w / 2 + 0.01, d_p, w, color=ORANGE, label="Paraformer-zh")
label_bars(ax, b1, "{:.3f}"); label_bars(ax, b2, "{:.3f}")
ax.set_xticks(x)
ax.set_xticklabels([f"{b}\nn = {a} / {c}" for b, a, c in zip(buckets, n_w, n_p)], fontsize=8)
ax.set_ylabel("COMET drop vs clean-text translation"); ax.set_ylim(0, 0.135)
ax.legend(loc="upper left")
ax.text(0.99, -0.30, "n = number of clips (Whisper / Paraformer)", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
save(fig, "f6_bucket_drop.png")

# ---- Fig 7: share of clips vs share of translation loss ---------------------------------------
def shares(n, d):
    n = np.array(n, float); loss = n * np.array(d)
    return 100 * n / n.sum(), 100 * loss / loss.sum()
cw, lw = shares(n_w, d_w); cp, lp = shares(n_p, d_p)
rows = [("Whisper: share of clips", cw), ("Whisper: share of COMET loss", lw),
        ("Paraformer: share of clips", cp), ("Paraformer: share of COMET loss", lp)]
fig, ax = plt.subplots(figsize=(6.6, 2.7))
ypos = [3.25, 2.55, 1.25, 0.55]
names = ["CER = 0 (exact)", "CER 0-5%", "CER 5-15%", "CER > 15%"]
for (label, v), y in zip(rows, ypos):
    left = 0
    for i, s in enumerate(v):
        ax.barh(y, s, left=left, height=0.55, color=RAMP[i], edgecolor="white", linewidth=1.5,
                label=names[i] if y == ypos[0] else None)
        if s >= 6:
            ax.text(left + s / 2, y, f"{s:.0f}%", ha="center", va="center", fontsize=8, color=INK if i < 2 else "white")
        left += s
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
ax.set_xlim(0, 100); ax.set_xlabel("Percent"); ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.45, -0.22), fontsize=8)
save(fig, "f7_loss_share.png")

# ---- Fig 8: error types among the 40 worst cases -----------------------------------------------
cats = ["Homophone error in a rare\nname or technical term", "Homophone error in a\ncommon word",
        "Foreign name or Latin-script\nword mangled", "Missing punctuation /\nnumbers written as words", "Whole-clip failure\n(loop, empty, unrelated)"]
cw8, cp8 = [12, 4, 2, 0, 2], [10, 4, 2, 3, 1]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
y = np.arange(len(cats))[::-1]; h = 0.36
b1 = ax.barh(y + h / 2 + 0.01, cw8, h, color=BLUE, label="Whisper (20 worst clips)")
b2 = ax.barh(y - h / 2 - 0.01, cp8, h, color=ORANGE, label="Paraformer (20 worst clips)")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{int(b.get_width())}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=8.2); ax.set_xlabel("Number of clips"); ax.set_xlim(0, 14)
ax.grid(axis="y", visible=False); ax.legend(loc="lower right")
save(fig, "f8_error_types.png")

# ---- Fig 9: COMET-22 in context of the reference paper ---------------------------------------------
items = [("This study: Whisper transcript, zh->en", 78.27, BLUE), ("This study: Paraformer transcript, zh->en", 78.99, BLUE),
         ("This study: clean Chinese text, zh->en", 81.95, BLUE),
         ("Paper: TowerInstruct, WMT23 general, xx->en", 82.77, GRAY), ("Paper: TowerInstruct, BConTrast de->en", 91.98, GRAY),
         ("Paper: TowerInstruct, WMT24 chat, xx->en", 92.17, GRAY), ("Paper: GPT-4o, WMT24 chat, xx->en", 92.41, GRAY),
         ("Paper: TowerChat + context, WMT24 chat", 93.01, GRAY), ("Paper: TowerChat + context + QAD, WMT24 chat", 93.39, GRAY)]
fig, ax = plt.subplots(figsize=(6.6, 3.5))
y = np.arange(len(items))[::-1]
bars = ax.barh(y, [i[1] for i in items], 0.62, color=[i[2] for i in items])
for b, it in zip(bars, items):
    ax.annotate(f"{it[1]:.1f}", (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=8)
ax.set_xlim(0, 105); ax.set_xlabel("COMET-22 (x100); different data and language pairs, so indicative only")
ax.grid(axis="y", visible=False)
save(fig, "f9_paper_context.png")
print("figures written to", OUT, sorted(os.listdir(OUT)))
