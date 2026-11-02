# Pilot results: effect of Chinese ASR errors on zh→en translation

Pipeline (849 utterances: 500 CoVoST 2 zh→en + 349 FLEURS, seeded subset):
reference zh text → Edge-TTS speech (4 voices) → ASR → Tower translation → compared with the English reference.

- **ASR:** Whisper large-v3, Paraformer-zh (FunASR)
- **MT:** `Unbabel/TowerInstruct-7B-v0.2`, fp16, greedy. ("TowerChat" is not a model name; this is the closest general Tower model with confirmed Chinese support.)
- **"clean" baseline:** Tower translation of the original Chinese reference, i.e. the ceiling when ASR makes no errors.
- **CER:** characters, after traditional→simplified conversion and punctuation removal.
- **COMET:** `Unbabel/wmt22-comet-da`, with the original Chinese reference as source.

## Headline results: 848 clips (`fleurs_00121` excluded)

`fleurs_00121` is excluded because its TTS audio is defective (see the audio check below), so no ASR or MT error can be attributed to it. Computed in notebook `07_rescore_excl`; the same notebook reproduced the 849-clip CERs exactly before excluding, so the comparison is like for like. The tables after this section are the original 849-clip results, kept for comparison.

| Model | CER CoVoST 2 (500) | CER FLEURS (348) | CER all (848) |
|---|---|---|---|
| Whisper large-v3 | 9.72% | 4.59% | 6.57% |
| Paraformer-zh | 6.69% | 6.01% | 6.27% |

| Input to Tower | BLEU | chrF++ | COMET |
|---|---|---|---|
| clean Chinese reference | 27.81 | 52.46 | 0.8195 |
| Whisper transcript | 24.24 | 49.21 | 0.7827 |
| Paraformer transcript | 23.50 | 48.89 | 0.7899 |

Drop caused by ASR errors: Whisper −3.57 BLEU / −3.25 chrF++ / −0.037 COMET; Paraformer −4.31 / −3.57 / −0.030.

By source, FLEURS (348): BLEU/chrF++ clean 30.02/55.62, Whisper 26.06/52.56, Paraformer 24.68/51.63; COMET clean 0.8683, Whisper 0.8456, Paraformer 0.8440. CoVoST 2 numbers are unchanged from the table below.

Per-clip CER bucket (COMET drop vs clean-transcript translation): Whisper exact 0.007, 0–5% 0.015, 5–15% 0.044, >15% 0.113 (n=172); Paraformer exact 0.009, 0–5% 0.016, 5–15% 0.051, >15% 0.074 (n=138). Pearson(CER, COMET drop): Whisper 0.50, Paraformer 0.33.

Effect of the exclusion (848 vs 849 clips): CER falls by 0.24 points for both models (Whisper 6.81→6.57, Paraformer 6.51→6.27). Clean-baseline scores move by −0.06 BLEU, −0.04 chrF++ and −0.0001 COMET; the ASR systems move by +0.06 BLEU, +0.10 chrF++ and +0.0005 COMET. The conclusions do not change.

## Original results on all 849 clips

### ASR character error rate (corpus)

| Model | CoVoST 2 (500) | FLEURS (349) | All (849) |
|---|---|---|---|
| Whisper large-v3 | 9.72% | 4.98% | 6.81% |
| Paraformer-zh | 6.69% | 6.40% | 6.51% |

### Translation quality (all 849)

| Input to Tower | BLEU | chrF++ | COMET |
|---|---|---|---|
| clean Chinese reference | 27.87 | 52.50 | 0.8196 |
| Whisper transcript | 24.18 | 49.11 | 0.7822 |
| Paraformer transcript | 23.44 | 48.79 | 0.7894 |

Drop caused by ASR errors: Whisper −3.69 BLEU / −3.39 chrF++ / −0.037 COMET; Paraformer −4.43 / −3.71 / −0.030.

By source (COMET): CoVoST 2 — clean 0.7856, Whisper 0.7389, Paraformer 0.7522. FLEURS — clean 0.8685, Whisper 0.8442, Paraformer 0.8426.
By source (BLEU / chrF++): CoVoST 2 — clean 24.77/48.34, Whisper 21.75/44.82, Paraformer 21.89/45.31. FLEURS — clean 30.11/55.67, Whisper 25.94/52.37, Paraformer 24.57/51.44.

### Translation drop by per-clip CER bucket (all 849)

| CER bucket | System | n | COMET (ASR input) | COMET (clean) | COMET drop |
|---|---|---|---|---|---|
| exact (0) | Whisper | 411 | 0.848 | 0.855 | 0.007 |
| 0–5% | Whisper | 98 | 0.825 | 0.840 | 0.015 |
| 5–15% | Whisper | 167 | 0.772 | 0.816 | 0.044 |
| >15% | Whisper | 173 | 0.613 | 0.728 | 0.115 |
| exact (0) | Paraformer | 427 | 0.834 | 0.843 | 0.009 |
| 0–5% | Paraformer | 94 | 0.825 | 0.841 | 0.016 |
| 5–15% | Paraformer | 189 | 0.765 | 0.816 | 0.051 |
| >15% | Paraformer | 139 | 0.661 | 0.739 | 0.078 |

Pearson correlation between per-clip CER and COMET drop: Whisper 0.52, Paraformer 0.37.

## Reading the numbers (and their limits)

- Translation quality falls monotonically as CER rises. Clips with exact transcripts lose almost nothing (0.007–0.009 COMET), so the loss comes from the erroneous clips.
- Paraformer has the lower CER overall (6.51% vs 6.81%) but a larger BLEU drop; Whisper's COMET drop is larger. CER alone does not rank the two systems for downstream translation.
- Whisper's worst clips include repetition-hallucination loops (e.g. a name repeated until the token limit); Tower then translates the loop, which produces large drops. Not yet counted separately.
- Caveats: pilot size (849); one TTS engine (Edge-TTS, cloud, not strictly reproducible); synthetic speech, not real; CoVoST 2 English references are noisy, which depresses absolute scores for all systems equally; Tower is a TowerInstruct model rather than a "TowerChat"; error types (substitution/deletion/homophone) were not classified yet.

## Error analysis of the worst cases

Based on the 40 largest COMET drops (20 per ASR model, from `worst_cases.csv`); classified by hand, not checked by a native speaker, and only about 5% of the data, so it shows what kinds of errors cause the big losses, not their share of all errors.

| Primary cause | Whisper (20) | Paraformer (20) |
|---|---|---|
| Homophone/near-homophone substitution in rare names or technical terms (e.g. 铂基→搏击, 南川绣线菊→南川秀县局, 寺坪→四平) | 12 | 10 |
| Homophone substitution in common words (e.g. 公测→公厕, 禁蒙面法→进盟面法) | 4 | 4 |
| Foreign names / Latin-script words mangled (e.g. Bishkek→"鄙视凯克bitchcake", Lakkha Singh) | 2 | 2 |
| Missing punctuation / numbers written as words, with exact or near-exact characters (e.g. 我吃的亏还少吗 has CER 0.0 but drops 0.31) | 0 | 3 |
| Whole-clip failure: repetition loop or empty/unrelated output | 2 | 1 |

What this shows:
- Most big drops come from **single-character homophone errors in rare proper nouns and technical terms**. Tower then translates the wrong word confidently ("Other combat drugs include sinupret and kampfer" for "platinum-based drugs ... cisplatin and carboplatin"). CoVoST 2 is built from Wikipedia sentences, which are full of such terms.
- **CER hides punctuation and number-format problems.** CER is computed after removing punctuation, but Paraformer's output has no punctuation and writes numbers as words ("五百", "一九八九p幺零九"), and that changes the translation. A CER of 0.0 can still cost 0.31 COMET.
- **Whisper's failure mode** is a repetition loop (fleurs_00126 repeats "克里斯蒂娜" until the token limit) or an unrelated short output (fleurs_00121 returned "《一切》"). **Paraformer's** is an empty output on the same clip (fleurs_00121). That one clip is the single worst case for both systems; the audio check below shows its TTS audio is defective (0.24 s long for a 55-character sentence), so this "ASR failure" is really a TTS failure.
- Some drops partly reflect the MT model rather than ASR: e.g. 斯勒格是英制单位 loses 0.28 COMET with only one wrong character, because the translation of the clean text already differs from the noisy reference.

## TTS audio check (notebook `06_audio_check`, all 849 clips)

Automatic checks only: durations, loudness, clipping, silence, and speech rate per character. This cannot tell whether the voices sound natural; that still needs human listening.

- Durations 0.24–20.7 s (median 5.6 s); **none longer than 30 s**; no clipping (peak ≤ 0.91); no very quiet clips (RMS −31 to −18 dB); no internal gaps longer than 2 s. All audio is 16 kHz mono.
- **One defective clip: `fleurs_00121`.** The sentence has 55 characters but the audio is only 0.24 s (89% silent). Edge-TTS returned almost nothing. This explains why Whisper output "《一切》" and Paraformer output nothing, and why it is the worst case for both systems. It counted as ~55 character errors for each system, worth 0.24 percentage points of corpus CER (confirmed by the recomputation above).
- 15 clips are flagged "mostly silent" but they are all very short CoVoST 2 phrases (3–5 characters, 1.87 s): padding, not defects.
- 2 clips have unusually fast speech: `fleurs_00211` (6.8 chars/s) and `fleurs_00200` (8.2 chars/s; also in the worst-case list). Both are long sentences with Latin-script names; the fast delivery may add to the ASR errors there.

Output: `audio_check.csv` (per-clip measurements) and `audio_flagged.csv` (17 flagged clips) in the `06_audio_check` notebook output on Kaggle.

## Where the files are

Kaggle (private): notebooks `01_datasets`, `02_tts`, `03_asr`, `04_translate`, `05_score`; datasets `zh-en-manifest`, `zh-tts-audio`, `zh-asr-transcripts`, `zh-mt-outputs`.
`05_score` output: `manifest_scored.jsonl`, `scores_overall.csv`, `scores_by_cer_bucket.csv`, `worst_cases.csv` (20 worst drops per system), `cer_vs_comet_drop.png`.
Local code: `notebooks/*.py` and `*.ipynb`.
