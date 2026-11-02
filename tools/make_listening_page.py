"""Build a self-contained listening page (listen.html) for checking the TTS clips by ear.

Usage:  python make_listening_page.py <folder that contains manifest_tts.jsonl and the unzipped tts audio>

The page is written into that same folder, so the audio links are relative and it works offline.
It picks the clips that matter for the ASR/translation results (defective, fast, worst-case, controls)
plus enough random clips that every one of the four voices is heard at least 4 times.
Ratings are saved in the browser and can be exported as a CSV.
"""
import sys, os, json, glob, random, html
from urllib.parse import quote

FIXED = {
    "fleurs_00121": "DEFECTIVE: audio is only 0.24 s for a 55-character sentence",
    "fleurs_00211": "fast speech (6.8 chars/s), long sentence",
    "fleurs_00200": "fast speech (8.2 chars/s), Latin-script names; also a worst case",
    "fleurs_00126": "Whisper got stuck repeating a name on this one",
    "fleurs_00305": "foreign place name (Bishkek) was mangled by ASR",
    "fleurs_00338": "numbers and foreign words; ASR wrote numerals as words",
    "covost2_00357": "worst case: 导入导出 heard as 倒入倒出",
    "covost2_00019": "worst case: rare surname phrase",
    "covost2_00194": "worst case: 铂基 heard as 搏击",
    "covost2_00184": "worst case: botanical term",
    "covost2_00036": "worst case: 禁蒙面法",
    "covost2_00018": "worst case: rare historical title",
    "covost2_00052": "CER 0.0 but large translation drop (question mark lost)",
    "covost2_00000": "very short clip (padding check)",
    "covost2_00443": "very short clip (padding check)",
    "covost2_00129": "very short clip (padding check)",
    "covost2_00114": "control (no known problem)",
    "covost2_00025": "control (no known problem)",
    "covost2_00281": "control (no known problem)",
    "fleurs_00154": "control (no known problem)",
    "fleurs_00259": "control (no known problem)",
}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TTS listening check</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mute:#6b6b66;--card:#fff;--line:#dcdcd4;--accent:#2b5fd9}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ececea;--mute:#9a9a94;--card:#202020;--line:#383838;--accent:#7aa2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:22px;margin:0 0 4px} .sub{color:var(--mute);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.top{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:baseline}
.id{font-weight:600} .why{color:var(--mute);font-size:14px} .tag{font-size:12px;border:1px solid var(--line);border-radius:99px;padding:1px 8px;color:var(--mute)}
.zh{font-size:20px;margin:8px 0}
audio{width:100%;margin:4px 0 8px}
.row{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;font-size:14px}
select,input[type=text]{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 6px}
input[type=text]{flex:1;min-width:200px}
button{font:inherit;background:var(--accent);color:#fff;border:0;border-radius:8px;padding:8px 14px;cursor:pointer}
.bar{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.done{outline:2px solid var(--accent)}
</style></head><body><main>
<h1>TTS listening check</h1>
<p class="sub">__N__ clips. For each one: play it, read the text, rate it. Ratings save in this browser as you go; use Export when you finish.</p>
<div class="bar"><span id="count"></span><button id="export">Export ratings (CSV)</button></div>
<div id="list"></div>
<script>
const CLIPS=__CLIPS__;
const KEY="tts-listening-v1";
let saved={};try{saved=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const list=document.getElementById("list");
const natural=["","5 - natural","4 - good","3 - acceptable","2 - robotic/odd","1 - bad"];
const match=["","Matches the text","Small slip (a word or two)","Wrong or missing words","Cut off / silent"];
function opts(a,v){return a.map(x=>`<option${x===v?" selected":""}>${x}</option>`).join("")}
CLIPS.forEach(c=>{
  const s=saved[c.id]||{};
  const d=document.createElement("div");d.className="card";d.id=c.id;
  d.innerHTML=`<div class="top"><span class="id">${c.id}</span><span class="tag">${c.voice}</span><span class="why">${c.why}</span></div>
  <div class="zh">${c.text}</div>
  <audio controls preload="none" src="${c.src}"></audio>
  <div class="row"><label>Naturalness <select data-k="nat">${opts(natural,s.nat||"")}</select></label>
  <label>Speech vs text <select data-k="match">${opts(match,s.match||"")}</select></label>
  <input type="text" data-k="note" placeholder="Notes (e.g. which word was wrong)" value="${(s.note||"").replace(/"/g,"&quot;")}"></div>`;
  d.querySelectorAll("[data-k]").forEach(el=>el.addEventListener("input",()=>{
    saved[c.id]=saved[c.id]||{};saved[c.id][el.dataset.k]=el.value;
    try{localStorage.setItem(KEY,JSON.stringify(saved))}catch(e){}
    refresh();}));
  list.appendChild(d);
});
function refresh(){
  let n=0;CLIPS.forEach(c=>{const s=saved[c.id]||{};const ok=s.nat&&s.match;if(ok)n++;document.getElementById(c.id).classList.toggle("done",!!ok)});
  document.getElementById("count").textContent=n+" of "+CLIPS.length+" rated";}
refresh();
document.getElementById("export").onclick=()=>{
  const q=v=>'"'+String(v||"").replace(/"/g,'""')+'"';
  const rows=[["id","voice","why_selected","text","naturalness","speech_vs_text","note"]];
  CLIPS.forEach(c=>{const s=saved[c.id]||{};rows.push([c.id,c.voice,c.why,c.textraw,s.nat,s.match,s.note])});
  const blob=new Blob(["\\ufeff"+rows.map(r=>r.map(q).join(",")).join("\\n")],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="tts_listening_ratings.csv";a.click();};
</script></main></body></html>"""


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    folder = os.path.abspath(sys.argv[1])
    mani = glob.glob(os.path.join(folder, "**", "manifest_tts.jsonl"), recursive=True)
    if not mani:
        sys.exit("manifest_tts.jsonl not found under " + folder)
    rows = {r["id"]: r for r in (json.loads(l) for l in open(mani[0], encoding="utf-8"))}
    wavs = {os.path.basename(p): p for p in glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)}
    if not wavs:
        sys.exit("no .wav files found - unzip the dataset's tts_audio.zip first")

    chosen = {i: why for i, why in FIXED.items() if i in rows and f"{i}.wav" in wavs}
    voices = sorted({r["tts_voice"] for r in rows.values()})
    rng = random.Random(7)
    for v in voices:
        have = sum(rows[i]["tts_voice"] == v for i in chosen)
        pool = [i for i in rows if rows[i]["tts_voice"] == v and i not in chosen and f"{i}.wav" in wavs]
        rng.shuffle(pool)
        for i in pool[:max(0, 4 - have)]:
            chosen[i] = "random clip (voice coverage)"

    clips = []
    for i, why in sorted(chosen.items()):
        r = rows[i]
        clips.append({"id": i, "voice": r["tts_voice"].replace("zh-CN-", "").replace("Neural", ""), "why": html.escape(why),
                      "text": html.escape(r["zh_ref"]), "textraw": r["zh_ref"],
                      "src": quote(os.path.relpath(wavs[f"{i}.wav"], folder).replace(os.sep, "/"))})
    data = json.dumps(clips, ensure_ascii=False).replace("</", "<\\/")     # keep clip text from closing the <script> tag
    page = PAGE.replace("__N__", str(len(clips))).replace("__CLIPS__", data)
    out = os.path.join(folder, "listen.html")
    open(out, "w", encoding="utf-8").write(page)
    per_voice = {v: sum(rows[c["id"]]["tts_voice"] == v for c in clips) for v in voices}
    print(f"wrote {out}\n{len(clips)} clips; per voice: {per_voice}")


if __name__ == "__main__":
    main()
