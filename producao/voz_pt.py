# Gera a voz PT-BR (Chatterbox, MIT) frase a frase, com controlo de qualidade por transcrição.
import sys, json, os, difflib, re, unicodedata
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from faster_whisper import WhisperModel
frases = json.load(open(sys.argv[1])); shard, n = int(sys.argv[2]), int(sys.argv[3]); out = sys.argv[4]
os.makedirs(out, exist_ok=True)
m = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
asr = WhisperModel("small", device="cpu", compute_type="int8")
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", unicodedata.normalize("NFKD", s.lower()).encode("ascii","ignore").decode())
rel = []
for k, f in enumerate(frases):
    if k % n != shard: continue
    best = (-1, None)
    for tent in range(3):
        w = m.generate(f["texto"], language_id="pt", audio_prompt_path="producao/ref_cadu.wav", exaggeration=0.5, cfg_weight=0.5)
        p = f"{out}/{f['id']}.wav"; ta.save(p, w, m.sr)
        segs, _ = asr.transcribe(p, language="pt")
        got = " ".join(s.text for s in segs)
        sim = difflib.SequenceMatcher(None, norm(f["texto"]), norm(got)).ratio()
        if sim > best[0]: best = (sim, w)
        if sim >= 0.85: break
    ta.save(f"{out}/{f['id']}.wav", best[1], m.sr)
    rel.append({"id": f["id"], "sim": round(best[0], 3), "tentativas": tent + 1}); print(rel[-1], flush=True)
json.dump(rel, open(f"{out}/qc_{shard}.json", "w"))
