import json, os, urllib.request, time
os.chdir("/mnt/c/workspace/kiro-gateway")
from dotenv import load_dotenv; load_dotenv()
KEY = os.getenv("PROXY_API_KEY", "").strip('"')
URL = "http://localhost:8000/v1/chat/completions"

def call(eff, prompt):
    b = {"model": "claude-opus-4.8", "messages": [{"role": "user", "content": prompt}], "stream": False}
    if eff:
        b["reasoning_effort"] = eff
    req = urllib.request.Request(URL, data=json.dumps(b).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=100) as r:
            d = json.load(r)
    except Exception as e:
        return None, None, time.time() - t, str(e)
    m = d["choices"][0]["message"]
    return len(m.get("reasoning_content") or ""), len(m.get("content") or ""), time.time() - t, None

P = "Why is the sky blue? Answer in one sentence."
for label, eff in [("default(expect max)", None), ("low", "low")]:
    rc, c, t, err = call(eff, P)
    if err:
        print(f"{label:20}: ERROR after {t:.1f}s: {err}")
    else:
        print(f"{label:20}: reasoning={rc} answer={c} time={t:.1f}s")
