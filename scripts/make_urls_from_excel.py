import sys, re, pathlib
import pandas as pd

excel = sys.argv[1]
out   = sys.argv[2]
df_map = pd.read_excel(excel, sheet_name=None, header=None)

url_re = re.compile(r"https?://[^\s)>\]]+", re.I)

raw = []
for _, frame in df_map.items():
    for val in frame.values.ravel():
        if isinstance(val, str):
            raw += url_re.findall(val)

# dedupe, preserve order
seen = set(); urls = []
for u in raw:
    if u not in seen:
        seen.add(u); urls.append(u)

# remove blocked hosts
blocked = ["sempa.site-ym.com"]
cleaned = [u for u in urls if not any(b in u for b in blocked)]

pathlib.Path(out).write_text("\n".join(cleaned), encoding="utf-8")
print(f"Wrote {len(cleaned)} URLs to {out}")
