#!/usr/bin/env python3
# Paired spec-decode probe corpus: fixed prompt cells (code d0/d4096/d8192,
# filler, structured), deterministic so two engines see identical input.
# Usage: spec_ab_probe_prompts.py <engine_url> <out.json>
import json, sys, time, urllib.request

URL = sys.argv[1]
OUT = sys.argv[2]

def ini_parser():
    return "Write a Python function to parse INI-style files, handling sections, comments, and quoted values.\n\ndef parse_ini(text):\n    config = {}\n    current = None\n    for line in text.splitlines():\n        line = line.strip()\n"

def json_diff():
    return "Write a Python function that deep-diffs two JSON dictionaries, returning a list of (path, before, after) tuples.\n\ndef deep_diff(a, b, path=\"\"):\n    diffs = []\n"

def lru_cache():
    return "Write a thread-safe LRU cache decorator in Python with O(1) operations.\n\nimport threading\nclass LRUCache:\n    def __init__(self, size):\n        self.size = size\n        self.data = collections.OrderedDict()\n        self.lock = threading.Lock()\n"

def toml_dump():
    return "Write a Python TOML-style config writer that emits sorted keys and int/str/bool values.\n\ndef dump_toml(data, indent=0):\n    out = []\n"

def jsonl_flush():
    return "Write a Python function that streams a large JSON Lines file through gzip, filtering records by a date range.\n\ndef filter_jsonl(src, dst, start, end):\n    with gzip.open(src) as fin, gzip.open(dst, 'wt') as fout:\n"

def env_expand():
    return "Write a Python function that expands ${VAR} and $VAR references in a shell-style string using a defaults map.\n\ndef expand_env(template, defaults=None):\n    defaults = defaults or {}\n"

CODE = [ini_parser(), json_diff(), lru_cache(), toml_dump(), jsonl_flush(), env_expand()]

def lorem_fill():
    return ("Write a short blog post about a mountain sunrise coffee ritual. Include sensory details.\n\n"
            # ~1200 tokens of filler prefix so ctx-focused tokens are spread
            "For reference, here are the opening lines of the draft so far: "
            "\"The alarm rang at five. I stepped out into cold air and climbed the trail "
            "past sleeping houses with only my headlamp on. Mist moved between the pines. "
            "The kettle in my pack was the heaviest thing I carried, and I was glad of it. "
            "Every switchback opened a wider view of the valley floor.\" "
            "Continue the post in the same voice with concrete detail about the dawn light.\n\n")

STRUCT = [
    ("Convert this CSV to JSON. Keep types: head,name,rating for rows; numbers as int.\n\nhead,name,rating\n1,ada,5\n2,linus,4\n3,grace,5\n\nJSON:", "csv"),
    ("Given this TOML, output a one-line JSON object with the keys sorted alphabetically.\n\n[limits]\nwindow = 7\nblock = 8\n[tokens]\nbos = 248044\neos = 248044\nmask = 248077\n\n", "toml"),
]

def long_code(prefix_msgs):
    base = jsonl_flush()
    return base + "\n\n# Here is the module the function belongs to:\n\n" + prefix_msgs

SRC_MODULE = "\n".join(
    f"def helper_{i}(rows, out, seen=set()):\n"
    f"    for r in rows:\n"
    f"        k = r.get('ts')\n"
    f"        if k in seen: continue\n"
    f"        seen.add(k)\n"
    f"        out.write(json.dumps(r) + '\\n')\n"
    for i in range(40)
)

CELLS = []
for i, p in enumerate(CODE[:3]):
    CELLS.append({"cell": f"code-d0-{i}", "messages": [{"role": "user", "content": p}], "max_tokens": 160})
for i, p in enumerate(CODE[:3]):
    CELLS.append({"cell": f"code-d4096-{i}", "messages": [{"role": "user", "content": long_code(SRC_MODULE)}], "max_tokens": 160})
for i, p in enumerate(CODE[:3]):
    CELLS.append({"cell": f"code-d8192-{i}", "messages": [{"role": "user", "content": long_code(SRC_MODULE) + "\n\n# Extended notes:\n\n" + SRC_MODULE}], "max_tokens": 160})
CELLS.append({"cell": "filler-d0-0", "messages": [{"role": "user", "content": lorem_fill()}], "max_tokens": 160})
CELLS.append({"cell": "filler-d8192-0", "messages": [{"role": "user", "content": lorem_fill() + "The valley held its breath. The sky turned from black to iron, iron to rose, rose to gold.\n\n"}], "max_tokens": 160})
for i, (p, tag) in enumerate(STRUCT[:2]):
    CELLS.append({"cell": f"structured-d0-{i}", "messages": [{"role": "user", "content": p}], "max_tokens": 160})

results = []
for c in CELLS:
    body = {"model": None, "temperature": 0, "stream": False,
            "max_tokens": c["max_tokens"], "messages": c["messages"]}
    t0 = time.time()
    req = urllib.request.Request(URL.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        spec = (resp.get("metrics") or {}).get("speculative_decoding") or {}
        usage = resp.get("usage") or {}
        results.append({
            "cell": c["cell"], "ok": True, "wall_s": round(time.time() - t0, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "draft_acceptance_rate": spec.get("draft_acceptance_rate"),
            "mean_acceptance_length": spec.get("mean_acceptance_length"),
            "acceptance_histogram": spec.get("acceptance_histogram"),
        })
    except Exception as e:  # noqa
        results.append({"cell": c["cell"], "ok": False, "error": repr(e)})
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)

for r in results:
    if r.get("ok"):
        print(f"{r['cell']:>16}  α={r['draft_acceptance_rate']!s:>7.4}  τ={r['mean_acceptance_length']!s:>7.4}  "
              f"ptok={r['prompt_tokens']} ctok={r['completion_tokens']} wall={r['wall_s']}s")
    else:
        print(f"{r['cell']:>16}  ERROR {r['error']}")
