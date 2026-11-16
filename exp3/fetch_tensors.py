"""Download only specific tensors from a remote .safetensors file via HTTP range requests."""
import json, struct, sys, urllib.request
import numpy as np

def get(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()

def fetch(repo, rev, fname, names):
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"
    n = struct.unpack("<Q", get(url, 0, 7))[0]
    header = json.loads(get(url, 8, 8 + n - 1))
    out = {}
    for name in names:
        info = header[name]
        s, e = info["data_offsets"]
        raw = get(url, 8 + n + s, 8 + n + e - 1)
        assert info["dtype"] == "BF16", info["dtype"]
        u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        out[name] = u16.view(np.float32).reshape(info["shape"])
    return out

if __name__ == "__main__":
    names = [f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in (15, 19)]
    FN = "model.safetensors-00001-of-00001.safetensors"
    edit = fetch("RikiyaT/Qwen3.5-0.8B-weight-space-programming", "main", FN, names)
    base = fetch("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", FN, names)
    np.savez("data/mit_edit_tensors.npz", **{f"edit_{k}": v for k, v in edit.items()}, **{f"base_{k}": v for k, v in base.items()})
    for k in names:
        d = edit[k] - base[k]
        cols = np.flatnonzero(np.abs(d).sum(0) > 0); rows = np.flatnonzero(np.abs(d).sum(1) > 0)
        sv = np.linalg.svd(d, compute_uv=False)
        print(k, "shape", d.shape, "changed values", int((d != 0).sum()), "| changed cols", len(cols), "rows", len(rows),
              "| top singular values", np.round(sv[:3], 4), "| rel. change", f"{np.linalg.norm(d) / np.linalg.norm(base[k]):.4f}")
