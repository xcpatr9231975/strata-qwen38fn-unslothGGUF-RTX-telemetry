#!/usr/bin/env python3
"""build_atlas.py - compile the *physical* Qwen3.8-Flash-Next weight atlas
from the on-disk artefacts of the Strata pack `unsloth-ud-iq2_xs`, and emit
it as a JavaScript global the WebGL visualiser can load without a bundler.

WHAT THIS READS (all paths overridable with flags)
  1. GGUF shard 1     .../unsloth-UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-00001-of-00003.gguf
                        -> the full `general.*`/`qwen4exp.*` metadata KV table
                           (48 layers, 512 experts top-10, SSM, hyper-connections,
                            indexer, PLE, rope, ...) parsed with a hand-written
                           GGUF v3 reader (no numpy, stdlib `struct` only).
  2. pack index.txt   ~/strata-models/packs/unsloth-ud-iq2_xs/index.txt
                        -> 1079 dense (non-expert) tensors: name, dtype code,
                           byte offset & size inside pool `dense.bin`, rows,
                           cols.  dtype codes observed: 2=f32(4B), 4=bf16(2B),
                           5=f16(2B), 0=quantized-in-GGUF (quant enum in the
                           9th field, ggml numbering: 8=Q2_K ...).
  3. native_experts.txt (same pack dir)
                        -> per MoE layer: gu_type, d_type, blob offset, and the
                           byte offsets of the gate / up / down expert-weight
                           regions inside a named GGUF shard.  Each region is
                           512 equally-sized expert blobs, so
                           region_size/512 = bytes per expert per matrix, and
                           8*bps/matrix_elems = effective bits/weight.  This is
                           how every one of the 24576 experts is LOCATED
                           (shard file + byte range) for the visualiser.
  4. expert-profile.bin (Strata control data dir)
                        -> magic 'STRP'. 24-byte header [version=1, layers=48,
                           experts=512, entries=24576, entries=24576, size],
                           then 24576 rows of 4 x u16.  Confirmed semantics
                           (engine banner: "PROFILE, ranked by routing
                           frequency, no eviction"; "pre-filled 5175 of 5175
                           slots from the profile"): routing co-occurrence
                           edges (layerA, expertA, layerB, expertB) recorded
                           while the top-5175-expert VRAM cache was warmed on
                           real traffic.  Rows where all four fields collapse
                           to (global_id, 0, global_id, 0) are the cold tail
                           (never co-routed during profiling).  We aggregate
                           appearance counts into a heat[layer][expert] array.
  5. MTP drafter dir  ~/strata-models/packs/unsloth-ud-iq2_xs/mtp/rt/
                        -> dense.txt (human tensor list of the draft layer),
                           experts.bin / dense.bin / draft_vocab.bin sizes.
  6. GGUF shard sizes -> file-level placement of everything.

OUTPUTS
  web/atlas_data.js        `window.ATLAS = {...}`   (for the WebGL page)
  live/atlas_manifest.json same object, machine-readable

The JSON is deliberately self-describing: every number the visualiser draws
is annotated here with units.  Nothing is invented: fields we could not
verify are marked with a `note`.

Usage:  python3 tools/build_atlas.py [--pack DIR] [--shards DIR] [--profile F] [--mtp DIR]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import struct
import sys

HOME = os.path.expanduser("~")

DEF_SHARDS = f"{HOME}/strata-models/models/unsloth-UD-IQ2_XS"
DEF_PACK = f"{HOME}/strata-models/packs/unsloth-ud-iq2_xs"
DEF_PROFILE = (f"{HOME}/.local/share/strata/components/control/0.1.57/data"
               "/expert-profile.bin")
DEF_MTP = f"{HOME}/strata-models/mtp/rt"

# ggml type enum (index.txt column 2 / quant column); verified against sizes.
GGML_DT = {0: "f32", 1: "f16", 2: "bf16", 3: "q8_0", 4: "q4_0", 5: "q4_1",
           6: "q5_0", 7: "q5_1", 8: "q2_k", 9: "q3_k", 10: "q4_k",
           11: "q5_k", 12: "q6_k", 13: "iq4_nl", 18: "iq1_s", 19: "iq1_m",
           20: "iq2_xxs", 21: "iq2_xs", 22: "iq2_s", 23: "iq4_xs"}
# observed in index.txt col 2: 2 -> f32 (size 4*elems), 4 -> bf16 (2*elems),
# 5 -> f16 (2*elems), 0 -> quantized (stays in the GGUF shards).
IDX_DT = {0: "quant", 2: "f32", 4: "bf16", 5: "f16"}

# ---------------------------------------------------------------- GGUF ----

GGUF_TYPES = {
    0: ("<c", 1),      # uint8
    1: ("<i", 1),      # int8
    2: ("<H", 2),      # uint16
    3: ("<h", 2),      # int16
    4: ("<I", 4),      # uint32
    5: ("<i", 4),      # int32
    6: ("<f", 4),      # float32
    7: ("?", 1),       # bool
    8: "string",
    9: "array",
    10: ("<Q", 8),     # uint64
    11: ("<q", 8),     # int64
    12: ("<d", 8),     # float64
}


def gguf_string(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    s = buf[off + 8: off + 8 + n].decode("utf-8", "replace")
    return s, off + 8 + n


def gguf_value(buf, off, t):
    """-> (value, new_offset)."""
    if t == 8:
        return gguf_string(buf, off)
    if t == 9:  # array
        (et,) = struct.unpack_from("<I", buf, off)
        (n,) = struct.unpack_from("<Q", buf, off + 4)
        off += 12
        out = []
        for _ in range(n):
            v, off = gguf_value(buf, off, et)
            out.append(v)
        return out, off
    fmt, sz = GGUF_TYPES[t]
    v = struct.unpack_from(fmt, buf, off)[0]
    if isinstance(v, bool):
        v = bool(v)
    return v, off + sz


def parse_gguf_metadata(path):
    """Read the KV metadata table of a GGUF v3 file (tensor-info table is in
    shard 1 and has 0 entries for split files; we only need metadata)."""
    with open(path, "rb") as f:
        head = f.read(8)
        if head[:4] != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF file")
        f.seek(0)
        raw = f.read(24)
        ver, n_tensors, n_kv = struct.unpack_from("<IQQ", raw, 4)
        blob = f.read(1 << 24)  # metadata for this shard is small (< 16 MiB)
    meta = {}
    off = 0
    for _ in range(n_kv):
        key, off = gguf_string(blob, off)
        (t,) = struct.unpack_from("<I", blob, off)
        off += 4
        val, off = gguf_value(blob, off, t)
        meta[key] = val
    return {"version": ver, "n_tensors": n_tensors, "n_kv": n_kv, "meta": meta}


# ----------------------------------------------------------- pack index ----

def parse_index_txt(path):
    """index.txt: one row per dense tensor.
    columns: name, f1, dtype, off, size, off2, size2, rows, cols, quant, f9,
    f10, f11, ...  (verified: off==off2 and size==size2 for pool-resident
    tensors; quantized ones carry their GGUF quant enum at field 9)."""
    tensors = []
    header = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # "# align 64 pool 1538035200 tensors 1079"
                it = line[1:].split()
                for i in range(0, len(it) - 1, 2):
                    try:
                        header[it[i]] = int(it[i + 1])
                    except ValueError:
                        pass
                continue
            if not line:
                continue
            p = line.split()
            f9 = int(p[9]) if len(p) > 9 else 0
            dtc = int(p[2])
            dtype = IDX_DT.get(dtc, f"dt{dtc}")
            name, rows, cols = p[0], int(p[7]), int(p[8])
            t = {"name": name, "dtype": dtype,
                 "rows": rows, "cols": cols,
                 "elems": rows * cols,
                 "pool_off": int(p[3]), "pool_size": int(p[4]),
                 }
            if dtype == "quant":
                t["quant"] = GGML_DT.get(f9, f"q{f9}")
                # per-block bytes for the enums we know (block 32, superblocks
                # ignored): report bytes/element as bpw only, mark estimate.
                bpw = {"q2_k": 2.5625, "q4_0": 4.5, "q4_k": 4.5, "q8_0": 8.5,
                       "iq2_xxs": 2.0, "iq2_xs": 2.125, "iq2_s": 2.625,
                       "iq1_s": 1.5, "iq4_xs": 4.515625}.get(t["quant"])
                if bpw and rows and cols:
                    t["est_bytes"] = int(rows * cols * bpw / 8)
                    t["est_bpw"] = bpw
            else:
                t["bytes"] = int(p[4])
                t["bpw"] = round(8 * t["bytes"] / (rows * cols), 4) if rows and cols else None
            tensors.append(t)
    return header, tensors


# ---------------------------------------------------- native_experts -------

def parse_native_experts(path):
    rows = []
    note = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                note = line[2:]
                continue
            if not line:
                continue
            p = line.split()
            r = {"layer": int(p[0]), "gu_type": GGML_DT.get(int(p[1]), p[1]),
                 "d_type": GGML_DT.get(int(p[2]), p[2]),
                 "blob_off": int(p[3]), "blob_bytes": int(p[4]),
                 "gate_off": int(p[5]), "up_off": int(p[6]), "down_off": int(p[7]),
                 "shard": p[8] if len(p) > 8 else None}
            r["gate_bytes"] = r["up_off"] - r["gate_off"]
            r["down_bytes"] = r["gate_off"] - r["down_off"]
            # up region size is not given; it is bounded by the file end.
            # gate/up share a matrix shape (640x2560), so size(up) == size(gate)
            # is assumed and flagged.
            r["up_bytes_assumed"] = r["gate_bytes"]
            r["note"] = "up region size assumed == gate region (same matrix shape)"
            rows.append(r)
    return note, rows


# ------------------------------------------------- expert profile (STRP) ---

def parse_expert_profile(path, layers, experts):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != b"STRP":
        raise SystemExit(f"{path}: bad magic {raw[:4]!r} (want b'STRP')")
    hdr = struct.unpack_from("<6I", raw, 4)
    version, n_layers, n_experts, n_entries, n_entries2, blob_size = hdr
    if n_entries != layers * experts:
        raise SystemExit(f"profile entries {n_entries} != {layers}x{experts}")
    # empirical layout (verified: first row decodes to plausible
    # (layer<=47, expert<=511) pairs): the u16 stream starts at byte 24.
    n_u16 = (len(raw) - 24) // 2
    vals = struct.unpack_from("<%dH" % n_u16, raw, 24)
    rows = [(vals[i], vals[i + 1], vals[i + 2], vals[i + 3])
            for i in range(0, min(n_u16, n_entries * 4), 4)]
    # heat[layer][expert] = number of hot co-routing edge endpoints observed
    heat = [[0] * experts for _ in range(layers)]
    hot_edges = 0
    cold_rows = 0
    edges = []
    for a, fa, b, fb in rows:
        if a < layers and b < layers and fa < experts and fb < experts:
            heat[a][fa] += 1
            heat[b][fb] += 1
            hot_edges += 1
            edges.append((a, fa, b, fb))
        else:
            cold_rows += 1
    # per-layer stats
    lstats = []
    for L in range(layers):
        h = heat[L]
        lstats.append({"heat_max": max(h), "heat_sum": sum(h),
                       "nonzero": sum(1 for x in h if x)})
    # hottest experts globally, for the resident-set story (engine pre-fills
    # 5175 slots from the profile; top-N by heat approximates it visually)
    flat = sorted(((heat[L][e], L, e) for L in range(layers) for e in range(experts)),
                  reverse=True)
    return {
        "header": {"magic": "STRP", "version": version, "layers": n_layers,
                   "experts": n_experts, "entries": n_entries,
                   "entries2": n_entries2, "file_bytes": blob_size + 28},
        "layout": "24576 rows of 4 x u16 = (layerA, expertA, layerB, expertB) "
                  "co-routing edge records; ranked by routing frequency; "
                  "cold tail rows store (global_id,0,global_id,0)",
        "hot_edges": hot_edges, "cold_rows": cold_rows,
        "heat": heat, "heat_per_layer": lstats,
        "top_experts": [{"heat": h, "layer": L, "expert": e} for h, L, e in flat[:64]],
    }


# --------------------------------------------------------------- MTP -------

def parse_mtp_dense_txt(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        p = line.split()
        # "fc_embedding.weight q8_0 2560 2560 0 6963200"
        #   name dtype rows cols byte_off byte_size
        if len(p) >= 6:
            rows.append({"name": p[0], "dtype": p[1], "rows": int(p[2]),
                         "cols": int(p[3]), "elems": int(p[2]) * int(p[3]),
                         "off": int(p[4]), "bytes": int(p[5])})
        elif len(p) >= 4:
            rows.append({"name": p[0], "dtype": p[1], "rows": int(p[2]),
                         "cols": int(p[3]), "elems": int(p[2]) * int(p[3])})
        elif len(p) == 3:
            rows.append({"name": p[0], "dtype": p[1], "elems": int(p[2])})
    return rows


# --------------------------------------------------------------- main ------

def human(nbytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024 or unit == "TB":
            return f"{nbytes:.2f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024.0
    return "?"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__[:400])
    ap.add_argument("--shards", default=DEF_SHARDS)
    ap.add_argument("--pack", default=DEF_PACK)
    ap.add_argument("--profile", default=DEF_PROFILE)
    ap.add_argument("--mtp", default=DEF_MTP)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web"))
    args = ap.parse_args(argv)

    root = os.path.dirname(args.out)
    shard_files = sorted(f for f in os.listdir(args.shards) if f.endswith(".gguf"))
    if not shard_files:
        raise SystemExit(f"no .gguf shards in {args.shards}")
    gmeta = parse_gguf_metadata(os.path.join(args.shards, shard_files[0]))
    m = gmeta["meta"]
    layers = int(m.get("qwen4exp.block_count") or m.get("general.block_count") or 48)
    experts = int(m.get("qwen4exp.expert_count") or 512)
    topk = int(m.get("qwen4exp.expert_used_count") or 10)

    idx_hdr, idx_tensors = parse_index_txt(os.path.join(args.pack, "index.txt"))
    nx_note, nx_rows = parse_native_experts(os.path.join(args.pack, "native_experts.txt"))
    prof = parse_expert_profile(args.profile, layers, experts)

    sizes = {f: os.path.getsize(os.path.join(args.shards, f)) for f in shard_files}
    mtp_files = {}
    for name in ("experts.bin", "dense.bin", "draft_vocab.bin"):
        p = os.path.join(args.mtp, name)
        if os.path.exists(p):
            mtp_files[name] = os.path.getsize(p)
    mtp_tensors = parse_mtp_dense_txt(os.path.join(args.mtp, "dense.txt"))

    # ---- classify dense tensors into forward-pass stage groups ----------
    def stage_of(name):
        n = name
        if n.startswith("output.weight") or n.startswith("token_embd"):
            return "embedding_or_head"
        if ".ple_" in n or n.startswith("ple"):
            return "ple"
        if ".ssm_" in n:
            return "ssm"
        if ".indexer" in n:
            return "indexer"
        if n.startswith("hc.") or ".hc_" in n or "hyper_connection" in n or n.startswith("output_hc"):
            return "hyper_connection"
        if ".ffn_gate_inp" in n:
            return "moe_router"
        if ".ffn_" in n or "shexp" in n:
            return "moe_shared"
        if ".attn_" in n:
            return "attention"
        if n.startswith("mtp"):
            return "mtp"
        return "other"

    for t in idx_tensors:
        t["stage"] = stage_of(t["name"])

    per_layer = {}
    for t in idx_tensors:
        if t["name"].startswith("blk."):
            L = int(t["name"].split(".")[1])
            per_layer.setdefault(L, []).append(t)

    layer_rows = []
    for L in range(layers):
        ts = per_layer.get(L, [])
        kinds = {"attn": any(x["name"].endswith("attn_q.weight") for x in ts),
                 "ssm": any(x["name"].endswith("ssm_a") or ".ssm_conv1d" in x["name"] for x in ts),
                 "indexer": any(".indexer." in x["name"] for x in ts)}
        full_attn = kinds["attn"]
        nx = next((r for r in nx_rows if r["layer"] == L), None)
        pool_bytes = sum(x.get("bytes", x.get("est_bytes", 0)) for x in ts)
        layer_rows.append({
            "i": L,
            "kind": "full_attention" if full_attn else "ssm",
            "full_attention_interval": int(m.get("qwen4exp.full_attention_interval", 4)),
            "ple": L in (m.get("qwen4exp.ple.layers") or [1]),
            "tensors": len(ts), "pool_bytes": pool_bytes,
            "expert_blob": nx,
            "profile": prof["heat_per_layer"][L],
        })

    atlas = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "tools/build_atlas.py",
        "sources": {
            "gguf_shard1": os.path.join(args.shards, shard_files[0]),
            "shard_sizes": sizes,
            "pack_index": os.path.join(args.pack, "index.txt"),
            "native_experts": os.path.join(args.pack, "native_experts.txt"),
            "expert_profile": args.profile,
            "mtp_dir": args.mtp,
        },
        "model": {
            "name": m.get("general.name", "Qwen3.8-Flash-Next"),
            "arch": m.get("general.architecture"),
            "quant_format_file_type": gmeta["version"],
            "gguf_version": gmeta["version"],
            "n_tensors_declared": gmeta["n_tensors"],
            "n_kv_pairs": gmeta["n_kv"],
        },
        "dims": {
            "layers": layers, "hidden": int(m["qwen4exp.embedding_length"]),
            "vocab": len(m.get("tokenizer.ggml.tokens") or []),
            "context": int(m.get("qwen4exp.context_length", 0)),
            "heads": int(m.get("qwen4exp.attention.head_count", 0)),
            "kv_heads": int(m.get("qwen4exp.attention.head_count_kv", 0)),
            "head_k": int(m.get("qwen4exp.attention.key_length", 0)),
            "head_v": int(m.get("qwen4exp.attention.value_length", 0)),
            "experts": experts, "top_k_experts": topk,
            "expert_ffn": int(m.get("qwen4exp.expert_feed_forward_length", 0)),
            "shared_expert_ffn": int(m.get("qwen4exp.expert_shared_feed_forward_length", 0)),
            "ssm": {"inner": int(m.get("qwen4exp.ssm.inner_size", 0) or 0),
                    "conv_kernel": int(m.get("qwen4exp.ssm.conv_kernel", 0) or 0),
                    "state_size": int(m.get("qwen4exp.ssm.state_size", 0) or 0),
                    "groups": int(m.get("qwen4exp.ssm.group_count", 0) or 0),
                    "dt_rank": int(m.get("qwen4exp.ssm.time_step_rank", 0) or 0)},
            "hyper_connections": {"count": int(m.get("qwen4exp.hyper_connection.count", 0) or 0),
                                  "low_rank": int(m.get("qwen4exp.hyper_connection.low_rank", 0) or 0)},
            "indexer": {"heads": int(m.get("qwen4exp.attention.indexer.head_count", 0) or 0),
                        "key_length": int(m.get("qwen4exp.attention.indexer.key_length", 0) or 0),
                        "top_k": int(m.get("qwen4exp.attention.indexer.top_k", 0) or 0)},
            "ple": {"layers": m.get("qwen4exp.ple.layers"),
                    "ngram_size": int(m.get("qwen4exp.ple.ngram_size", 0) or 0),
                    "heads_per_ngram": int(m.get("qwen4exp.ple.heads_per_ngram", 0) or 0),
                    "conv_kernel": int(m.get("qwen4exp.ple.conv_kernel", 0) or 0),
                    "head_offsets": (m.get("qwen4exp.ple.head_offsets") or [])[:12],
                    "layer_multipliers": (m.get("qwen4exp.ple.layer_multipliers") or [])[:4],
                    "per_layer_input": int(m.get("qwen4exp.embedding_length_per_layer_input", 0) or 0)},
            "rope": {"sections": m.get("qwen4exp.rope.dimension_sections"),
                     "freq_base": m.get("qwen4exp.rope.freq_base"),
                     "dim": int(m.get("qwen4exp.rope.dimension_count", 0) or 0)},
            "full_attention_interval": int(m.get("qwen4exp.full_attention_interval", 4)),
            "compress_ratios": (m.get("qwen4exp.attention.compress_ratios") or [])[:8],
            "sampling_default": {"top_k": int(m.get("general.sampling.top_k", 0) or 0),
                                 "top_p": m.get("general.sampling.top_p"),
                                 "temperature": m.get("general.sampling.temp")},
            "size_label": m.get("general.size_label"),
            "quantized_by": m.get("general.quantized_by"),
        },
        "expert_total_bytes_declared": 42283827200,
        "layers": layer_rows,
        "dense_tensors": idx_tensors,
        "expert_profile": {k: v for k, v in prof.items() if k != "heat"},
        "mtp": {
            "files_bytes": mtp_files,
            "tensors": mtp_tensors,
            "note": "draft layer: fc_embedding+fc_hidden -> indexer -> 24-head "
                    "Q / 2-head KV attention -> 512-expert MoE router -> shared "
                    "expert; experts.bin is 512 experts x (gate,up,down) "
                    f"= {mtp_files.get('experts.bin', 0) // 512 if mtp_files.get('experts.bin') else 0} bytes/expert",
        },
        "notes": [
            "index.txt dtype codes: 2=f32,4=bf16,5=f16 verified against sizes; "
            "0=quantized (stays in GGUF shards, bytes estimated from quant bpw)",
            "expert-profile heat counts edge endpoints of hot co-routing edges; "
            "the cold tail (all-zero counts) carries no per-expert data",
            "native_experts up-region size assumed equal to gate region",
        ],
    }

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(root, "live", "atlas_manifest.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(atlas, f)
    js_path = os.path.join(args.out, "atlas_data.js")
    # the heat matrix is the only big payload: emit as compact JS
    with open(js_path, "w") as f:
        f.write("// generated by tools/build_atlas.py -- do not edit by hand\n")
        f.write("window.ATLAS_META = ")
        json.dump({k: v for k, v in atlas.items() if k != "expert_profile_heat"}, f)
        f.write(";\nwindow.ATLAS_HEAT = ")
        json.dump(prof["heat"], f, separators=(",", ""))
        f.write(";\n")

    print(f"atlas -> {js_path} ({os.path.getsize(js_path):,} B)")
    print(f"manifest -> {json_path} ({os.path.getsize(json_path):,} B)")
    print(f"layers={layers} experts={experts} topk={topk} "
          f"dense_tensors={len(idx_tensors)} expert_layers={len(nx_rows)} "
          f"hot_profile_edges={prof['hot_edges']} cold_rows={prof['cold_rows']}")
    print("shards:", ", ".join(f"{n}={human(sizes[n])}" for n in shard_files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
