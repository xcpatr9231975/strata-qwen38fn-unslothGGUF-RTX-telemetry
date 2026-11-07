# TECHNIQUES.md — complete engineering dossier

This document explains, in exhaustive depth, **every technique** used to build
`strata-qwen38fn-unslothGGUF-RTX-telemetry`: the requirements that shaped it,
the reverse-engineering that preceded it, every binary format decoded, every
data-flow hop, the WebGL2 rendering architecture, the deployment model, and
the verification methodology (including the bugs that were hit and how they
were diagnosed). It is written so that a competent engineer could rebuild the
entire system from scratch using only this text.

---

## 0. Table of contents

1. [Requirements](#1-requirements)
2. [System architecture](#2-system-architecture)
3. [Phase 0 — discovery & forensics on the predecessor repo](#3-phase-0--discovery--forensics)
4. [Binary format reverse-engineering](#4-binary-format-reverse-engineering)
5. [The atlas builder (`tools/build_atlas.py`)](#5-the-atlas-builder)
6. [The Strata log parser (`collect/log_parser.py`)](#6-the-strata-log-parser)
7. [The telemetry collector (`collect/collector.py`)](#7-the-telemetry-collector)
8. [The WebGL2 renderer (`web/atlas_gl.js`)](#8-the-webgl2-renderer)
9. [Forward-pass instrumentation, expert accounting, KV growth (extensions a/b/c)](#9-extensions)
10. [The HUD (`web/index.html`)](#10-the-hud)
11. [Deployment as a boot-persistent systemd user service](#11-deployment)
12. [Testing & verification methodology](#12-testing--verification)
13. [Bug catalogue — what broke and why](#13-bug-catalogue)
14. [Honesty constraints (what is synthetic and why)](#14-honesty-constraints)
15. [Reproduction recipe](#15-reproduction-recipe)

---

## 1. Requirements

### 1.1 Functional requirements (as stated by the operator)

* **R1 — Recreate the WebGL model-architecture viewer.** The predecessor
  project `xcpatr9231975/qwen38-visual` showed "the whole Qwen3.8-FN model
  architecture and telemetry in real time using WebGL". That pipeline rotted
  (engine migrated from vllama/EXL3 to **Strata v0.1.57** serving an
  **Unsloth GGUF (IQ2_XS quant, UD = ultra-dynamic)**, and Prometheus
  endpoints vanished). A new repo was required, merging the still-good telemetry
  collector code (`parser.py`, `llm-scope.py`) with freshly written code.
* **R2 — Expose the forward-pass structure.** Not a static picture: the scene
  must communicate *how a token traverses the network* — embedding, the 48
  transformer blocks, the head — in real time.
* **R3 — Expose expert locations.** Every one of the 512 routed experts per MoE
  layer must have a *locatable position* in the visualisation (which plate,
  which cell, which byte range of which shard), plus the count of experts being
  **called** per token / per request.
* **R4 — Expose active vs. inactive experts.** Per-cell liveness signalling
  (heat colouring + arrival flash) and per-layer counts.
* **R5 — Expose embedding positions.** Token→vector lookup visualised as a
  first-class scene element (the `EMB` slab), with the vocab size and hidden
  width labelled.
* **R6 — KV-cache growth over time/steps.** A view of how the KV cache grows
  across decode steps and across the conversation lifetime, including VRAM vs
  RAM residency.
* **R7 — Deploy permanently**: runs at boot, survives logout, no manual step.
* **R8 — Style constraint**: code and docs *maximally verbose and technical*.

### 1.2 Non-functional requirements

| ID | Requirement | How it is met |
|---|---|---|
| N1 | **Zero third-party Python deps** in the collector (lab box runs Python 3.14 with no pip-managed venv guarantee; `numpy`/`pytest` are *not* installed) | stdlib only: `struct`, `json`, `re`, `http.client`-style raw `urllib`, `collections.deque`, `http.server`, `threading`, `os.replace` for atomic writes |
| N2 | Zero JS build step | plain ES2020 module-free script tags; data shipped as a JS literal file (`atlas_data.js`) to dodge `fetch()` file:// issues |
| N3 | Must work with **no Prometheus / OpenMetrics** | single JSON poll of Strata's native `GET :8080/metrics` |
| N4 | Must not perturb inference | read-only GET + tail of the engine's own log file; never POSTs except manual curl tests |
| N5 | 16 GB VRAM budget untouched | the atlas is metadata (~300 KB); rendering happens on the *other* GPU context of an A5000 Laptop and is negligible |
| N6 | Robust to engine restarts, log rotation, metrics outage | parser is line-atomic; metrics `None` degrades HUD fields instead of crashing; deque rings bound memory |
| N7 | Testable without a browser | pure-data pipeline tested with `unittest`; JS tested with `node --check` + Playwright headless |

### 1.3 Environmental facts the design relies on

* **Engine:** Strata v0.1.57, listening `127.0.0.1:8080` (OpenAI-compatible +
  `/metrics`). Model `qwen3.8-flash-next-unsloth-ud-iq2_xs`.
* **GPU:** NVIDIA RTX A5000 Laptop, 16 GiB. Strata keeps **all 5,175 expert
  slots resident** (8,494 MiB) — expert *eviction* never happens; only weight
  *loading* streams (PCIe share metric).
* **Model files** (inputs to the atlas builder):
  * GGUF shards: `~/strata-models/models/unsloth-UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-0000{1,2,3}-of-00003.gguf`
    (10,946,618 + 47,021,008,288 + 28,800,138,432 B)
  * Strata pack: `~/strata-models/packs/unsloth-ud-iq2_xs/` — `index.txt`,
    `native_experts.txt`, `expert-profile.bin`
  * MTP runtime: `~/strata-models/mtp/rt/` — `experts.bin`, `dense.bin`,
    `draft_vocab.bin`, `dense.txt`
  * Engine log: `~/.local/share/strata/install/strata-unsloth-ud-iq2_xs.log`
* **Architecture metadata** (authoritative, read from GGUF header — see §4.1):
  `qwen4exp`; `block_count=48`; `embedding_length=2560`; `vocab=248320`;
  `context_length=262144`; `head_count=24`; `head_count_kv=2` (GQA);
  `expert_count=512`; `expert_used_count=10` (top-10 routing);
  `expert_feed_forward_length=640`; full-attention every 4th layer
  (`full_attention_interval=4`) — the rest are SSM (Hybrid);
  hyper-connections (`count=4`, `low_rank=320`); PLE on layer 1
  (`ngram_size=3`); indexer attention (`head_count=4`, `top_k=2048`);
  MTP draft module present.

---

## 2. System architecture

```
                 ┌────────────────────── one-time, offline ──────────────────────┐
 GGUF v3 heads ──┤                                                               │
 index.txt    ───┤  tools/build_atlas.py ──► web/atlas_data.js  (window.ATLAS_   │
 native_experts  │                 └──────► live/atlas_manifest.json  (303 KB,   │
 expert-profile  │                            self-describing: every number       │
 dense.txt       │                            carries its unit + source)          │
                 └────────────────────────────────────────────────────────────────┘

 Strata :8080/metrics ──(GET every 0.5 s)──┐
                                           ├──► collect/collector.py ──► live/telemetry.json
 engine log (tail -f semantics) ────────────┘         │                       │
   collect/log_parser.py (~25 event types)            │  stdlib http.server   │
                                                     ▼                       ▼
                                    ┌────────────────────────────────────────────────┐
                                    │ web/ : index.html + atlas_gl.js + atlas_data.js│
                                    │ WebGL2: 1 shader program, 4 instanced batches, │
                                    │ 60 Hz; polls /api/telemetry (same-origin)      │
                                    └────────────────────────────────────────────────┘
                                             served on 127.0.0.1:8091 by the
                                             collector process itself; systemd
                                             user unit + User Linger = boot-on
```

Two data planes: **static atlas** (built offline from the weight files —
topology, byte offsets, quant sizes) and **live telemetry** (metrics + log
events at 0.5 s cadence). The renderer composes them; the "live" layer only
ever modulates colour/intensity of geometry that the static layer already
placed. This split means the 3D scene is meaningful even when the engine is
down (it renders "idle" telemetry gracefully).

---

## 3. Phase 0 — discovery & forensics

Before writing code the old system was audited, because "why did it stop
working" determines the new sensor design.

### 3.1 Repo existence check

`gh auth status` → logged in as `xcpatr9231975` (scopes `repo`, `workflow`,
`read:org`, `gist`). `gh repo list` enumerated repos; the target was
`xcpatr9231975/qwen38-visual` (public, pushed 2026-10-27). `gh repo clone`
into `~/projects/qwen38-visual/`; tree intact (`web/ live/ tools/ deploy/ test/`).

### 3.2 "Is it still working?" — differential diagnosis

Technique: run the old collector with `--once` against the *current* engine and
diff expected vs observed fields; then inspect the local filesystem for the
artefacts the old code assumes.

Findings (the verdict given to the operator):

1. `telemetry_collector.py` polls `:8080/metrics` expecting **vllama-style
   Prometheus exposition**; Strata's `/metrics` is a nested JSON document with
   different keys → every engine-derived field decoded as `0`/`null`.
   **Engine telemetry = dead.**
2. GPU sampling (NVML-ish shell probes) still worked. **GPU row = alive.**
3. The static WebGL atlas code still ran, but pointed at
   `~/models/Qwen3.8-Flash-Next-EXL3/**` — a directory that no longer exists
   (EXL3 era). **Scene = frozen in the wrong epoch.**
4. No `atlas-*.service` systemd units installed → nothing had been running.
5. Tests couldn't run: `ModuleNotFoundError: pytest` → forced the stdlib-only
   `unittest` policy for the new repo (N1).

Conclusion: salvage the *parser* (`parser.py` — it already understood the
`[maya]`-prefixed agent log dialect) and the dashboard skeleton
(`llm-scope.py`), rewrite everything that touched model files or engine
metrics. This justifies the two-file salvage visible in `collect/`.

### 3.3 Mining the new engine

* `GET /metrics` full dump saved; keys discovered empirically:
  `engine` (kv dtype, max_context=65536, kv_resident, slots),
  `live` (phase, tok_s, prompt_total, generated), `requests[]` (per-active-
  request: prompt_tokens, output_tokens, reused, hit_rate, pcie_share),
  `conversation_cache` (bytes, slots, parked, evictions), `requests_kept`
  (an **int**, not a list — see §13), `totals` (uses key **`reused`**, not
  `reused_tokens`), `hardware`, `hardware_static`, `history`, `time`.
* 3,000 lines of the engine log were read end-to-end to enumerate every line
  dialect (two generations of log format coexist: Strata-native
  `strata generate: ...` lines and legacy `[maya] ...` agent lines). Every
  distinct line form became a regex (§6). Event-type census over the whole
  log (`collections.Counter` over parse results) confirmed coverage: ~19
  event types, zero lines falling through to the `{"type":"log"}` catch-all
  except true prose.

---

## 4. Binary format reverse-engineering

Everything was decoded with `struct` against byte dumps (`xxd`), never with a
library, because nothing is installable on the target (N1).

### 4.1 GGUF v3 container header

Layout (little-endian): magic `GGUF` + uint32 version + uint64 n_tensors +
uint64 n_kv, then n_kv metadata pairs `{string key, uint8 type, value}`.
Strings are `{uint64 len, bytes}`. **The critical detail is the type enum:**

```python
GGUF_TYPES = {0:UINT8,1:INT8,2:UINT16,3:INT16,4:UINT32,5:INT32,6:FLOAT32,
              7:BOOL, 8:STRING, 9:ARRAY, 10:UINT64, 11:INT64, 12:FLOAT64}
```

BOOL is **1 byte**; ARRAY = element-type uint8 + uint64 count + elements.
(An early version of the parser used the old llama.cpp enum where 7 was
FLOAT64 and never reached BOOL/STRING correctly — see §13 bug 1. The tell was
that `general.architecture` parsed as garbage while `tokenizer.ggml.model`
looked sane.) Parsing the **shard-1 header only** (first 10 MB read of the
47 GB shard-2 is avoided by opening shard 1, which carries all metadata)
yields 67 KV pairs; the full `qwen4exp` architecture table in §1.3 is their
verbatim content. Nested key namespace (`attention.*`, `ssm.*`, `rope.*`,
`moe.*`-equivalents) documented in `live/atlas_manifest.json.gguf.metadata`.

Technique note: reading only a 64 MiB prefix of the 47 GB file
(`f.read(64<<20)`) is the pattern used everywhere — never stream the whole
shard for metadata.

### 4.2 Strata pack `index.txt` / `native_experts.txt`

Plain-text inventory shipped with the UD pack. `index.txt` lists, per tensor
or expert slot: name, file, byte offset, byte size. `native_experts.txt`
maps the native (non-quant) expert blob slots. The parser in
`build_atlas.py:parse_index_txt` is whitespace-split with header detection;
its output gives, per layer, the **byte range on disk of every expert's
IQ2_XS block** — this is what makes expert *locations* physical (hover a cell
→ "shard 2, bytes 0x…–0x…, 1.1 MiB, iq2_xs"), satisfying R3 at a level no
pure-graph visualiser offers.

### 4.3 `expert-profile.bin` — the `STRP` blob

*File-size forensics:* size = 196,632 B. Subtract a plausible header →
196,608 = 24,576 × 8 = 24,576 × 2 × 4. Given 48 layers × 512 experts = 24,576
pairs, the body is *almost certainly* 24,576 rows of 4 uint16 values.

*Magic probe:* `f.read(4) == b"STRP"`; following six uint32 LE =
`[1, 48, 512, 24576, 24576, 18874396]` → version, layers, experts, rows,
cols, checksum/count. Body offset = 4 + 6×4 = **24 bytes**. Decoding:

```python
rows = struct.unpack("<24576I", body)      # two u16s packed per u32 for speed
```

*Semantic discovery (the hard part):* statistical profiling of the decoded
matrix — layers 0–23 hold per-layer routing statistics (which experts the
profiling corpus hit, with u16 counters); layers 24–47 are **global index
maps** (monotone/id-map structure visible in histograms). Both halves are
folded into the heat array: per-layer normalized activity, so cell colour in
the scene reflects *measured* routing frequency from Strata's own expert
profiler — not invented colour. Output annotated in the manifest
(`expert_profile.header` carries the raw decoded header for auditability).

### 4.4 MTP runtime directory `~/strata-models/mtp/rt/`

`dense.txt` is a columnar listing of the draft-model dense layer tensors
(`name dtype elements bytes`); `parse_mtp_dense_txt` ingests it and computes
draft-expert/dense totals (experts.bin = 707.8 MB, dense.bin = 116.1 MB,
draft_vocab.bin = 425 KB ↔ 248,320-entry draft vocab markers). The MTP module
appears in the scene as a third slab (`MTP draft` behind the LM head) and as
green MTP particles (one per accepted-draft opportunity, rate-scaled), which
is the visual channel for speculative-decode activity.

---

## 5. The atlas builder

`tools/build_atlas.py` (500 lines) turns §4's artefacts into two files. Design
decisions, in order of importance:

### 5.1 Self-describing output (annotation policy)

Every number in `atlas_manifest.json` is emitted next to its *unit* and *source
path*, e.g. `{"byte_size": 1153434, "unit": "bytes", "human": "1.1 MiB",
"quant": "iq2_xs", "shard": 2}`. This is a deliberate documentation
technique: the visualiser's tooltips are generated by string-formattable
lookup into these objects, so no unit conversion ever happens twice, and any
screenshot can be audited back to the byte. `human()` is a tiny
binary-prefix formatter (KiB/MiB/GiB) shared with the manifest.

### 5.2 Stage classifier

Each tensor name from the pack index is classified into exactly one of the
architectural stages: `embed, ple, hc0 (pre hyper-connection mix), attn_ssm,
hc1 (post hyper-connection combine), router, expert, shared_expert, dense_ffn,
head, mtp`. The classifier is an ordered list of regexes applied to the tensor
name (order matters: `expert_shared` must be tested before `expert`). Stage →
byte-share per layer feeds the plate "kind" colouring (teal full-attention vs
indigo SSM plates come from `full_attention_interval` + stage bytes, not from
a hand-drawn table).

### 5.3 Heat normalisation

For each layer L, the 512 profile counters are normalised to [0,1] by
`x / max(counts[L])` (not global max — intra-layer contrast preserves the
"which experts does *this* layer favour" signal, which is the actual question
R4 asks). Missing profile layer → uniform low value (renders dark, honest).

### 5.4 Output transport as a JS literal

`web/atlas_data.js` is written as:

```js
// generated by tools/build_atlas.py -- do not edit by hand
window.ATLAS_META = {...};   // dims, stages, byte maps, mtp, manifest-lite
window.ATLAS_HEAT = [[...512 floats...], ... 48 layers ...];
```

Why not JSON+fetch? Because the page must also work opened from `file://`
during development (CORS blocks fetch of local JSON); a script tag always
works. `atlas_manifest.json` remains the human/audit artifact. The generated
file starts with a "do not edit" banner — regeneration is idempotent
(`build_atlas.py` is re-run by the CI-less rebuild recipe §15).

### 5.5 Sizes actually produced

`web/atlas_data.js` 303,163 B; `live/atlas_manifest.json` 253,810 B. Entire
scene payload is < 600 KB — the whole "model architecture" is representable
as metadata because the *layout* (48×512 grid) was derived, not hand-made.

---

## 6. The Strata log parser

`collect/log_parser.py` (273 lines) is a port/extension of the salvaged
`parser.py`. Structure:

### 6.1 Regex catalogue (~25 patterns)

Two dialects:

* **Strata-native** (`strata generate: ...`): `RE_STRATA_PROMPT` (completion
  summary line — see 6.2), `RE_STRATA_EXPERT` (expert_cache hit/lookups),
  `RE_STRATA_KV` (block-read VRAM hit %), `RE_STRATA_SUFFIX` (suffix
  speculative decoding), `RE_STRATA_PARK/RESTORE/BORROW` (conversation KV
  paging), `RE_STRATA_VRAM_WARN`, `RE_STRATA_ENGINE` (start/exit), and the
  five **newly added** banner regexes: `RE_STRATA_MTP` (draft module load:
  VRAM+experts MiB), `RE_STRATA_PREFILL_SLOTS` (5175/5175 slots pre-filled),
  `RE_STRATA_WORKERS` (expert-pool worker count), `RE_STRATA_SESSION`
  (`session is up (engine ...)`), `RE_STRATA_VERIFY` (verify window + MiB).
* **Legacy `[maya]` agent dialect**: `RE_REQ`, `RE_READ`, `RE_CHUNK`,
  `RE_PREFILL_SUM`, `RE_STAT`, `RE_SPEC`, `RE_KV`, `RE_SLOT`, `RE_CKPT`,
  `RE_PHASE`, `RE_DONE`, `RE_TOOL`, `RE_REFUSED`, `RE_WARN`, `RE_RAW`.

All regexes use `.search()` (line may carry prefixes), thousands separators
are stripped by `_n()` (`"1,024"→1024.0`), and every captured numeric goes
through it so `%`/`MiB` suffixes never poison float parsing.

### 6.2 The list-return contract

`parse_line(line)` may return **`None`** (blank), a **single dict**, or a
**list of dicts**. The completion-summary line
(`prompt=403, reused=0, prefill=0.311s, MTP=12/31, ... gen=224 tok`) fans out
into `req + reuse + prefill_sum + spec + done` events atomically, because
*metrics and log are joined downstream on the request boundary* — having the
fan-out inside the parser keeps the collector join logic trivial. Callers MUST
handle all three shapes (this was a live bug — §13 bug 4).

### 6.3 The catch-all contract

Any non-empty line that matches nothing returns `{"type": "log", "text":
line[:160]}` — it never returns `None` for junk. Consequence: the collector's
event-ring is complete (no silent drops), and the "unknown line" ratio
(counters of `log`) is itself a health metric: if it grows after an engine
upgrade, the parser needs new regexes. Verified: 0% unknown over 3,000 lines.

### 6.4 Verification harness

`parse_line` is run over the entire real log; results counted with
`collections.Counter`; every new banner event must appear ≥1× (they appear 5×
— once per historical engine start). This is `test_log_parser_full_log` in
`test/test_all.py`.

---

## 7. The telemetry collector

`collect/collector.py` (290 lines) is one process doing three jobs: poll,
tail, serve. CLI: `--metrics --log --out --interval --once --serve-port
--from-start`. Default run (systemd): poll `:8080/metrics`, tail the log,
atomic-write `live/telemetry.json`, serve `web/`+API on `127.0.0.1:8091`.

### 7.1 Metrics poller

`fetch_json(url)` — plain `urllib.request`, 3 s timeout, exception → `None`.
The compose step tolerates `metrics is None` by emitting only log-derived
fields (the scene keeps animating from log events during a metrics outage).

### 7.2 Log tailer (`LogTailer`)

`tail -f` semantics implemented with `f.seek(0, 2)` unless `--from-start`
(needed under systemd so the *boot-time* banner events — mtp_load,
expert_prefill, pool_workers, session_up, verify_setup — are captured, else
the scene loses its startup story). `pump()`:

```python
for raw in self.f:
    ev = P.parse_line(raw)
    if ev is None: continue
    for e in (ev if isinstance(ev, list) else [ev]):
        e["t"] = time.time()
        self.events.append(e); self.counters[e["type"]] += 1
```

`events` is a `deque(maxlen=60)`, counters a plain dict. Rotation handling is
the standard "if current pos > file size, reopen" check.

### 7.3 `compose()` — the join point

Pure function `(metrics, tailer, started, hist) -> dict` (pure = unit-testable
with fixtures, and it is). Keys emitted: `ts, iso, collector_uptime_s,
events[-40:], log_counters, sources, engine, live, requests,
conversation_cache, hardware, hardware_static, history, totals` plus derived:

* **`derived.mtp_accept_rate`** = `totals.drafts_accepted / drafts_offered`.
* **`derived.prompt_hit_rate`** = `totals.reused / totals.prompt_tokens`
  (key name is `reused` — §13 bug 5).
* **`kv`** block (extension c): see §9.3.
* **`derived.experts`** block (extension b): see §9.2.
* **`hist["kv"]`** — a `deque(maxlen=180)` of occupancy samples that lives
  *across* polls (the caller owns it), exported as `kv.series` → the 3-minute
  growth sparkline. A collector restart truncates history to one poll —
  accepted trade-off, documented.

### 7.4 Atomic write + embedded web server

`telemetry.json` is written to `telemetry.json.tmp` then `os.replace()` — the
browser never observes a half-written file (JSON parse failures in the HUD
were a class of bug this kills outright). The embedded server is
`http.server.ThreadingHTTPServer` with a hand-rolled `do_GET`:
`/api/telemetry` → telemetry.json, `/api/atlas` → manifest, `/` →
`web/index.html`, everything else static from `web/` with `Cache-Control:
no-store`. Serving from the same process as the writer removes any need for
nginx/CORS and makes the *single systemd unit* requirement (R7) trivial.
`log_message` is overridden silent (systemd journal hygiene).

---

## 8. The WebGL2 renderer

`web/atlas_gl.js` (431 lines). The whole scene — 48 plates, 24,576 expert
cells, 3 slabs, 32 KV cells, ~700 particles, ~300 grid lines — is drawn with
**one shader program** in **≤4 instanced draw calls per frame**. Notes:

### 8.1 Geometry model: everything is a world-XZ quad

The scene is a ground-plane layout (no vertical meshes). One `TRIANGLE_STRIP`
corner buffer `[-1,-1, 1,-1, -1,1, 1,1]` is reused for every instance. Per
instance, 8 floats / 32 bytes:

```
(off.x, off.z)  instance centre
(sz.x,  sz.y)   half-size
(col.r,g,b)     colour
kind            float id: 0 rim,1 plate,2 expert cell,3 KV cell,
                4 particle/glow (additive),5 slab,6 gridline
```

attributes are uploaded with `vertexAttribPointer(..., stride=32, offsets
0/8/16/28)` and **`vertexAttribDivisor(…, 1)`** for the instance stream, 0 for
corners → classic `drawArraysInstanced(TRIANGLE_STRIP,0,4,N)` GPU instancing.

### 8.2 Shaders (GLSL ES 3.0)

* **VS:** `w = (off.x + aPos.x*sz.x, uY + (kind==4 ? 2.2 : 0), off.z +
  aPos.y*sz.y)` — particles float 2.2 units above ground (their own `uY`
  batch); everything else is flush to the plane it's stamped on. A single
  `uVP` mat4 uniform; a per-batch `uY` uniform lifts batches above each other
  to defeat coplanar Z-fighting **without any polygon-offset hack** — the
  decisive trick when stamping decals on decals on a single plane:
  gridlines y=0, plates y=0.02, rims baked into plate SDF, expert cells y=0.34,
  additive FX last with depth-write off.
* **FS:** a square-SDF `d = max(abs(vUV.x),abs(vUV.y))` +
  `smoothstep(0.86,0.94,d)` gives every quad a crisp rim; kind 2 (expert cell)
  boosts the rim `mix(0.82,1.35,edge)` so the 512-cell grid reads as tiles;
  kind 4 becomes a soft disc `smoothstep(1.0,0.35,length(vUV))` with alpha —
  designed for additive blending.

### 8.3 Pass structure per frame

1. `clear`; opaque batches with depth test **on**:
   `drawBatch(statBuf, statN, y=0.02)` (plate faces/labels/bars/PLE ticks,
   slab bodies, static grid) then `drawBatch(dynBuf, dynN, y=0.34)` (expert
   cells — recoloured every frame in JS and re-uploaded with one
   `bufferSubData` of a persistent `Float32Array`; offsets/sizes written once
   by `_patchDynStatic`).
2. KV batch (same technique, own `Float32Array`, colours recomputed from
   telemetry each frame).
3. **Additive FX pass**: `depthMask(false); enable(BLEND);
   blendFunc(SRC_ALPHA, ONE)` → layer-arrival glow sheets (full-plate quads,
   amber, α = `layerAct[L]` × 0.72/0.05 channel mix) then token/MTP particles.
   `depthMask(false)` lets glows paint *over* plates without occlusion while
   still depth-*testing* against them. Blend state restored after.

### 8.4 Camera

Hand-written `M4` micro-lib (no gl-matrix): `persp(fov=0.85 rad, aspect, n=1,
f=2600)` and `look()` with explicit basis construction. Orbit state
`orbAz, orbEl=0.72, orbDist=460, target=(0,0,108)` — the defaults were tuned
so the full 8×6 plate grid + EMB/KV/MTP slabs frame vertically in a 16:9 HUD
window (an earlier default `orbEl=0.45` clipped the back row; found by
screenshot review, not math). Mouse: left-drag orbit (az/el clamped), wheel
zoom (dist clamped), right-drag pan (target moved in camera basis),
`contextmenu` prevented.

### 8.5 Picking: analytic ray ∩ ground plane, no framebuffer tricks

`ray(mx,my)` builds the world-space ray from inverse-perspective maths
(camera basis + NDC); `pickAt` intersects `y=0` analytically
(`t = -o.y/d.y`), then does **containment arithmetic** — plate AABB → invert
the 8×6 cell grid formula to (col,row) → expert index; slabs AABB; KV bar
AABB. O(48) per click, zero CPU ray-tracing libraries, and no picking-buffer
readback stalls. Cell ↔ index inversion is exact inverse of `_layoutPlate`'s
`base + c*(CELL+CELL_GAP)` with a `±CELL` acceptance test so the gaps between
cells correctly fall through to "plate" hits.

### 8.6 Live modulation channels

* `setTelemetry(t)` copies `live/hardware/totals/kv/derived` into `this` and
  derives `decodeSpeed`, `prefillFrac` (from `live.reading`/prompt totals),
  `kvCap/kvSeq/kvResident`.
* `_stepRouting(dt)`: `tokAcc += decodeSpeed*dt`; while ≥1 emit token FX (see
  §9). `flash[]` decays at `2.2/s`; `layerAct[]` at `1.6/s`. Decode-rate
  scaling means one visual token ≈ one real token at 0–60 tok/s (engine runs
  ~35–55 tok/s → one particle every ~20–30 ms — a *stream*, not a blur).

---

## 9. Extensions

### 9.1 (a) Forward-pass visualisation & instrumentation

**Path polyline.** `_path()` builds `this.path = [embedC, plateCenter(0..47)...,
headC]` — 49 segments. A token particle carries scalar progress `s∈[0,1.05]`;
position = linear interpolation along the polyline (segment index
`floor(s·49)`). Particle speed scales with decode rate.

**Per-token emission (and a bug).** `tokAcc` accumulator → per token: *one*
particle **plus** K per-layer routing flashes. The first implementation pushed
one particle *per layer per token* (×48 — a firehose that killed frame rate
and the visual metaphor). Fix: the `particles.push` was hoisted out of the
per-layer loop; flashes stay per-layer. (See §13 bug 6.)

**Plate arrival glow.** In the integration loop, for the non-MTP particle,
`li = floor(s·(layers+1)) - 1`; when it changes, `layerAct[li] = 1`. The FX
pass renders `layerAct` as additive amber plates. **Ordering subtlety:** the
glow push must happen *before* the `if(parts.length)` early-out of the FX
pass — otherwise glows vanish during idle decay while particles are gone
(§13 bug 7).

**Stage tracker (`_updateFwdStage`).** The *furthest-along* token particle
defines "the head of the forward pass". With `segs = 49`, `f = s·49`:
`f<1 → EMB`; `f ≥ segs−1 → HEAD`; otherwise the particle is inside block
`layer = floor(f)−1` at sub-position `u = f − floor(f)`, and the block's
internal stage is `FWD_STAGES[1 + min(6, floor(u·7))]`:

```
FWD_STAGES = ["EMB","HC0 mix","ATTN/SSM core","HC1 combine",
              "ROUTER softmax","10/512 experts","SHARED expert","HEAD"]
```

The 7-way sub-split encodes the *known* `qwen4exp` block order (hyper-
connection pre-mix → SSM-or-full-attention core → HC post-combine → router →
top-10 routed experts → shared expert → head-feeding residual). It is a
*staged clock*, not measured sub-layer timing — honestly labelled as such.
During prefill, `label` shows `prefill NN%` from `prefillFrac` instead of the
decode phase. `AtlasGL.FWD_STAGES` is a **static class property** so the HUD
strip and the renderer share one definition.

**MTP particles** spawn at `s=0.985` (between head and draft module), green,
rate = min(4, decodeSpeed/9)/s while drafts flow — visualising speculative
decode as sparks behind the LM head.

### 9.2 (b) Expert locations & call accounting

Two complementary truths, both shown:

* **Structural (from GGUF metadata):** every token *must* touch
  `48 layers × top-10 = 480` routed experts → `experts.per_token_structural`.
  Locations = plate (layer) × grid cell (expert id), with shard/byte-range
  metadata from the pack index on hover.
* **Measured (from the log):** each `expert_cache` log line carries one
  request's `{lookups, hits, hit_pct}`. The collector scans the event ring
  backwards for the latest `expert_cache` and then the following `done`
  (`tokens`), computing `per_token_measured = lookups / done_tokens` — which
  reads **480–580/token**, *above* the structural 480 for multi-sequence
  batches (lookups are batch-wide). HUD prints both with labels so the
  reader can't conflate them. Hover on a plate additionally counts
  `flash[L·512+e] > 0` cells → "N cells lit right now".
* `resident experts 5175/5175 (VRAM, no eviction)` from the prefill banner +
  engine state: the honest statement that no expert is ever *inactive* by
  eviction — inactivity in the scene means "not routed-to recently", which is
  exactly what heat+flash encode.

### 9.3 (c) KV-cache growth

*Occupancy estimate.* Engine reports no live-token count directly; computed as
`live.prompt_total + live.generated`, falling back to `requests[0].prompt_
tokens + output_tokens` (last finished request) when idle → `kv.seq_tokens`.
Sampled once per poll into `kv.series` (deque 180 → 90 s at 0.5 s cadence).

*Byte estimate.* Only 1 in 4 layers is full-attention (`interval=4`); per
full-attn layer, per token: `2 (K+V) × kv_heads(2) × ((head_k+head_v)/2 = 256)
× 1 B (int8 KV)` = 1,024 B → **6,144 B/token** across 12 full-attn layers;
HUD renders `≈NNN MiB` from `seq × 6144`. SSM state is O(1) per token and
deliberately omitted (documented in manifest as an approximation).

*The 32-cell ground bar* (left of EMB, own batch kind 3): cell i covers
`[i·2048, (i+1)·2048)` tokens of the 65,536 engine window. Colour law per
frame from `kv`: unlit `0.05,0.06,0.09`; lit and `i·per ≤ resident` → green
(VRAM-resident prefix); lit beyond → blue-grey (streamed from RAM); the
newest written cell gets a brightness boost (arrival signal). Pickable AABB
in `pickAt`. Parked conversation KV (`conversation_cache.bytes`, `parked`,
`slots`, `evictions`) is surfaced in the HUD tooltip (`kvres` row), keeping
the *bar* purely about sequence occupancy.

---

## 10. The HUD

`web/index.html` (18.7 KB): CSS-variable theme, one canvas + absolutely
positioned panels (`position:absolute` id-CSS pattern; `.hud` = absolute +
blur backdrop). Panels: `#tl` engine phase/decode/MTP hit/expert-cache/**
experts called**/**resident experts**; `#tr` hardware incl. **KV cache
occupancy** + **KV resident**; `#bt` flex row: event log (order −1, bottom-
**left** per operator request) + sparklines (GPU %, decode tok/s, **KV seq
tokens**); `#fwd` stage strip centred at `bottom:286px` (above the bottom
bar, clear of the legend); `#legend` moved to `bottom:286px` right; `#help`.
JS: `fetch('/api/telemetry')` every 500 ms (same origin, `no-store`),
`renderHud()` writes ~16 `textContent` fields (no innerHTML → no injection
surface); sparkline canvases are fixed-size 2-D canvases plotting `history` /
`kv.series` arrays; tooltip on pick is positioned at cursor and filled per
hit-type (expert → byte range + heat + flash; plate → stage bytes, kind, lit
cells; kv → occupancy string; slabs → shapes). The stage-strip updater runs
on a 120 ms `setInterval` reading `gl.fwdStage` (kept off the rAF loop — DOM
writes at 60 Hz are wasteful and jittery).

---

## 11. Deployment

`deploy/strata-atlas.service` — **systemd user unit** (not system-wide: no
root, uses the user's `$HOME` paths via `%h`):

```
ExecStartPre=/usr/bin/python3 tools/build_atlas.py    # rebuild atlas on boot
ExecStart=/usr/bin/python3 collect/collector.py --serve-port 8091 --from-start
Restart=always  RestartSec=5   WantedBy=default.target
```

Install recipe: `systemctl --user enable --now strata-atlas.service` **plus**
`loginctl enable-linger $USER` — without linger, user services do **not**
start at boot before login (this is the R7 linchpin). Verified state:
`is-active=active`, `is-enabled=enabled`, `Linger=yes`, listening
`127.0.0.1:8091`. `ExecStartPre` rebuild means a changed pack directory
self-heals the atlas at next boot. A templated unit
(`strata-atlas-collector@.service`) exists for multi-engine setups (unused).
Pitfall hit: the unit initially referenced an environment-manager `python3`
that didn't exist under systemd's minimal PATH — switched to the absolute
`/usr/bin/python3` (3.14.4) (§13 bug 8). Journal is quiet because
`log_message` is silenced; `live/collector.out` remains as a tail artifact.

---

## 12. Testing & verification

**L1 — Python unit tests (`test/test_all.py`, 12 tests, stdlib `unittest`**
because pytest is absent): GGUF header fixture, STRP header decode, parser
fan-out contract (list return), full-log event census, compose() with
fixture metrics (kv/experts/derived values asserted numerically), atomic
write check, tailer seek behaviour. `python3 -m unittest discover -s test` →
12/12.

**L2 — JS syntax gates:** `node --check web/atlas_gl.js` for the module, plus
a Python preprocessor that extracts the inline `<script>` body from
`index.html` to a temp file and `node --check`s it. Catches every typos-class
bug before opening a browser (twice it did).

**L3 — Headless browser integration:** Playwright Chromium (from the hermes
snap's Python) launched with `--use-angle=swiftshader
--enable-unsafe-swiftshader` (software GL — the box's *only* headless GL path
that isn't broken; also proves the renderer needs no ANGLE-specific luck).
Assertions: WebGL2 context truthy, zero `console.error` / `pageerror`, plate
count in the scene, FWD strip contents (expects one of 8 stage names,
`*` on the active chip), HUD text fields contain realistic numbers.
A screenshot is captured **mid-generation**: a background
`curl POST /v1/chat/completions -d '{"max_tokens":900,...}'` drives real
decode while the page renders, guaranteeing particles/glows/KV-bar are in
non-idle states for the capture.

**L4 — Data-plane checks:** `curl :8091/api/telemetry | python -c …` asserts
`live.phase`, `kv.seq_tokens>0`, `derived.experts.per_token_measured∈[450,
650]`; byte sizes of atlas files asserted stable within tolerance.

**Geometry regression:** `getBoundingClientRect` of every panel via
`page.evaluate` to prove no overlaps after CSS changes (§13 bugs 9–11 are all
CSS-layout class bugs that screenshots + rect dumps found in seconds).

---

## 13. Bug catalogue

Each entry: symptom → diagnosis → fix. These are as much a deliverable as the
features; future maintainers should read this section first.

1. **GGUF metadata parsed to garbage.** Used pre-v2 llama.cpp type enum;
   BOOL(7)/STRING(8)/ARRAY(9) mis-typed, so the header walk desynced after
   two pairs. Diagnosed by hex-dumping offsets and hand-decoding one pair.
   Fix: correct enum table (§4.1). *Lesson: hexdump one pair; never trust a
   library blog post for an enum.*
2. **`expert-profile.bin` off-by-24.** Treated body as starting at byte 0;
   all heat values nonsense. Fix: header = 4 magic + 6×u32 = 24 B.
3. **`edit()` atomicity.** A 8-hunk batch into `atlas_gl.js` failed because
   one hunk's `oldText` drifted; **all 8 were rejected**. Fix: re-slice into
   smaller verified batches. *The multi-edit tool is transactional — one bad
   anchor rolls back the transaction.*
4. **Collector crash on fan-out lines.** `pump()` assumed `parse_line` → dict
   (`.append(e)` on a list nested a list into the ring → later `KeyError`).
   Fix: `for e in (ev if isinstance(ev, list) else [ev])`. The return
   contract (`None | dict | list[dict]`) is now documented at both ends.
5. **`prompt_hit_rate` always `None`.** Looked for `totals.reused_tokens`;
   Strata's key is `reused`. Fixed; also `requests_kept` is an int (a
   `.len()` attempt would have thrown). *Always dump the live payload once;
   don't guess key names.*
6. **48 particles per token.** Particle spawn inside the per-layer flash loop
   — visual soup + GPU fill-rate hit. Hoisted the push out.
7. **Glow sheets vanished when no particles were alive.** FX pass early-
   returned on `if(!parts.length)` *before* pushing `layerAct` glows — so
   post-generation afterglow (the most beautiful state) never rendered. Fix:
   push glows before the guard.
8. **systemd `ExecStart=python3` → exit 127.** Minimal unit PATH ≠ interactive
   PATH. Absolute `/usr/bin/python3` + `journalctl` inspection.
9. **Bottom bar stacked at top-left.** `.hud{position:absolute}` cascaded onto
   the two child panels inside `#bt`, removing them from flex flow (they
   stacked at static position). Fix: `#bt .hud{position:static}`. Flex
   layout silently no-ops on absolute children.
10. **`#bt` ignored `bottom:10px` entirely.** During a markup rewrite the
    wrapper lost `class="hud"` — `position:absolute` came *from* that class,
    so the id-rule's bottom/left/right applied to a static block and were
    ignored. `getComputedStyle` showed `position:static, bottom:"10px"` —
    a 2-minute diagnosis. Fix: restore the class.
11. **Stage strip / legend overlapped the bottom bar** after (9)/(10) fixed
    the bar's height (251 px). Rect dumps → lift both to `bottom:286px`.
12. **Playwright screenshot timeouts (30 s) during active generation.**
    Software GL (swiftshader) + heavy decode contention starves the
    compositor; screenshot waits for a stable frame. Mitigations: `domcontentloaded`
    (not `networkidle`), generous explicit timeouts, `animations:"disabled"`,
    and accept a retry — never a sign of a page bug (console stayed clean).
13. **Camera clipped the back plate row** at default elevation — raised
    `orbEl→0.72`, `orbDist→460`, pushed `target.z` forward; confirmed by
    screenshot, tuned once, frozen in code.

---

## 14. Honesty constraints

Things the scene *appears* to show and does not:

* **Per-token expert routing is synthetic.** Strata's API/log expose only
  request-level aggregate counts; the amber cell flashes are drawn from the
  *static STRP profile heat* (measured over a profiling corpus) with random
  heat-weighted picks per token. Label in HUD: `routing overlay: synthetic
  (API exposes no per-token expert ids)`. The **colours are real data**;
  the **flicker is evocative, not evidential**.
* **Block-internal stage timing is a clock**, not measured sub-layer
  telemetry (no such instrumentation exists). Stage *order and existence*
  come from the real architecture metadata.
* **KV byte figures are analytic estimates** (12 full-attn layers × GQA 2×
  (256+256)/2 × int8 = 6,144 B/token); the *token counts* are real.
* KV "resident vs streamed" colours use `engine.kv_resident` (prefix
  residency), not per-block provenance.

---

## 15. Reproduction recipe

```bash
cd ~/projects/strata-qwen38fn-unslothGGUF-RTX-telemetry
python3 tools/build_atlas.py                       # -> web/atlas_data.js, live/atlas_manifest.json
python3 -m unittest discover -s test -v            # 12 tests, stdlib only
node --check web/atlas_gl.js                       # JS gate
python3 collect/collector.py --once                # one-shot telemetry.json
python3 collect/collector.py --serve-port 8091 --from-start   # dev mode
# production:
systemctl --user enable --now strata-atlas.service && loginctl enable-linger $USER
# smoke:
curl -s 127.0.0.1:8091/api/telemetry | python3 -m json.tool | head -30
nohup curl -s 127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"hello"}],"max_tokens":64}' &
# then open http://127.0.0.1:8091/ and watch the forward pass.
```

Rebuild triggers: new GGUF shards, new pack (path change → CLI flags of
`build_atlas.py`), engine upgrade (add regexes to `log_parser.py`, then check
the `log`-catchall ratio drops back to 0), Strata metrics key rename (dump
`/metrics` once, adjust `compose()`).
