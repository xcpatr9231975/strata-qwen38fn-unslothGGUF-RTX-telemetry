# strata-qwen38fn-unslothGGUF-RTX-telemetry

**A WebGL2 "physical weight atlas" for Qwen3.8-Flash-Next, alive with live engine
telemetry.** 48 MoE blocks × 512 routed experts rendered where they actually
*live on disk* — IQ2_XS byte ranges, shard numbers, quant kinds — overlaid with
real-time forward-pass instrumentation, expert-call accounting and KV-cache
growth, streamed from a Strata v0.1.57 inference engine running the Unsloth
GGUF build on an RTX A5000 Laptop GPU (16 GiB).

Every geometric fact is parsed from the model files; every moving colour is
pulled from the engine's live metrics + log. Nothing in the scene is faked —
the only synthetic overlay (per-token routing flicker) is labelled as such,
both here and in the HUD.

### Live capture — mid-decode, 10 s loop

![Live atlas: engine thinking at ~16 tok/s, token particles crossing glowing
blocks, KV bar filling, stage strip lit on ATTN/SSM core](docs/assets/atlas_demo.webp)

*Smaller [GIF version](docs/assets/atlas_demo.gif) · [static
frame](docs/assets/atlas_still.png) if animation is disabled.*

---

## What am I looking at

| Scene element | What it is | Where the number comes from |
|---|---|---|
| **48 glowing plates** (8×6 grid) | the transformer blocks; teal = full-attention (every 4th), indigo = SSM (hybrid) | `block_count`, `full_attention_interval` in the GGUF header |
| **512 tiny tiles per plate** | the routed experts of that layer, heat-tinted by Strata's own *measured* routing profile (`STRP` blob) and byte-mapped to their IQ2_XS ranges in the GGUF shards | `expert-profile.bin` + pack `index.txt` + GGUF tensor index |
| **amber tiles flashing** | a token just routed through this expert (overlay — the *colours* are real data, the flicker is evocative; the API exposes no per-token expert ids) | engine decode-rate scaled |
| **EMB slab** | the 248,320 × 2,560 token embedding table | GGUF metadata |
| **travelling spark** | one generated token walking `embed → 48 blocks → head`; plates glow amber on arrival | `live.tok_s` from `/metrics` |
| **bottom stage strip** | the block-internal pipeline clock: `EMB · HC0 mix · ATTN/SSM core · HC1 combine · ROUTER softmax · 10/512 experts · SHARED expert · HEAD` | architecture metadata (stage *order*), particle position (which stage now) |
| **green/blue bar (left)** | the KV cache: 32 cells = 65,536-token window; green = VRAM-resident, blue-grey = streamed from RAM, brightest = newest written | `engine.kv_resident`, live seq occupancy |
| **green sparks behind the head** | MTP speculative-draft activity | `totals.drafts_offered/accepted` |
| **HUD panels** | engine phase, decode tok/s, MTP accept %, expert-cache hits, experts called (measured vs structural), GPU/VRAM/PCIe, KV occupancy + growth sparkline, engine event log | `:8080/metrics` JSON + purpose-built log parser |

## Live features

- **Forward-pass instrumentation** — token particles, per-block arrival glow
  and an 8-stage pipeline strip that follows the head of the pass in real time.
- **Expert accounting** — structural (48 layers × top-10 = **480/token**) vs.
  *measured* routing lookups from the engine log (e.g. 545.7/token for a
  batched request), plus cache-hit % and the honest note that all **5,175
  expert slots are VRAM-resident — no eviction ever happens**.
- **KV-cache growth** — occupancy bar on the ground plane, 90 s growth-curve
  sparkline, int8 byte estimate (6,144 B/token over the 12 full-attention
  layers), parked-conversation KV.

## Quick start

```bash
# one-time (or automatic on every boot): build the static atlas from the model files
python3 tools/build_atlas.py

# run collector + HUD server on 127.0.0.1:8091
python3 collect/collector.py --serve-port 8091 --from-start
# open http://127.0.0.1:8091/

# tests (stdlib unittest only — no pytest on the lab box, no numpy needed)
python3 -m unittest discover -s test -v
```

### Permanent deployment (systemd user unit, boot-persistent)

```bash
systemctl --user enable --now deploy/strata-atlas.service
loginctl enable-linger $USER          # without linger user units do NOT start at boot
```

The unit rebuilds the atlas on boot (`ExecStartPre`), tails the engine log
from the start (so boot-banner events like `mtp_load`, `expert_prefill
5175/5175`, `session_up` land in the scene), and serves the dashboard +
`/api/telemetry` from one stdlib-only process. `Restart=always`.

## How it works (1-minute version)

```
GGUF headers · pack index.txt · expert-profile.bin (STRP) · MTP dense.txt
        │  tools/build_atlas.py (offline, stdlib struct only)
        ▼
 web/atlas_data.js  ──────────────┐  static geometry: topology, byte maps, heat
                                 ├──►  WebGL2 renderer (web/atlas_gl.js)
 Strata :8080/metrics (JSON) ──┐  │   1 shader program, ≤4 instanced draws/frame,
 engine log (tail -f) ──────────┤  │   analytic ray-plane picking
        │  collect/log_parser.py │  │            ▲ 60 Hz
        │  (≈25 event types)     ▼  │            │ 500 ms poll of /api/telemetry
        └─► collect/collector.py ─► live/telemetry.json (atomic os.replace) ◄────┘
                 └── embedded http.server on :8091 (static + API, no nginx)
```

Full engineering dossier — requirements, every binary format decoded (GGUF v3
type-enum trap, `STRP` layout forensics, MTP layout), the renderer internals
(instance layout, SDF rimming, the `uY` layer-lifting trick, additive FX
pass), deployment notes, testing methodology and a 13-entry bug catalogue with
diagnoses: **[TECHNIQUES.md](TECHNIQUES.md)**.

## Repo layout

```
collect/   collector.py (poll+tail+compose+serve) · log_parser.py (~25 event
           types, fan-out contract) · llm-scope.py (salvaged predecessor)
tools/     build_atlas.py — offline atlas builder (GGUF/STRP/pack/MTP parsers)
web/       index.html (HUD) · atlas_gl.js (WebGL2) · atlas_data.js (generated)
live/      telemetry.json · atlas_manifest.json (auditable, unit-annotated)
test/      stdlib unittest suite (12 tests, runs without network)
deploy/    systemd user units + notes
docs/      assets (demo webp/gif/still) + TECHNIQUES.md lives one level up
```

## Served endpoints

| Endpoint | Content |
|---|---|
| `/` | HUD + WebGL scene |
| `/api/telemetry` | live telemetry snapshot (0.5 s cadence, atomic file behind it) |
| `/api/atlas` | full atlas manifest (every number annotated with unit + source) |

## Honest caveats

- Per-token expert *flashes* are drawn from a measured routing profile —
  Strata exposes no per-token expert ids, so the flicker pattern is
  representative, not evidential. (HUD says so too.)
- Block-internal stage timing is a geometric clock derived from the real
  architecture stage order, not sub-layer telemetry.
- KV byte figures are analytic (12 full-attention layers × GQA × int8); token
  counts are real.

---

Lab tooling for the `qwen4exp` (Qwen3.8-Flash-Next) Unsloth UD-IQ2_XS GGUF on
Strata 0.1.57. Supersedes the dead vllama-era
[`qwen38-visual`](https://github.com/xcpatr9231975/qwen38-visual) telemetry.
