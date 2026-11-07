# strata-qwen38fn-unslothGGUF-RTX-telemetry

Real-time **WebGL2 visualisation of the physical Qwen3.8-Flash-Next weight
atlas** (512-expert-per-layer MoE, 48 blocks, IQ2_XS GGUF served by
[Strata] 0.1.57 on an RTX A5000 Laptop GPU) **overlaid with live engine +
GPU telemetry**. Replaces the dead vllama-era `qwen38-visual` telemetry: the
collector speaks Strata's native JSON `/metrics` dialect and tails the
engine log with a purpose-built parser.

> 📚 **Full engineering dossier:** [`TECHNIQUES.md`](TECHNIQUES.md) — requirements,
> every binary format decoded (GGUF v3, STRP, pack index, MTP), the WebGL2
> renderer architecture, deployment, testing methodology, and the complete bug
> catalogue with diagnoses.

## What the scene shows (every number comes from disk, nothing is faked)

| Scene element | Source of truth | Meaning |
|---|---|---|
| 48 layer plates, 8×6 grid | GGUF `qwen4exp.block_count=48` | one plate per transformer block |
| Rim colour cyan/magenta | `attention.full_attention_interval=4` | cyan = full-attention (L%4==3, 12 blocks); magenta = SSM/linear-attention (36 blocks, `state_size=128`, `inner_size=6144`, `time_step_rank=48`) |
| 32×16 = **512 cells per plate** | `expert_count=512`, `expert_used_count=10` | one cell per routed expert; top-10 of 512 active per token/layer |
| Cell tint (navy→teal→gold) | `expert-profile.bin` (magic `STRP`, 196 632 B) | heat = # co-routing edge endpoints from the 24 576 u16-pair edge records (12 288 hot edges, 12 288 cold tail rows); heat>0 ⇒ resident in the warmed VRAM cache ("pre-filled 5175 of 5175 slots from the profile … no eviction", 8 494 MiB) |
| Amber flash on cells | synthesised top-K=10 | Strata's API exposes only the aggregate hit %, **not** per-token expert ids — routing flashes are synthetic and labelled as such in the HUD |
| Router bar per plate | `ffn_gate_inp.weight` (2560×512 bf16, 2.62 MiB) | MoE gating projection |
| Gold notch | GGUF `qwen4exp.ple.layers=[1]` | blk.1 hosts the PLE n-gram path (`ngram_size=3`, 8 heads/ngram, `ple_key.weight` 2560×10240 f32) |
| Green slab (left) | `token_embd.weight`, vocab 248 320 × hidden 2 560 | embedding position space |
| Purple slab (right) | `output.weight` (transposed) | LM head + sampling defaults (`top_k=30 top_p=0.9 temp=1.0`) |
| Far green slab | `~/strata-models/mtp/rt/` (experts.bin 707.8 MiB / 512×1 382 400 B, dense.bin 116.1 MiB, 29 dense tensors) | MTP draft layer; green particles loop head→MTP→verify |
| Token particles | live `tok_s` (≈34.8 decode) | polyline embed→blk0…47→head |
| Hover tooltip | `native_experts.txt` (512 rows/layer: gate/up/down byte offsets per expert, shard column) | exact file offsets + bytes-per-weight of the expert under the cursor |

## Telemetry pipeline

```
strata serve :8080 ── GET /metrics ──┐
                                     ├─ collect/collector.py ─ live/telemetry.json
engine log ── collect/log_parser.py ─┘        (atomic write, every 2 s)
   ~/.local/share/strata/install/strata-unsloth-ud-iq2_xs.log
                                     │ --serve-port 8091
                                     └─ http://127.0.0.1:8091/  (web/index.html
                                        + atlas_data.js + /api/telemetry)
```

* **`/metrics` JSON**: `engine` (expert_slots=5175, expert_cache_mib, KV),
  `live` (phase/tok_s), `requests`, `conversation_cache`, `hardware`
  (GPU util/temp/power/VRAM, CPU/RAM), `history` arrays, `totals`
  (`drafts_accepted/drafts_offered` → MTP accept rate, `reused/prompt_tokens`
  → prompt-hit rate, both pre-derived in `derived`).
* **`log_parser.py`** turns ~19 log dialects into typed events, incl. the
  five Strata startup banners added here: `mtp_load`, `expert_prefill`,
  `pool_workers`, `session_up`, `verify_setup` (plus req/reuse/spec/kv/
  expert_cache/slot_save/cache_restore/borrow/done/engine_start/vram_warn).
* **`collect/llm-scope.py`**: the original terminal dashboard, ported
  (`import log_parser as P`), still works standalone:
  `python3 collect/llm-scope.py --log ~/.local/share/strata/install/strata-unsloth-ud-iq2_xs.log`.

## Atlas builder — `tools/build_atlas.py`

Parses, from stdlib only (`struct`+`json`, no numpy):

1. **GGUF v3 metadata** (3 shards, reads header+KV from shard 00001):
   dims, SSM, hyper-connections (count=4 rank=320), indexer attention
   (4 heads, k=128, top_k=2048), PLE, RoPE sections `[11,11,10,0]`,
   `compress_ratios` pattern, quantisation (`general.file_type=20` IQ2_XS,
   imatrix 926 entries).
2. **`packs/unsloth-ud-iq2_xs/index.txt`** → 1 079 dense tensors with
   stage classification (embedding_or_head / attention / indexer / ssm /
   moe_router / moe_shared / hyper_connection / ple).
3. **`native_experts.txt`** → per-layer byte regions of the expert blob
   (total 42 283 827 200 B; per-expert gate+up vs down sizes → effective bpw
   ≈ 2.55 / 1.28 for routed experts).
4. **`expert-profile.bin`** (STRP header `[1,48,512,24576,24576,…]`, u16
   stream at byte 24) → heat matrix + top-expert list.
5. **`mtp/rt/dense.txt`** (6-column records) + file sizes → MTP structure.

Outputs: `web/atlas_data.js` (`window.ATLAS_META` + `window.ATLAS_HEAT`,
consumed directly by the renderer — no bundler) and
`live/atlas_manifest.json` (full machine-readable manifest, ~250 KB).

## Run

```bash
cd ~/projects/strata-qwen38fn-unslothGGUF-RTX-telemetry
python3 tools/build_atlas.py            # once after model/pack changes
python3 collect/collector.py --once     # single snapshot smoke test
python3 collect/collector.py --serve-port 8091   # collector + web server
# open http://127.0.0.1:8091/
```

Camera: drag orbit · wheel zoom · right-drag pan · click fly-to · `r` reset.

## Tests

```bash
python3 -m unittest discover -s test -v   # 12 tests, stdlib only
```

Cover GGUF v3 type-table decode (synthetic header), STRP decode round-trip,
expert-blob arithmetic, dense.txt byte offsets, the five new parser events,
multi-event completion-summary lines, and `compose()` snapshot derivation
against the live engine when it is up.

## Known honesty notes

* Per-token expert activation is **synthetic** (heat-weighted top-K) — the
  API publishes only aggregate expert-cache hit %; labelled in HUD/legend.
* Heat model treats STRP rows as undirected (L,expert) endpoint pairs; the
  12 288-record half is edge records, the rest decode to index maps.
* Scene assumes the IQ2_XS UD pack + MTP RT drafter paths below; if the pack
  moves, `build_atlas.py` flags missing files loudly rather than guessing.

## Absolute paths baked in as defaults (override with CLI flags)

| Path | Contents |
|---|---|
| `~/strata-models/models/unsloth-UD-IQ2_XS/Qwen3.8-Flash-Next-UD-IQ2_XS-0000{1,2,3}-of-00003.gguf` | model shards (43.8+26.8+0.01 GB) |
| `~/strata-models/packs/unsloth-ud-iq2_xs/` | `index.txt`, `native_experts.txt`, `conversions.json`, `expert-profile.bin` |
| `~/.local/share/strata/components/control/0.1.57/data/expert-profile.bin` | canonical STRP profile |
| `~/strata-models/mtp/rt/` | MTP drafter (`experts.bin`, `dense.bin`, `dense.txt`, `draft_vocab.bin`) |
| `~/.local/share/strata/install/strata-unsloth-ud-iq2_xs.log` | engine log tail |
| `http://127.0.0.1:8080/metrics` | engine JSON snapshot |


## Live scene features

- **Forward-pass visualisation** — one token particle per decoded token
  travels the polyline `embedding → blk0…blk47 → head`; each block plate
  lights up amber the instant the particle head arrives. The chip strip
  (`EMB → HC0 mix → ATTN/SSM core → HC1 combine → ROUTER softmax →
  10/512 experts → SHARED expert → HEAD`) tracks the particle's sub-block
  stage live; during prefill it shows the read fraction.
- **Expert accounting** — hovering a block plate reports the number of lit
  expert cells *right now*; the HUD reports routed-expert lookups for the
  last request (`experts.log` → `expert_cache` event, measured per-token)
  next to the structural bound of 48 layers × top-10 = 480/token, plus the
  cache-hit %. All 5,175 expert slots are VRAM-resident (`resident experts`
  row): expert eviction cannot occur — only streamed weight loading can.
  The per-token routing flash overlay itself remains **synthetic** (API
  exposes no per-token expert ids).
- **KV cache growth bar** — 32 ground cells (front-left, 1 cell ≈ 2,048
  tokens of the 65,536 engine window): solid green = VRAM-resident prefix,
  blue = KV streamed from RAM, the newest written cell flashes. HUD shows
  occupancy in tokens + ≈MiB (int8, full-attn layers only), parked
  conversation KV (GiB, slots, evictions). The `KV seq tokens` sparkline
  keeps a 3-minute history.
- **HUD layout** — engine panel top-left, hardware top-right, event log
  bottom-left, sparklines bottom-right, stage strip centred above the
  bottom bar (no window-edge overlap).
