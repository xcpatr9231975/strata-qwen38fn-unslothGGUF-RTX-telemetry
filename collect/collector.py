#!/usr/bin/env python3
"""
collector.py — Strata telemetry bridge for the WebGL weight atlas.

The Strata 0.1.57 engine exposes a single rich JSON snapshot at
    GET http://127.0.0.1:8080/metrics
(engine phase, tok/s, totals, conversation cache, GPU util/temp/power/VRAM,
PCIe budget, KV streaming %, per-token history arrays, lifetime totals) and
writes a human-readable log to
    ~/.local/share/strata/install/strata-<model>.log
which contains structured lines we parse into typed events (see log_parser.py:
speculative accept ratios, expert-cache warm, KV streaming, prefix reuse,
slot save/restore, borrow, engine start/exit, VRAM warnings, MTP load,
expert prefill, pool workers, session up, verify setup).

This process merges both sources into live/telemetry.json every `--interval`
seconds, and can *serve* the web/ directory plus the merged snapshot on
`--serve-port` so the visualiser needs no separate web server:

    GET /                    -> web/index.html
    GET /api/telemetry       -> live/telemetry.json content
    GET /api/atlas           -> live/atlas_manifest.json content
    GET <anything else>      -> static file under web/

Safety posture: engine metrics/log are read-only; nothing is ever POSTed to
the engine. `--once` writes a single snapshot and exits (cron-friendly).
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time, urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_parser import parse_line  # noqa: E402

HOME = os.path.expanduser("~")
DEF_METRICS = "http://127.0.0.1:8080/metrics"
DEF_LOG = f"{HOME}/.local/share/strata/install/strata-unsloth-ud-iq2_xs.log"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_OUT = os.path.join(ROOT, "live", "telemetry.json")
DEF_WEB = os.path.join(ROOT, "web")


def fetch_json(url: str, timeout: float = 3.0):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


class LogTailer:
    """Follows the engine log like `tail -F`, yielding parsed events.

    Survives rotation: on ENOENT/shrink, reopens from 0. Events are kept in
    a ring so the JSON snapshot carries the most recent `ring` entries.
    """
    def __init__(self, path: str, ring: int = 60, from_start: bool = False):
        self.path, self.ring = path, ring
        self.events: deque = deque(maxlen=ring)
        self.counters: dict = {}
        self.from_start = from_start
        self.f = None
        self.pos = 0
        self.open()

    def open(self):
        if self.f:
            try:
                self.f.close()
            except Exception:
                pass
        try:
            self.f = open(self.path, "r", errors="replace")
            if not self.from_start:
                self.f.seek(0, os.SEEK_END)
        except FileNotFoundError:
            self.f = None

    def pump(self):
        """Read whatever is available; return list of new event dicts."""
        new = []
        if self.f is None:
            self.open()
            return new
        while True:
            line = self.f.readline()
            if not line:
                try:
                    if os.path.getsize(self.path) < self.f.tell():
                        self.open()  # rotated
                except OSError:
                    self.open()
                break
            evs = parse_line(line.rstrip("\n"))
            if evs:
                if isinstance(evs, dict):
                    evs = [evs]
                for ev in evs:
                    ev.setdefault("t", time.strftime("%H:%M:%S"))
                    self.events.append(ev)
                    self.counters[ev["type"]] = self.counters.get(ev["type"], 0) + 1
                    new.append(ev)
        return new


def compose(metrics, tailer, started, hist=None):
    """Merge engine metrics + parsed log state into the telemetry snapshot.

    `hist` is a persistent dict of deques kept across polls by the caller;
    currently one series: kv.seq_tokens sampled every poll (KV growth curve).
    """
    if hist is None:
        hist = {"kv": deque(maxlen=180)}
    out = {
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "collector_uptime_s": round(time.time() - started, 1),
        "events": list(tailer.events)[-40:],
        "log_counters": dict(tailer.counters),
        "sources": {
            "metrics": {"url": None, "ok": metrics is not None},
            "log": {"path": tailer.path},
        },
    }
    if metrics:
        for k in ("engine", "live", "requests", "conversation_cache",
                  "hardware", "hardware_static", "history", "totals"):
            if k in metrics:
                out[k] = metrics[k]
        # derive a couple of convenience fields the visualiser wants
        tot = metrics.get("totals") or {}
        off = tot.get("drafts_offered") or 0
        acc = tot.get("drafts_accepted") or 0
        eng = metrics.get("engine") or {}
        live = metrics.get("live") or {}
        reqs = metrics.get("requests") or []
        cc = metrics.get("conversation_cache") or {}
        out["derived"] = {
            "mtp_accept_rate": (acc / off) if off else None,
            "prompt_hit_rate": ((tot.get("reused") or 0) / tot["prompt_tokens"])
                               if tot.get("prompt_tokens") else None,
        }

        # ---- (c) KV cache occupancy -------------------------------------
        # engine.kv is the KV dtype (e.g. "int8"), kv_resident = tokens whose
        # KV lives in VRAM, max_context = per-sequence capacity. Occupancy =
        # the live sequence (prompt+generated) else the last finished req.
        seq = None
        if live.get("prompt_total") is not None:
            seq = int(live["prompt_total"]) + int(live.get("generated") or 0)
        elif reqs:
            r0 = reqs[0]
            seq = int(r0.get("prompt_tokens") or 0) + int(r0.get("output_tokens") or 0)
        if seq is not None:
            hist["kv"].append(seq)
        out["kv"] = {
            "dtype": eng.get("kv"),                        # "int8"
            "capacity_tokens": eng.get("max_context"),     # 65536
            "resident_tokens": eng.get("kv_resident"),     # VRAM-resident prefix
            "seq_tokens": seq,                             # occupancy now (last/active)
            "cache_bytes": cc.get("bytes"),                # parked conversation KV
            "cache_slots": cc.get("slots"),
            "parked": cc.get("parked"),
            "evictions": cc.get("evictions"),
            "series": list(hist["kv"]),                    # growth curve, one pt/poll
        }

        # ---- (b) experts actually called ----------------------------------
        # Each `expert_cache` log line reports ONE request's totals: lookups
        # (= tokens x 48 layers x top-10) and hits in the resident cache.
        ec = dn = None
        for ev in reversed(list(tailer.events)):
            ty = ev.get("type")
            if ec is None and ty == "expert_cache":
                ec = ev
            elif ec is not None and dn is None and ty == "done":
                dn = ev
                break
        exp = {"per_token_structural": 48 * 10}  # every token touches 48 MoE layers x top-10
        if ec:
            exp["last_request_lookups"] = ec.get("lookups")
            exp["last_request_hits"] = ec.get("hits")
            exp["last_hit_pct"] = ec.get("hit_pct")
            if dn and dn.get("tokens"):
                exp["last_request_tokens"] = dn["tokens"]
                if ec.get("lookups"):
                    exp["per_token_measured"] = ec["lookups"] / dn["tokens"]
        out["derived"]["experts"] = exp
    return out


class Serve:
    """Serves web/ statically; /api/telemetry and /api/atlas read live files."""
    def __init__(self, web_dir, live_dir, port):
        self.web, self.live, self.port = web_dir, live_dir, port

    def handler(self):
        web, live = self.web, self.live
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass
            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                p = self.path.split("?")[0]
                if p in ("/", ""):
                    p = "/index.html"
                if p == "/api/telemetry":
                    try:
                        return self._send(open(os.path.join(live, "telemetry.json"), "rb").read(), "application/json")
                    except FileNotFoundError:
                        return self._send(b'{"state":"waiting"}', "application/json")
                if p == "/api/atlas":
                    try:
                        return self._send(open(os.path.join(live, "atlas_manifest.json"), "rb").read(), "application/json")
                    except FileNotFoundError:
                        return self._send(b'{"error":"run tools/build_atlas.py"}', "application/json")
                fp = os.path.normpath(os.path.join(web, p.lstrip("/")))
                if not fp.startswith(web) or not os.path.isfile(fp):
                    self.send_error(404)
                    return
                ctype = ("application/javascript" if fp.endswith(".js")
                        else "application/json" if fp.endswith(".json")
                        else "text/html" if fp.endswith(".html") else "text/plain")
                self._send(open(fp, "rb").read(), ctype)
        return H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default=DEF_METRICS)
    ap.add_argument("--log", default=DEF_LOG)
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true", help="single snapshot, exit")
    ap.add_argument("--serve-port", type=int, default=0, help="serve web/ + API on this port")
    ap.add_argument("--from-start", action="store_true", help="read log from beginning")
    args = ap.parse_args(argv)

    tailer = LogTailer(args.log, from_start=args.from_start or args.once)
    hist = {"kv": deque(maxlen=180)}
    started = time.time()
    errors = 0

    if args.serve_port:
        srv = Serve(DEF_WEB, os.path.dirname(args.out), args.serve_port)
        httpd = ThreadingHTTPServer(("127.0.0.1", args.serve_port), srv.handler())
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        print(f"[collector] serving http://127.0.0.1:{args.serve_port}/ (web: {DEF_WEB})", flush=True)

    if args.once:
        m = fetch_json(args.metrics)
        tailer.pump()
        snap = compose(m, tailer, started, hist)
        snap["sources"]["metrics"]["url"] = args.metrics
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(snap, f, indent=1)
        print(f"[collector] wrote {args.out} ({os.path.getsize(args.out)} B)", flush=True)
        return 0

    print(f"[collector] metrics={args.metrics} log={args.log} out={args.out}", flush=True)
    while True:
        try:
            m = fetch_json(args.metrics)
            errors = 0
        except Exception as e:
            m = None
            errors += 1
            if errors in (1, 10, 100):
                print(f"[collector] metrics poll failed ({errors}x): {e}", flush=True)
        tailer.pump()
        snap = compose(m, tailer, started, hist)
        snap["sources"]["metrics"]["url"] = args.metrics
        tmp = args.out + ".tmp"
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(snap, f)
        os.replace(tmp, args.out)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
