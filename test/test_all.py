"""Stdlib-only tests (python3 -m unittest discover test -v).

Covers: GGUF v3 metadata parsing, STRP expert-profile decode, native_experts
offset table, dense.txt reader, log_parser event taxonomy (including the five
new Strata banner events), and collector.compose() snapshot shape.
Run from repo root: python3 -m unittest discover -s test -v
"""
import json, os, struct, sys, tempfile, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "collect"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from log_parser import parse_line  # noqa: E402
import build_atlas as BA           # noqa: E402

LIVE = os.path.join(ROOT, "live")
MANIFEST = os.path.join(LIVE, "atlas_manifest.json")


class TestLogParser(unittest.TestCase):
    def test_strata_banner_events(self):
        cases = [
            ("strata mtp: draft layer loaded, 837 MiB of VRAM (experts 675, dense 111), files read in 0.42 s (1,853 MiB/s)", "mtp_load"),
            ("strata generate: pre-filled 5175 of 5175 slots from the profile in 1.3 s (7078 MB/s); slot 0 verified", "expert_prefill"),
            ("strata generate: 5 expert-pool workers + the host thread", "pool_workers"),
            ("strata generate: session is up (engine 0.1.57)", "session_up"),
            ("strata verify: window up to 12 tokens, 101.3 MiB of device buffers", "verify_setup"),
        ]
        for line, want in cases:
            ev = parse_line(line)
            self.assertIsNotNone(ev, f"no event for: {line}")
            self.assertEqual(ev["type"], want, f"{line!r} -> {ev}")

    def test_traffic_events(self):
        ev = parse_line("[maya] done: 4930 tokens in 487 s (10.9 tok/s) (stop, cancel=False), expert cache 95.5% hit")
        self.assertEqual(ev["type"], "done"); self.assertAlmostEqual(ev["tok_s"], 10.9)
        ev = parse_line("glm spec: 950 of 1434 drafts accepted (66.2%)")
        self.assertEqual(ev["type"], "spec"); self.assertEqual(ev["accepted"], 950)
        # Strata completion summary -> list of multiple events
        evs = parse_line("strata serve: prompt 50,511 tokens = 41,642 reused + 8,869 read in 1,010 ms (8781.0 tok/s), "
                         "4,930 generated in 142,000 ms (34.7 tok/s), drafts accepted 950 of 1434, 3 checkpoints")
        self.assertIsInstance(evs, list)
        self.assertEqual([e["type"] for e in evs],
                         ["req", "reuse", "prefill_sum", "spec", "done"])
        self.assertIsNone(parse_line(""))
        fb = parse_line("random noise")
        self.assertEqual(fb["type"], "log")  # catch-all fallback keeps unmatched lines

    def test_all_patterns_run_on_real_log(self):
        log = os.path.expanduser("~/.local/share/strata/install/strata-unsloth-ud-iq2_xs.log")
        if not os.path.exists(log):
            self.skipTest("engine log absent")
        types = {}
        with open(log, errors="replace") as f:
            for line in f:
                ev = parse_line(line.rstrip("\n"))
                if not ev:
                    continue
                for e in (ev if isinstance(ev, list) else [ev]):
                    types[e["type"]] = types.get(e["type"], 0) + 1
        self.assertGreater(len(types), 8, f"only {types}")


class TestAtlas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(MANIFEST):
            raise unittest.SkipTest("run tools/build_atlas.py first")
        with open(MANIFEST) as f:
            cls.a = json.load(f)

    def test_manifest_shape(self):
        d = self.a["dims"]
        self.assertEqual(d["layers"], 48); self.assertEqual(d["experts"], 512)
        self.assertEqual(d["top_k_experts"], 10); self.assertEqual(d["hidden"], 2560)
        self.assertEqual(len(self.a["layers"]), 48)

    def test_layer_kinds(self):
        kinds = [l["kind"] for l in self.a["layers"]]
        self.assertEqual(kinds[3], "full_attention")   # interval 4 -> L%4==3
        self.assertEqual(kinds[0], "ssm")
        self.assertEqual(sum(k == "full_attention" for k in kinds), 12)

    def test_expert_blob_arithmetic(self):
        eb = self.a["layers"][0]["expert_blob"]
        self.assertGreater(eb["gate_bytes"], 0)
        self.assertEqual(eb["gate_bytes"] % 512, 0)      # per-expert integral
        self.assertEqual(eb["down_bytes"] % 512, 0)
        self.assertIn("00002-of-00003", eb["shard"])

    def test_profile_roundtrip(self):
        prof = BA.parse_expert_profile(BA.DEF_PROFILE, 48, 512)
        self.assertEqual(len(prof["heat"]), 48)
        self.assertEqual(len(prof["heat"][0]), 512)
        self.assertGreater(prof["hot_edges"], 0)
        self.assertGreater(prof["top_experts"][0]["heat"], 0)
        self.assertEqual(prof["header"]["magic"], "STRP")
        self.assertEqual(prof["header"]["entries"], 48 * 512)

    def test_gguf_header_rejects_bad_magic(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf") as t:
            t.write(b"JUNK" + b"\0" * 100); t.flush()
            with self.assertRaises(SystemExit):
                BA.parse_gguf_metadata(t.name)

    def test_gguf_type_table(self):
        # synthetic minimal GGUF v3: magic, ver=3, n_tensors=0, n_kv=2
        def s(b): return struct.pack("<Q", len(b)) + b.encode()
        buf = b"GGUF" + struct.pack("<IQ", 3, 0) + struct.pack("<Q", 2)
        buf += s("general.architecture") + struct.pack("<I", 8) + s("qwen4exp")
        buf += s("qwen4exp.block_count") + struct.pack("<I", 4) + struct.pack("<i", 48)
        with tempfile.NamedTemporaryFile(suffix=".gguf") as t:
            t.write(buf); t.flush()
            m = BA.parse_gguf_metadata(t.name)["meta"]
            self.assertEqual(m["general.architecture"], "qwen4exp")
            self.assertEqual(m["qwen4exp.block_count"], 48)

    def test_dense_txt_offsets(self):
        t = [x for x in self.a["mtp"]["tensors"] if x["name"] == "fc_embedding.weight"][0]
        self.assertEqual((t["rows"], t["cols"]), (2560, 2560))
        self.assertEqual(t["bytes"], 6963200)  # q8_0: 2560x2560 -> block quant


class TestCollector(unittest.TestCase):
    def test_compose_shape(self):
        sys.path.insert(0, os.path.join(ROOT, "collect"))
        import collector
        metrics = {"engine": {"expert_slots": 5175}, "totals":
                   {"prompt_tokens": 100, "reused": 60,
                    "drafts_offered": 10, "drafts_accepted": 7}}
        class FakeT:
            path, counters = "/dev/null", {"done": 2}
            events = [{"type": "done", "t": "00:00:00"}]
        out = collector.compose(metrics, FakeT(), time.time())
        self.assertEqual(out["engine"]["expert_slots"], 5175)
        self.assertAlmostEqual(out["derived"]["mtp_accept_rate"], 0.7)
        self.assertAlmostEqual(out["derived"]["prompt_hit_rate"], 0.6)

    def test_live_snapshot_valid_json(self):
        p = os.path.join(LIVE, "telemetry.json")
        if not os.path.exists(p):
            self.skipTest("collector not run yet")
        with open(p) as f:
            d = json.load(f)
        self.assertIn("ts", d)


if __name__ == "__main__":
    unittest.main()
