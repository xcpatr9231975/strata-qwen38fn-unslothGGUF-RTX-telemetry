"""Read-only Strata/Maya log collector and local telemetry dashboard."""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

import log_parser as P

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/events.jsonl'
MAX_EVENTS = 200_000
MAX_FILE_BYTES = 32 * 1024 * 1024
EVENTS = deque(maxlen=MAX_EVENTS)
_lock = threading.Lock()
_seq = 0
DEFAULT_LOG = None
CONTEXT_CAP = 65536
SESSION = uuid.uuid4().hex
HTML = (ROOT / 'index.html').read_bytes()


def load_history():
    """Stream bounded history; tolerate incomplete/corrupt records."""
    global _seq
    with _lock:
        for path in (DATA.with_suffix('.jsonl.1'), DATA):
            if not path.exists():
                continue
            with path.open(errors='replace') as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                        if not isinstance(ev, dict) or not isinstance(ev.get('type'), str):
                            continue
                        if not isinstance(ev.get('ts'), (int, float)) or not math.isfinite(ev['ts']):
                            continue
                        saved_id = ev.get('id')
                        _seq = max(_seq + 1, saved_id) if isinstance(saved_id, int) else _seq + 1
                        ev['id'] = _seq
                        EVENTS.append(ev)
                    except (ValueError, TypeError):
                        continue


def record(ev):
    global _seq
    with _lock:
        _seq += 1
        ev = dict(ev, id=_seq, ts=time.time())
        EVENTS.append(ev)
        try:
            DATA.parent.mkdir(parents=True, exist_ok=True)
            if DATA.exists() and DATA.stat().st_size >= MAX_FILE_BYTES:
                DATA.replace(DATA.with_suffix('.jsonl.1'))
            payload = (json.dumps(ev, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
            with DATA.open('ab+') as f:
                # A crash may leave an incomplete final record. Keep it isolated
                # so it cannot swallow the next valid record during reload.
                if f.tell():
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b'\n':
                        f.write(b'\n')
                f.write(payload)
        except (OSError, ValueError) as exc:
            print(f'[llm-scope] persistence failed: {exc}', file=sys.stderr)


def tail(path, on_line, poll=0.5, stop=None):
    """Start at EOF; read replacements/truncations from start; buffer partial lines."""
    stop = stop or threading.Event()
    f = None
    first = True
    pending = ''
    try:
        while not stop.is_set():
            try:
                stat = path.stat()
                if f is None or (stat.st_dev, stat.st_ino) != (os.fstat(f.fileno()).st_dev, os.fstat(f.fileno()).st_ino) or stat.st_size < f.tell():
                    if f:
                        f.close()
                    f = path.open('r', errors='replace')
                    if first:
                        f.seek(0, os.SEEK_END)
                        first = False
                    pending = ''
                chunk = f.readline()
                if chunk:
                    pending += chunk
                    if pending.endswith('\n'):
                        try:
                            on_line(pending)
                        except Exception as exc:
                            print(f'[llm-scope] parse error: {exc}', file=sys.stderr)
                        pending = ''
                    continue
            except OSError:
                if f:
                    f.close()
                f = None
                pending = ''
            stop.wait(poll)
    finally:
        if f:
            f.close()


def engine_tail(stop):
    def consume(line):
        events = P.parse_line(line)
        for ev in events if isinstance(events, list) else [events]:
            if ev:
                record(ev)
    tail(DEFAULT_LOG, consume, stop=stop)


def gpu_sampler(stop, gpu=0, poll=2.0):
    if not shutil.which('nvidia-smi'):
        print('[llm-scope] nvidia-smi unavailable; GPU lane disabled')
        return
    fields = 'utilization.gpu,temperature.gpu,power.draw,memory.used,memory.free'
    while not stop.is_set():
        try:
            result = subprocess.run(['nvidia-smi', f'--id={gpu}', f'--query-gpu={fields}', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
            values = []
            for v in result.stdout.strip().split(','):
                try:
                    n = float(v.strip())
                    values.append(n if math.isfinite(n) else None)
                except ValueError:
                    values.append(None)
            if len(values) != 5:
                raise ValueError('unexpected GPU response')
            u, temp, power, used, free = values
            record(dict(type='gpu', gpu=gpu, u=u, t=temp, p=power, m=used, f=free / 1024 if free is not None else None))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'[llm-scope] GPU sample failed: {exc}', file=sys.stderr)
        stop.wait(poll)


def monitor_tail(path, stop):
    import re
    pattern = re.compile(r'U:(\d+)% .*?T:([\d.]+).*?P:([\d.]+)W .*?M:(\d+)MiB .*?F:([\d.]+)GiB')
    def consume(line):
        m = pattern.search(line)
        if m:
            u, temp, power, used, free = map(float, m.groups())
            record(dict(type='gpu', u=u, t=temp, p=power, m=used, f=free))
    tail(path, consume, stop=stop)


def disk_counters(text, device):
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 14 and fields[2] == device:
            return int(fields[5]), int(fields[9]), int(fields[11])
    return None


def sampler(stop, device='nvme0n1', poll=2.0):
    last = None
    while not stop.is_set():
        try:
            psi = {}
            for line in Path('/proc/pressure/io').read_text().splitlines():
                fields = line.split()
                psi[fields[0]] = float(dict(v.split('=') for v in fields[1:])['avg10'])
            counters = disk_counters(Path('/proc/diskstats').read_text(), device)
            now = time.monotonic()
            read = write = None
            if counters and last:
                dt = now - last[0]
                read = max(0, counters[0] - last[1][0]) * 512 / dt / 1e6
                write = max(0, counters[1] - last[1][1]) * 512 / dt / 1e6
            last = (now, counters) if counters else None
            mem = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                k, v = line.split(':', 1)
                if k in ('Dirty', 'Writeback'):
                    mem[k] = int(v.split()[0]) / 1024
            record(dict(type='sys', io_some=psi.get('some'), io_full=psi.get('full'), disk=device, disk_r_mbs=read, disk_w_mbs=write, in_flight=counters[2] if counters else None, dirty_mib=mem.get('Dirty'), writeback_mib=mem.get('Writeback')))
        except (OSError, ValueError, KeyError) as exc:
            print(f'[llm-scope] system sample failed: {exc}', file=sys.stderr)
        stop.wait(poll)


def event_page(after, limit=2000):
    with _lock:
        oldest = EVENTS[0]['id'] if EVENTS else _seq + 1
        reset = after > _seq
        if reset:
            after = 0
        out = []
        for e in EVENTS:
            if e['id'] > after:
                out.append(e)
                if len(out) == limit:
                    break
        cursor = out[-1]['id'] if out else after
        return dict(now=time.time(), session=SESSION, events=out, cursor=cursor, more=cursor < _seq, reset=reset, gap=after > 0 and after < oldest - 1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ('/', '/index.html'):
            self.send_body(200, HTML, 'text/html; charset=utf-8')
        elif url.path == '/api/events':
            try:
                after = int(parse_qs(url.query).get('after', ['0'])[0])
                if after < 0:
                    raise ValueError()
            except ValueError:
                self.send_body(400, b'{"error":"after must be a nonnegative integer"}')
                return
            self.send_body(200, json.dumps(event_page(after)).encode())
        elif url.path == '/api/state':
            with _lock:
                counts = {t: 0 for t in ('req', 'done', 'refused')}
                prompt = io = None
                for e in EVENTS:
                    if e['type'] in counts:
                        counts[e['type']] += 1
                    if e['type'] == 'req':
                        prompt = e.get('prompt_tokens')
                    if e['type'] == 'sys':
                        io = e.get('io_full')
            self.send_body(200, json.dumps(dict(requests=counts['req'], completed=counts['done'], refusals=counts['refused'], last_prompt=prompt, io_full=io, engine_log=DEFAULT_LOG.name if DEFAULT_LOG else None, context_cap=CONTEXT_CAP, scope='retained history')).encode())
        else:
            self.send_body(404, b'{"error":"not found"}')


def main():
    global DEFAULT_LOG, CONTEXT_CAP
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=8089)
    ap.add_argument('--log', default=os.environ.get('LLM_SCOPE_LOG'))
    ap.add_argument('--monitor', help='use this GPU monitor log instead of nvidia-smi')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--disk', default='nvme0n1')
    ap.add_argument('--context-cap', type=int, default=65536, help='dashboard context limit; match engine configuration')
    ap.add_argument('--poll', type=float, default=2.0, help='system/GPU sampling seconds')
    args = ap.parse_args()
    if args.context_cap <= 0:
        ap.error('--context-cap must be positive')
    CONTEXT_CAP = args.context_cap
    if args.poll < 0.2 or not math.isfinite(args.poll):
        ap.error('--poll must be finite and at least 0.2 seconds')
    if args.log:
        DEFAULT_LOG = Path(args.log).expanduser()
    else:
        preferred = Path.home() / '.local/share/strata/install/strata-unsloth-ud-iq2_xs.log'
        candidates = list(preferred.parent.glob('strata-*.log'))
        DEFAULT_LOG = preferred if preferred.exists() else max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    if DEFAULT_LOG is None:
        ap.error('no engine log found; specify --log')
    # Bind before starting collectors so a port conflict cannot write history.
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    load_history()
    stop = threading.Event()
    jobs = [(engine_tail, (stop,)), (sampler, (stop, args.disk, args.poll))]
    jobs.append((monitor_tail, (Path(args.monitor).expanduser(), stop)) if args.monitor else (gpu_sampler, (stop, args.gpu, args.poll)))
    threads = [threading.Thread(target=fn, args=params, daemon=True) for fn, params in jobs]
    for thread in threads:
        thread.start()
    print(f'[llm-scope] log: {DEFAULT_LOG}\n[llm-scope] dashboard: http://{args.host}:{args.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        for thread in threads:
            thread.join(timeout=6)


if __name__ == '__main__':
    main()
