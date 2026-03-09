
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MAD⇄TFS server (rebuild) — Termux friendly (v3 Dynamic Routing)

Ejecuta: python server.py
Abre:    http://127.0.0.1:8787/
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
import json, re, os, sys, gzip
import datetime as dt

PORT = 8787
AENA_URL = "https://www.aena.es/es/infovuelos.html"

UA = (
    "Mozilla/5.0 (Linux; Android 13; Mobile) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36"
)

def utcnow_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z')

def fetch(url: str) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip",
            "Connection": "close",
        },
    )
    with urlopen(req, timeout=25) as r:
        raw = r.read()
        enc = (r.headers.get("Content-Encoding") or "").lower().strip()
        if "gzip" in enc:
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
        return raw.decode("utf-8", errors="replace")

def parse_infovuelos(html_text: str, want_dest: str) -> list:
    txt = re.sub(r"\s+", " ", html_text)
    time_re = re.compile(r"([0-2]\d:[0-5]\d)")
    flight_re = re.compile(r"\b([A-Z]{2,3}\s?\d{2,4})\b")

    flights = []
    for m in re.finditer(rf"\b{re.escape(want_dest)}\b", txt, re.IGNORECASE):
        chunk = txt[max(0, m.start() - 260) : min(len(txt), m.end() + 260)]
        t = time_re.search(chunk)
        f = flight_re.search(chunk)
        if t and f:
            flights.append({
                "depTime": t.group(1),
                "flight": f.group(1).replace(" ", "").upper()
            })

    seen = set(); out = []
    for x in flights:
        k = x["depTime"]
        if k in seen: continue
        seen.add(k)
        out.append(x)

    out.sort(key=lambda x: int(x["depTime"][:2]) * 60 + int(x["depTime"][3:]))
    return out

def parse_flightstatus(html_text: str) -> list:
    txt = re.sub(r"\s+", " ", html_text)
    flight_re = re.compile(r"\b([A-Z]{2,3}\d{2,4})\b")
    time_re = re.compile(r"\b([0-2]\d:[0-5]\d)\b")

    flights = []
    for m in flight_re.finditer(txt):
        chunk = txt[max(0, m.start() - 140) : min(len(txt), m.end() + 140)]
        t = time_re.search(chunk)
        if t:
            flights.append({"depTime": t.group(1), "flight": m.group(1).upper()})

    seen = set(); out = []
    for x in flights:
        k = x["depTime"]
        if k in seen: continue
        seen.add(k)
        out.append(x)

    out.sort(key=lambda x: int(x["depTime"][:2]) * 60 + int(x["depTime"][3:]))
    return out

def filter_past_flights(flights: list, day: str) -> list:
    if day != "today":
        return flights
    now = dt.datetime.now().strftime("%H:%M")
    return [f for f in flights if f["depTime"] >= now]

def get_flights(orig: str, dest: str, day: str) -> dict:
    orig = orig.upper()
    dest = dest.upper()
    route_label = f"{orig}→{dest}"
    aena_url = f"{AENA_URL}?origin_ac={orig}&mov=S&destiny={dest}"
    fs_url = f"https://flight-status.com/route/{orig.lower()}-to-{dest.lower()}"

    flights = []
    try:
        h = fetch(aena_url)
        flights = parse_infovuelos(h, dest)
    except Exception as e:
        pass

    if flights:
        flights = filter_past_flights(flights, day)
        return {"source": "AENA", "flights": flights, "route": route_label}

    flights2 = []
    try:
        h2 = fetch(fs_url)
        flights2 = parse_flightstatus(h2)
    except Exception as e:
        pass

    if flights2:
        flights2 = filter_past_flights(flights2, day)
    return {"source": "FlightStatus" if flights2 else "None", "flights": flights2, "route": route_label}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[HTTP] %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path)
        qs = parse_qs(p.query)

        if p.path.startswith("/") and not p.path.startswith("/api/"):
            here = os.path.dirname(os.path.abspath(__file__))
            filename = p.path[1:] if p.path != "/" else "index (6).html"
            file_path = os.path.join(here, filename)
            
            if not os.path.exists(file_path):
                if filename == "index (6).html":
                    file_path = os.path.join(here, "index.html")
                if not os.path.exists(file_path):
                    return self._send(404, "File not found")
            
            ext = os.path.splitext(file_path)[1]
            content_type = "text/html; charset=utf-8"
            if ext == ".js": content_type = "application/javascript; charset=utf-8"
            elif ext == ".css": content_type = "text/css; charset=utf-8"
            
            with open(file_path, "r", encoding="utf-8") as f:
                return self._send(200, f.read(), content_type)

        if p.path == "/api/flights":
            orig = (qs.get("orig", ["MAD"])[0] or "MAD").strip()
            dest = (qs.get("dest", ["TFS"])[0] or "TFS").strip()
            day = (qs.get("day", ["today"])[0] or "today").strip()
            data = get_flights(orig, dest, day)
            return self._send(200, json.dumps(data, ensure_ascii=False), "application/json; charset=utf-8")

        return self._send(404, "Not found")

def main():
    try:
        print(f"Servidor listo: http://127.0.0.1:{PORT}")
        HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
    except OSError as e:
        print(f"ERROR al arrancar: {e}")
        raise

if __name__ == "__main__":
    main()
