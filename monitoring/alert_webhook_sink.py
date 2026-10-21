"""Minimal Alertmanager webhook sink for local alert-delivery drills.

Logs every notification body as one JSON line on stdout and answers 200, so
`docker compose logs alert-webhook-sink` shows exactly what on-call would get.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            payload = {"raw": body.decode("utf-8", "replace")}
        print(json.dumps({"path": self.path, "payload": payload}), flush=True)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9095
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
