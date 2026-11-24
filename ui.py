import os
import json
import argparse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from main import PIIEncryptionService

def run_server(port=5000, html_file="index.html"):
    """
    Simple HTTP server serving the HTML/CSS/JS website and handling PII detection & encryption API.
    Uses standard library only (no external web frameworks required).
    """
    service = PIIEncryptionService()

    if not os.path.exists(html_file):
        raise FileNotFoundError(f"Website file '{html_file}' not found.")

    class PIIRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/index.html':
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(html_file, 'rb') as f:
                    self.wfile.write(f.read())
            elif self.path.startswith('/build/') or os.path.exists(self.path.lstrip('/')):
                file_path = self.path.lstrip('/')
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    self.send_response(200)
                    if file_path.endswith('.css'):
                        self.send_header("Content-Type", "text/css; charset=utf-8")
                    elif file_path.endswith('.js'):
                        self.send_header("Content-Type", "application/javascript; charset=utf-8")
                    else:
                        self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    with open(file_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            content_len = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_len)
            payload = json.loads(post_body.decode('utf-8'))

            if self.path == '/api/process':
                text = payload.get('text', '')
                password = payload.get('password', None)
                result = service.process_string(text, password_or_key=password)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))

            elif self.path == '/api/decrypt':
                text = payload.get('text', '')
                password = payload.get('password', None)
                decrypted = service.decrypt_text(text, password_or_key=password)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"decrypted_text": decrypted}).encode('utf-8'))

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, PIIRequestHandler)
    url = f"http://127.0.0.1:{port}"

    print("=" * 60)
    print("  PII Detection & Encryption Web Server Running")
    print(f"  URL: {url}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped. Goodbye!")

def main():
    parser = argparse.ArgumentParser(description="Run PII Website Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000)")
    args = parser.parse_args()

    run_server(port=args.port)

if __name__ == '__main__':
    main()
