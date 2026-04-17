#!/usr/bin/env python3

import http.server
import ssl
import socketserver

PORT = 8000
DIRECTORY = "." # Serves files from the current directory
HANDLER = http.server.SimpleHTTPRequestHandler

# Create SSL context
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
# Load certificate and key files
try:
    context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
except FileNotFoundError:
    print("Error: cert.pem or key.pem not found. Please generate them first using openssl.")
    exit()

with socketserver.TCPServer(("", PORT), HANDLER) as httpd:
    # Wrap the socket with SSL
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    print(f"Serving HTTPS on port {PORT}")
    print(f"Access at: https://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
