"""Serve this folder on a local port and open the calculator in your browser.

Double-clicking the .html works too, but Chrome and Edge refuse to start Web
Workers on a file:// page, which drops the search from every core to one. Served
over http they are allowed, so this is the fast way in. Nothing leaves the
machine: it binds to 127.0.0.1 only.
"""

import http.server
import os
import threading
import webbrowser

PAGE = "KCD2-Dice-Calculator.html"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
url = "http://127.0.0.1:%d/%s" % (httpd.server_address[1], PAGE)
threading.Timer(0.3, webbrowser.open, [url]).start()
print("serving " + url)
print("close this window when you are done")
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    pass
