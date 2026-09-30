"""A lightweight, read-only browser explorer for LexNet data."""

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import lexnet
from explorer_queries import LexnetQueries, _text, split_query


WEB_PATH = Path(__file__).with_name('web')
PAGE_PATH = WEB_PATH / 'explorer.html'
CSS_PATH = WEB_PATH / 'explorer.css'
SCRIPT_PATH = WEB_PATH / 'explorer.js'
PAGE = PAGE_PATH.read_text(encoding='utf8')
CSS = CSS_PATH.read_text(encoding='utf8')
SCRIPT = SCRIPT_PATH.read_text(encoding='utf8')


def _records(frame, limit=2000):
    return [
        {column: _text(value) for column, value in row.items()}
        for row in frame.head(limit).to_dict(orient='records')
    ]


def create_server(queries, data_path, host='127.0.0.1', port=0):
    """Create the local HTTP server without starting it."""
    class ExplorerHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            params = parse_qs(request.query)
            try:
                if request.path == '/':
                    return self._send(PAGE, 'text/html; charset=utf-8')
                if request.path == '/explorer.css':
                    return self._send(CSS, 'text/css; charset=utf-8')
                if request.path == '/explorer.js':
                    return self._send(SCRIPT, 'text/javascript; charset=utf-8')
                if request.path == '/favicon.ico':
                    return self._send(b'', 'image/x-icon', status=204)
                if request.path == '/api/meta':
                    return self._json({
                        'path': data_path,
                        'entries': len(queries.entries),
                        'units': len(queries.nodes),
                        'lexical_functions': sorted(set(queries.lf_names.values()), key=str.casefold),
                        'lf_hierarchy': queries.lexical_function_hierarchy(),
                        'features': sorted(set(queries.feature_names.values()), key=str.casefold),
                    })
                if request.path == '/api/inspect':
                    return self._json(queries.inspector_payload(
                        self._param(params, 'type'), self._param(params, 'id')
                    ))
                if request.path == '/api/node':
                    return self._json(queries.inspector_payload('node', self._param(params, 'id')))
                if request.path == '/api/search':
                    return self._search(params)
                return self._json({'error': 'Not found'}, status=404)
            except Exception as error:
                return self._json({'error': str(error)}, status=500)

        def _search(self, params):
            kind = self._param(params, 'kind')
            query = self._param(params, 'q')
            if kind == 'word':
                result = queries.search_words(
                    query,
                    mode=self._param(params, 'mode', 'contains'),
                    include_forms=self._param(params, 'forms') == '1',
                )
                count = result.entry_id.nunique()
                summary = f'{len(result):,} lexical units in {count:,} entries.'
            elif kind == 'lf':
                lexical_function_id = self._param(params, 'lexical_function_id')
                family_id = self._param(params, 'family_id')
                if query:
                    result = queries.search_lexical_functions(query)
                elif lexical_function_id:
                    result = queries.search_lexical_function_ids([lexical_function_id])
                elif family_id.startswith('group:'):
                    result = queries.search_lexical_function_ids(
                        queries.lexical_function_ids_for_group(int(family_id.removeprefix('group:')))
                    )
                else:
                    result = queries.search_lexical_function_ids(
                        queries.lexical_function_ids_for_family(family_id)
                    )
                summary = f'{len(result):,} lexical-function relations.'
            elif kind == 'feature':
                result = queries.search_features(
                    query, require_all=self._param(params, 'combine') == 'all'
                )
                count = result.entry_id.nunique()
                summary = f'{len(result):,} lexical units in {count:,} entries.'
            else:
                return self._json({'error': 'Unknown search type'}, status=400)
            if len(result) > 2000:
                summary += ' Showing the first 2,000 rows.'
            return self._json({'status': summary, 'rows': _records(result)})

        @staticmethod
        def _param(params, name, default=''):
            return params.get(name, [default])[0]

        def _json(self, payload, status=200):
            return self._send(
                json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                'application/json; charset=utf-8', status,
            )

        def _send(self, body, content_type, status=200):
            if isinstance(body, str):
                body = body.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, format, *args):
            return

    return ThreadingHTTPServer((host, port), ExplorerHandler)


def launch(data_path, host='127.0.0.1', port=0, open_browser=True):
    """Load a network and serve its browser interface until interrupted."""
    network = lexnet.LexicalNetwork(data_path)
    server = create_server(LexnetQueries(network.data), data_path, host, port)
    url = f'http://{host}:{server.server_port}/'
    print(f'LexNet Explorer: {url}')
    print('Press Ctrl-C to stop.')
    if open_browser:
        threading.Timer(0.2, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping LexNet Explorer.')
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description='Explore a LexNet export in a lightweight GUI.')
    parser.add_argument('data', help='Path to a LexNet data folder')
    parser.add_argument('--port', type=int, default=0, help='Local port (default: choose automatically)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically')
    args = parser.parse_args()
    launch(args.data, port=args.port, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()
