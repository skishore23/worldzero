"""Local-only observatory server. No shell execution, uploaded code, or LLM calls."""
from dataclasses import replace
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import json
import threading

from .core import Config, Law
from .experiment import simulate,inheritance
from .viewer import write_report


def serve(results_dir: Path, port: int=8765) -> None:
    path=Path(results_dir)/'observatory.html'
    write_report(Path(results_dir),path)
    lock=threading.BoundedSemaphore(2)
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,body,content='application/json'):
            encoded=body if isinstance(body,bytes) else json.dumps(body).encode()
            self.send_response(status);self.send_header('Content-Type',content)
            self.send_header('Content-Length',str(len(encoded)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(encoded)
        def do_GET(self):
            if self.path in ('/','/observatory.html'):
                self.send(200,path.read_bytes(),'text/html; charset=utf-8')
            else:self.send(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/api/run':return self.send(404,{'error':'Not found'})
            origin=self.headers.get('Origin')
            allowed={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
            if origin and origin not in allowed:return self.send(403,{'error':'Cross-origin request rejected'})
            if not self.headers.get('Content-Type','').startswith('application/json'):
                return self.send(415,{'error':'Use application/json'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Invalid body size')
                body=json.loads(self.rfile.read(length))
                seed=body.get('seed');kind=body.get('policy');condition=body.get('condition')
                if type(seed)is not int or not 0<=seed<2**53:raise ValueError('Invalid seed')
                if kind not in {'random','forager','experimenter','informed'}:raise ValueError('Only local scripted controls are available')
                if condition not in {'easy','pressure','severe','null'}:raise ValueError('Invalid condition')
                if not lock.acquire(blocking=False):return self.send(429,{'error':'Two experiments already running'})
                try:
                    m={'easy':.24,'pressure':.32,'severe':.48,'null':.32}[condition]
                    cfg=replace(Config(),metabolism=m)
                    law=Law((0,1),'null') if condition=='null' else None
                    w,r,tr=simulate(seed,kind,cfg,law=law,capture=True)
                    pair,children=inheritance(w,capture=True)
                    self.send(200,{'example':{'seed':seed,'run':'live-single-run','episode':r,'parent':tr,'inheritance':pair,'successors':children}})
                finally:lock.release()
            except (ValueError,TypeError,json.JSONDecodeError) as exc:self.send(400,{'error':str(exc)})
            except Exception as exc:self.send(500,{'error':f'{type(exc).__name__}: {exc}'})
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    print(f'WorldZero observatory: http://127.0.0.1:{server.server_port} (local controls only)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
