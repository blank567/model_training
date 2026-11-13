"""本机W&B专用CONNECT中继与SSH反向隧道守护；凭据不经过明文中继。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import select
import socket
import socketserver
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'tmp/wandb_network'
ALLOW_EXACT = {'api.wandb.ai', 'api.forge.coreweave.com', 'storage.googleapis.com'}


def allowed_target(authority):
    match = re.fullmatch(r'([A-Za-z0-9.-]+):([0-9]+)', authority)
    if not match or int(match[2]) != 443:
        return False
    host = match[1].lower().rstrip('.')
    return host in ALLOW_EXACT or host.endswith('.wandb.ai')


def header(sock, limit=16384):
    data = b''
    while b'\r\n\r\n' not in data:
        chunk = sock.recv(min(4096, limit - len(data)))
        if not chunk:
            raise ConnectionError('Incomplete CONNECT header')
        data += chunk
        if len(data) >= limit and b'\r\n\r\n' not in data:
            raise ValueError('CONNECT header too large')
    first, extra = data.split(b'\r\n\r\n', 1)
    return first, extra


class Relay(socketserver.BaseRequestHandler):
    def handle(self):
        upstream = None
        established = False
        try:
            self.request.settimeout(10)
            request, early_client = header(self.request)
            words = request.split(b'\r\n', 1)[0].decode('ascii').split()
            if len(words) != 3 or words[0] != 'CONNECT' or not allowed_target(words[1]):
                self.request.sendall(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n')
                return
            authority = words[1].lower()
            upstream = socket.create_connection(self.server.upstream, timeout=10)
            upstream.sendall(f'CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n'.encode())
            response, early_server = header(upstream)
            if response.split(b'\r\n', 1)[0].split()[1] != b'200':
                self.request.sendall(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n')
                return
            self.request.sendall(b'HTTP/1.1 200 Connection Established\r\n\r\n' + early_server)
            established = True
            if early_client:
                upstream.sendall(early_client)
            sockets = [self.request, upstream]
            for sock in sockets:
                sock.settimeout(30)
            # 只搬运加密TLS字节，不解密、不记录请求头、API Key或数据内容。
            while True:
                ready, _, _ = select.select(sockets, [], [], 180)
                if not ready:
                    break
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    (upstream if source is self.request else self.request).sendall(data)
        except (OSError, ValueError, UnicodeError, IndexError):
            if not established:
                try:
                    self.request.sendall(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n')
                except OSError:
                    pass
        finally:
            if upstream is not None:
                upstream.close()


class RelayServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def write_state(value):
    STATE.mkdir(parents=True, exist_ok=True)
    value = {**value, 'updated_unix_seconds': time.time()}
    path = STATE / 'status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


@contextmanager
def singleton():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'watcher.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def run(config_path):
    settings = json.loads(Path(config_path).read_text(encoding='utf-8'))
    log_dir = ROOT / settings['log_dir']
    if not log_dir.resolve().is_relative_to(ROOT / 'runs/assistant/system'):
        raise ValueError('Unexpected network log directory')
    log_dir.mkdir(parents=True, exist_ok=True)
    child = None
    with singleton():
        stop = STATE / 'stop.request'
        if stop.exists():
            stop.unlink()
        with RelayServer(('127.0.0.1', settings['local_relay_port']), Relay) as server:
            server.upstream = ('127.0.0.1', settings['upstream_proxy_port'])
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            state = {'status': 'starting', 'pid': os.getpid(), 'ssh_pid': None,
                     'local_relay_port': settings['local_relay_port'], 'remote_proxy_port': settings['remote_proxy_port'],
                     'upstream_proxy_port': settings['upstream_proxy_port'], 'log_dir': str(log_dir)}
            try:
                with (log_dir / 'ssh.log').open('ab') as stream:
                    while not stop.exists():
                        if child is None or child.poll() is not None:
                            state.update(status='connecting', ssh_pid=None)
                            write_state(state)
                            child = subprocess.Popen([settings['ssh_executable'], '-F', str(ROOT / settings['ssh_config']),
                                '-N', '-T', settings['remote_alias']], stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                            state['ssh_pid'] = child.pid
                        time.sleep(2)
                        state['status'] = 'running' if child.poll() is None else 'reconnecting'
                        write_state(state)
                        if child.poll() is not None:
                            for _ in range(4):
                                if stop.exists(): break
                                time.sleep(1)
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
                server.shutdown()
                state.update(status='stopped', ssh_pid=None)
                write_state(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run', 'status', 'stop'))
    parser.add_argument('--config', default=str(ROOT / 'configs/remote/wandb_network.json'))
    args = parser.parse_args()
    if args.action == 'run':
        run(args.config)
    elif args.action == 'stop':
        STATE.mkdir(parents=True, exist_ok=True)
        (STATE / 'stop.request').write_text('stop\n', encoding='utf-8')
        print('Stop requested; only the local W&B network helper will stop.')
    else:
        path = STATE / 'status.json'
        value = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {'status': 'not_started'}
        value['heartbeat_recent'] = time.time() - value.get('updated_unix_seconds', 0) < 15
        print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
