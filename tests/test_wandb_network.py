"""验证只允许W&B HTTPS、拒绝任意代理访问，并保持TLS字节原样。"""
from pathlib import Path
import socket
import socketserver
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from wandb_network import Relay, RelayServer, allowed_target, header
from sync_lora_wandb import verify_steps


@pytest.mark.parametrize('target', ['api.wandb.ai:443', 'api.forge.coreweave.com:443',
                                   'files.wandb.ai:443', 'storage.googleapis.com:443'])
def test_first_party_tls_destinations_are_allowed(target):
    assert allowed_target(target)


@pytest.mark.parametrize('target', ['example.com:443', 'api.wandb.ai.evil.com:443',
                                   'api.wandb.ai:80', '127.0.0.1:443', 'user@api.wandb.ai:443',
                                   'api.wandb.ai:443\r\nX: value'])
def test_other_destinations_and_non_tls_ports_are_denied(target):
    assert not allowed_target(target)


class EchoProxy(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(3)
        request, extra = header(self.request)
        self.server.authorities.append(request.split(b'\r\n', 1)[0])
        self.request.sendall(b'HTTP/1.1 200 Connection established\r\n\r\n')
        if extra: self.request.sendall(extra)
        while True:
            data = self.request.recv(4096)
            if not data: return
            self.request.sendall(data)


def test_connect_preserves_opaque_tls_bytes_and_does_not_open_denied_targets():
    with RelayServer(('127.0.0.1', 0), EchoProxy) as upstream, RelayServer(('127.0.0.1', 0), Relay) as relay:
        upstream.authorities = []
        relay.upstream = upstream.server_address
        threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (upstream, relay)]
        for thread in threads: thread.start()
        try:
            payload = b'\x16\x03\x01opaque-tls-payload'
            with socket.create_connection(relay.server_address, timeout=3) as client:
                client.sendall(b'CONNECT api.wandb.ai:443 HTTP/1.1\r\nHost: api.wandb.ai:443\r\n\r\n' + payload)
                reply, extra = header(client)
                assert b'200' in reply
                while len(extra) < len(payload): extra += client.recv(4096)
                assert extra == payload
            with socket.create_connection(relay.server_address, timeout=3) as client:
                client.sendall(b'CONNECT example.com:443 HTTP/1.1\r\n\r\n')
                reply, _ = header(client)
                assert b'403' in reply
            assert upstream.authorities == [b'CONNECT api.wandb.ai:443 HTTP/1.1']
        finally:
            upstream.shutdown(); relay.shutdown()


def test_success_requires_complete_cloud_history_not_only_a_zero_exit_code():
    assert verify_steps([0, 10, 3125, 3125], [0, 10, 3125, 3125], 3125)['history_sequence_verified']
    for cloud in ([], [0, 10], [0, 10, 3125], [0, 20, 3125, 3125]):
        with pytest.raises(ValueError, match='incomplete'):
            verify_steps([0, 10, 3125, 3125], cloud, 3125)
