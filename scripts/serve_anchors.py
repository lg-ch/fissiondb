"""Bounded HTTP service for native anchors. Default bind: loopback only."""
import argparse
import hmac
import json
import os
import queue
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fissiondb.anchors import AnchorIndex, AnchorBatchError, IdempotencyConflict
from fissiondb.metatypes import FloatSpec


def native_where(where):
    if where is None:
        return None
    if not isinstance(where, dict):
        raise ValueError('where must be an object')
    result = dict(where)
    for field, condition in where.items():
        if isinstance(condition, dict):
            if set(condition) == {'range'} and isinstance(condition['range'], list) and len(condition['range']) == 2:
                result[field] = ('range', *condition['range'])
            elif set(condition) == {'regex'} and isinstance(condition['regex'], str):
                result[field] = ('re', condition['regex'])
            elif condition == {'exists': True}:
                result[field] = ('exists',)
            else:
                raise ValueError('Invalid filter operator')
    return result


class AnchorServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, address, index, *, workers=1, nprobe=None, rerank=400,
                 memory_bytes=800_000_000, api_key=None, max_body=16 * 1024 * 1024, s3_url=None, calibration=None, latency_budget_ms=0, code_bytes=0):
        if not 1 <= workers <= 64 or max_body < 1:
            raise ValueError('Invalid worker count or body limit')
        if address[0] not in ('127.0.0.1', 'localhost', '::1') and not api_key:
            raise ValueError('Set FISSIONDB_API_KEY for a non-loopback bind')
        self.index, self.api_key, self.max_body = index, api_key, max_body
        self._slots = threading.BoundedSemaphore(workers + 2)
        self._contexts = queue.Queue()
        self._all_contexts = []
        self._metrics_lock = threading.Lock()
        self._metrics = dict(requests=0, errors=0, request_seconds=0.0, rejected=0)
        try:
            for _ in range(workers):
                if calibration is not None:
                    if s3_url:raise ValueError('Calibrated service currently requires local storage')
                    context = index.calibrated_context(calibration,memory_bytes=memory_bytes,
                        latency_budget_ms=latency_budget_ms,code_bytes=code_bytes)
                else:
                    context = index.context(nprobe=nprobe, rerank=rerank, threads=1,
                                            memory_bytes=memory_bytes, s3_url=s3_url)
                self._contexts.put(context)
                self._all_contexts.append(context)
            super().__init__(address, Handler)
        except Exception:
            for context in self._all_contexts:
                context.close()
            raise

    def process_request(self, request, address):
        if not self._slots.acquire(blocking=False):
            with self._metrics_lock:
                self._metrics["rejected"] += 1
            try:
                request.settimeout(1)
                request.sendall(b'HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\nRetry-After: 1\r\n\r\n')
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self._slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self._slots.release()

    def server_close(self):
        super().server_close()
        for context in self._all_contexts:
            context.close()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        pass

    def handle_one_request(self):
        self._started = time.perf_counter()
        return super().handle_one_request()

    def reply(self, status, payload):
        with self.server._metrics_lock:
            self.server._metrics["requests"] += 1
            self.server._metrics["errors"] += int(status >= 400)
            self.server._metrics["request_seconds"] += time.perf_counter() - self._started
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        if self.server.api_key and not hmac.compare_digest(
                self.headers.get('X-API-Key', '').encode('utf-8'), self.server.api_key.encode('utf-8')):
            self.reply(401, {'error': 'unauthorized'})
            return False
        return True

    def do_GET(self):
        if self.path == '/health':
            try:
                self.server.index.count
                if any(not context._handle for context in self.server._all_contexts):
                    raise OSError('Query capacity unavailable; restart service')
                self.reply(200, {'status': 'ok'})
            except (OSError, RuntimeError):
                self.reply(503, {'status': 'unavailable'})
            return
        if not self.authorized():
            return
        if self.path != '/stats':
            self.reply(404, {'error': 'unknown endpoint'})
            return
        try:
            index = self.server.index
            result = index.stats()
            worker = index._maintenance
            result.update(compactions=worker.runs if worker else 0,
                          maintenance_error=worker.last_error if worker else None)
            packer = index._packer
            result.update(packs=packer.runs if packer else 0,
                          pack_error=packer.last_error if packer else None,
                          unpacked_bytes=index.unpacked_bytes)
            with self.server._metrics_lock:
                result["http"] = dict(self.server._metrics)
            self.reply(200, result)
        except (OSError, RuntimeError):
            self.reply(503, {'error': 'index unavailable'})

    def do_POST(self):
        if not self.authorized():
            return
        try:
            if self.headers.get('Transfer-Encoding'):
                self.reply(400, {'error': 'Content-Length required'})
                return
            length = int(self.headers.get('Content-Length', '-1'))
            if length < 0 or length > self.server.max_body:
                self.reply(413, {'error': 'invalid body length'})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Truncated request')
            body = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object')
            index = self.server.index
            if self.path == '/search':
                try:
                    context = self.server._contexts.get(timeout=0.1)
                except queue.Empty:
                    self.reply(503, {'error': 'search capacity exhausted'})
                    return
                try:
                    ids, scores, stats = context.search(body['qvec'], top_k=body.get('top_k', 10),
                        where=native_where(body.get('where')), allowed_ids=body.get('allowed_ids'))
                finally:
                    self.server._contexts.put(context)
                result = {'ids': ids.tolist(), 'scores': scores.tolist(), 'stats': stats}
            elif self.path == '/insert':
                result = {'doc_id': index.insert(body['vec'], body.get('metadata'),
                                                idempotency_key=body.get('idempotency_key'))}
            elif self.path == '/insert_batch':
                if len(body['vecs']) > 256:
                    raise ValueError('Batch limit is 256 vectors')
                ids = index.insert_batch(body['vecs'], body.get('metadata'),
                                         idempotency_keys=body.get('idempotency_keys'), group_commit=body.get('group_commit', True))
                result = {'doc_ids': ids.tolist()}
            elif self.path == '/update':
                index.update(body['doc_id'], body['vec'], body.get('metadata'))
                result = {'updated': body['doc_id']}
            elif self.path == '/delete':
                index.delete(body['doc_id'])
                result = {'deleted': body['doc_id']}
            elif self.path == '/metadata/add':
                index.add_metadata(body['doc_ids'], body['metadata'])
                result = {'updated': len(body['doc_ids'])}
            elif self.path == '/metadata':
                index.set_metadata(body['doc_id'], body['metadata'])
                result = {'updated': body['doc_id']}
            elif self.path == '/pack':
                index.pack_live()
                result = {'packed': True, 'unpacked_bytes': index.unpacked_bytes}
            elif self.path == '/compact':
                result = index.compact()
            else:
                self.reply(404, {'error': 'unknown endpoint'})
                return
            self.reply(200, result)
        except AnchorBatchError as exc:
            conflict = isinstance(exc.__cause__, IdempotencyConflict)
            self.reply(409 if conflict else 503,
                       {'error': str(exc), 'committed_ids': exc.committed_ids,
                        'retry_with_same_keys': not conflict})
        except IdempotencyConflict as exc:
            self.reply(409, {'error': str(exc)})
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.reply(400, {'error': str(exc)})
        except (OSError, RuntimeError, MemoryError):
            self.reply(503, {'error': 'index unavailable; retry mutations with the same idempotency keys'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--live-dir', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--calibration', help='Validated calibration JSON for this index')
    parser.add_argument('--latency-budget-ms', type=float, default=0)
    parser.add_argument('--code-bytes', type=int, default=0)
    parser.add_argument('--nprobe', type=int, default=None, help='Cells to probe; default: indexed dimension / 2, capped at the cell count')
    parser.add_argument('--rerank', type=int, default=400)
    parser.add_argument('--memory-bytes', type=int, default=800_000_000)
    parser.add_argument('--auto-compact-bytes', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--auto-compact-interval', type=float, default=30)
    parser.add_argument('--auto-pack-bytes', type=int, default=0)
    parser.add_argument('--auto-pack-interval', type=float, default=30)
    parser.add_argument('--s3-url')
    parser.add_argument('--residual-dir')
    parser.add_argument('--float-specs', default='{}', help='JSON mapping of float fields to decimal precision')
    args = parser.parse_args()
    try:
        precisions = json.loads(args.float_specs)
        if not isinstance(precisions, dict) or any(type(v) is not int or not 0 <= v <= 9 for v in precisions.values()):
            raise ValueError('Expected field -> integer precision in [0, 9]')
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    with AnchorIndex(args.index, args.base, live_dir=args.live_dir, residual_dir=args.residual_dir,
            float_specs={field: FloatSpec(decimals) for field, decimals in precisions.items()},
            auto_compact_bytes=args.auto_compact_bytes,
            auto_compact_interval=args.auto_compact_interval,
            auto_pack_bytes=args.auto_pack_bytes,
            auto_pack_interval=args.auto_pack_interval) as index:
        server = AnchorServer((args.host, args.port), index, workers=args.workers,
            nprobe=args.nprobe, rerank=args.rerank, memory_bytes=args.memory_bytes,
            calibration=json.load(open(args.calibration)) if args.calibration else None,
            latency_budget_ms=args.latency_budget_ms, code_bytes=args.code_bytes,
            api_key=os.environ.get('FISSIONDB_API_KEY'), s3_url=args.s3_url)
        def stop(*_):
            threading.Thread(target=server.shutdown, daemon=True).start()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
