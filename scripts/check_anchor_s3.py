"""Compare a bounded anchor fixture on local disk and a real S3-compatible prefix.

Without --upload, prints the upload plan and performs no network calls.
The prefix must be new; existing objects are never overwritten or deleted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import quote, urlsplit

import numpy as np


def upload_fixture(client, bucket, prefix, files):
    existing = client.list_objects_v2(Bucket=bucket, Prefix=prefix + '/', MaxKeys=1)
    if existing.get('KeyCount', 0):
        raise RuntimeError('Test prefix already contains objects; choose another prefix')
    for name, path in files.items():
        with path.open('rb') as stream:
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
            digest = hasher.hexdigest()
            stream.seek(0)
            client.put_object(Bucket=bucket, Key=f'{prefix}/{name}', Body=stream,
                              IfNoneMatch='*', Metadata={'sha256': digest})
        head = client.head_object(Bucket=bucket, Key=f'{prefix}/{name}')
        if head['ContentLength'] != path.stat().st_size:
            raise RuntimeError('Object size mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--region', required=True)
    parser.add_argument('--profile')
    parser.add_argument('--endpoint-url', help='HTTPS S3 endpoint, e.g. https://s3.fr-par.scw.cloud')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--max-upload-bytes', type=int, default=64 * 1024 * 1024)
    parser.add_argument('--nprobe', type=int, default=32)
    parser.add_argument('--rerank', type=int, default=300)
    parser.add_argument('--query-count', type=int, default=20)
    parser.add_argument('--output', type=Path, default=Path('s3-validation.json'))
    args = parser.parse_args()
    endpoint = (args.endpoint_url or f'https://s3.{args.region}.amazonaws.com').rstrip('/')
    parsed = urlsplit(endpoint)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path:
        parser.error('Use an HTTPS endpoint with hostname only, without credentials or path')
    prefix = args.prefix.strip('/')
    if not prefix or any(p in ('.', '..') for p in prefix.split('/')):
        parser.error('Use a new, nonempty test prefix')
    files = {'blocks.bin': args.index / 'blocks.bin', 'base.f16bin': args.base}
    total = sum(path.stat().st_size for path in files.values())
    if total > args.max_upload_bytes:
        parser.error(f'Upload size {total} exceeds the explicit byte budget')
    if not 1 <= args.query_count <= 100 or not 1 <= args.nprobe <= 1024 or not 10 <= args.rerank <= 1000:
        parser.error('Query count, probes or rerank exceeds the bounded test budget')
    report = {'bucket': args.bucket, 'prefix': prefix, 'region': args.region,
              'endpoint': endpoint, 'upload_bytes': total, 'executed': False}
    if not args.upload:
        print(json.dumps(report, indent=2))
        return
    import boto3
    from mangrove.anchors import AnchorIndex
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    from botocore.config import Config
    client = session.client('s3', endpoint_url=endpoint,
                            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
    upload_fixture(client, args.bucket, prefix, files)
    credentials = session.get_credentials().get_frozen_credentials()
    names = ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN', 'AWS_REGION')
    old = {name: os.environ.get(name) for name in names}
    values = (credentials.access_key, credentials.secret_key, credentials.token or '', args.region)
    try:
        os.environ.update(dict(zip(names, values)))
        url = f'{endpoint}/{quote(args.bucket, safe="")}/{quote(prefix, safe="/")}'
        dim = int(np.fromfile(args.queries, np.uint32, count=2)[1])
        vectors = np.fromfile(args.queries, np.float32, offset=8).reshape(-1, dim)[:args.query_count]
        times = []
        with AnchorIndex(args.index, args.base) as index:
            with index.context(nprobe=args.nprobe, rerank=args.rerank) as local, \
                 index.context(nprobe=args.nprobe, rerank=args.rerank, s3_url=url) as remote:
                for vector in vectors:
                    expected, scores, _ = local.search(vector)
                    start = time.perf_counter()
                    actual, remote_scores, _ = remote.search(vector)
                    times.append((time.perf_counter() - start) * 1000)
                    np.testing.assert_array_equal(actual, expected)
                    np.testing.assert_allclose(remote_scores, scores, atol=1e-6)
        report.update(executed=True, provider=parsed.hostname, queries=len(vectors),
                      ids_match=True, p50_ms=float(np.median(times)), p99_ms=float(np.percentile(times, 99)))
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
