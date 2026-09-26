"""Portable, checksummed backups of immutable data plus a committed live prefix.

Derived live rows and compressed live snapshots are rebuilt after restoration.
No hard links: a completed backup is independent of the source files.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil


def _copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest=hashlib.sha256()
    with open(source,'rb') as src, open(target,'xb') as dst:
        while block:=src.read(1024*1024):
            dst.write(block);digest.update(block)
        dst.flush();os.fsync(dst.fileno())
    return {'size':target.stat().st_size,'sha256':digest.hexdigest()}


def _sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def _publish(destination,manifest):
    for root,_,_ in os.walk(destination,topdown=False):_sync_directory(root)
    with open(destination/'backup.json.tmp','x',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.rename(destination/'backup.json.tmp',destination/'backup.json')
    _sync_directory(destination);_sync_directory(destination.parent)


def create(index,destination):
    """Copy a complete portable backup. Refuse existing output or insufficient space."""
    if index.base_path is None:raise ValueError('Portable backup requires a local base')
    destination=Path(destination).resolve()
    files=[(index.directory/name,'index/'+name) for name in ('meta.txt','anchors.bin','offs.bin','scale.bin')]
    files.append((index.base_path,'base.f16bin'))
    if index.residual_dir is not None:
        files.extend((index.residual_dir/name,'residual/'+name) for name in ('residual.meta','res512.bin'))
    else:files.append((index.directory/'blocks.bin','index/blocks.bin'))
    required=sum(p.stat().st_size for p,_ in files)
    if index.live:required+=(index.live_dir/'live.log').stat().st_size
    destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(destination.parent).free<required+16*1024*1024:
        raise OSError('Insufficient free space for a full independent backup')
    destination.mkdir()  # No overwrite, including incomplete prior backups.
    manifest={'format':'fissiondb-backup-v1','residual':index.residual_dir is not None,
              'live':index.live,'int8':index.int8,'files':{}}
    if index.live:
        live=destination/'live';live.mkdir()
        index.snapshot_live(live)
        path=live/'live.log';digest=hashlib.sha256()
        with path.open('rb') as f:
            while block:=f.read(1024*1024):digest.update(block)
        manifest['files']['live/live.log']={'size':path.stat().st_size,'sha256':digest.hexdigest()}
    for source,name in files:
        manifest['files'][name]=_copy(source,destination/name)
    manifest['float_specs']={k:v.decimals for k,v in index.float_specs.items()}
    _publish(destination,manifest)
    return destination


def restore(source,destination):
    """Verify/copy into a new directory. Return paths only after complete publication."""
    source=Path(source).resolve();destination=Path(destination).resolve()
    manifest=json.loads((source/'backup.json').read_text(encoding='utf-8'))
    if manifest.get('format') not in ('fissiondb-backup-v1', 'mangrove-backup-v1'):raise ValueError('Unsupported backup format')
    expected={'index/meta.txt','index/anchors.bin','index/offs.bin','index/scale.bin','base.f16bin'}
    expected.update({'residual/residual.meta','residual/res512.bin'} if manifest['residual'] else {'index/blocks.bin'})
    if manifest['live']:expected.add('live/live.log')
    if set(manifest['files'])!=expected:raise ValueError('Invalid backup inventory')
    required=sum(v['size'] for v in manifest['files'].values())
    destination.parent.mkdir(parents=True,exist_ok=True)
    if required<0 or shutil.disk_usage(destination.parent).free<required+16*1024*1024:
        raise OSError('Insufficient restore space')
    destination.mkdir()
    for name,metadata in manifest['files'].items():
        original=source/name
        if original.is_symlink() or source not in original.resolve().parents:raise ValueError('Invalid backup path')
        if original.stat().st_size!=metadata['size']:raise OSError('Backup size mismatch: '+name)
        if _copy(original,destination/name)!=metadata:raise OSError('Backup checksum mismatch: '+name)
    _publish(destination,manifest)
    from .metatypes import FloatSpec
    return {'int8':manifest['int8'],'float_specs':{k:FloatSpec(v) for k,v in manifest.get('float_specs',{}).items()},'directory':destination/'index','base_path':destination/'base.f16bin',
            'residual_dir':destination/'residual' if manifest['residual'] else None,
            'live_dir':destination/'live' if manifest['live'] else None}
