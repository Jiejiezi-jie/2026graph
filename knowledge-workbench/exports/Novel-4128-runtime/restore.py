"""Verify and restore the supplied Novel-4128 index; standard library only.

Never calls an API, builds an index, or overwrites existing runtime directories.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(bundle):
    bundle = bundle.resolve()
    checks = json.loads((bundle / 'checksums.json').read_text(encoding='utf-8'))
    if not isinstance(checks, dict) or not checks:
        raise ValueError('Invalid checksum manifest')
    for name, expected in checks.items():
        relative = Path(name)
        path = (bundle / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(bundle):
            raise ValueError(f'Unsafe manifest path: {name}')
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Checksum mismatch or missing file: {name}')
    data_files = {p.relative_to(bundle).as_posix() for p in (bundle / 'data').rglob('*') if p.is_file()}
    if data_files != {name for name in checks if name.startswith('data/')}:
        raise ValueError('Unlisted or missing data files')
    return checks


def restore(bundle, target):
    checks = verify(bundle)
    target = target.resolve()
    destinations = [target / 'data' / name for name in ('active', 'workspace')]
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise ValueError(f'Target already exists; stop the backend and back it up first: {destination}')
        if not destination.resolve().is_relative_to(target):
            raise ValueError(f'Unsafe target path: {destination}')
    for name, destination in zip(('active', 'workspace'), destinations):
        shutil.copytree(bundle / 'data' / name, destination)
    for name, expected in checks.items():
        if name.startswith('data/') and sha256(target / name) != expected:
            raise ValueError(f'Restored checksum mismatch: {name}')
    return len([name for name in checks if name.startswith('data/')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, help='Workbench project root (contains backend/, frontend/, scripts/)')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    bundle = Path(__file__).resolve().parent
    if args.verify_only or args.target is None:
        checks = verify(bundle)
        print(f'Package verified: {len(checks)} files. No runtime data changed.')
        return
    count = restore(bundle, args.target)
    print(f'Restored and verified {count} data files into {args.target.resolve()}')
    print('Next: configure the matching environment and your own API key; start scripts/serve_web.py.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f'ERROR: {exc}')
