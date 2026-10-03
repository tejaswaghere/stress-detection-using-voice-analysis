"""
download_cremad.py — Fetch the CREMA-D audio (7,442 WAV files, ~600 MB) into data/CREMA-D/.

CREMA-D (Cao et al., 2014) is distributed through GitHub with Git LFS under the
Open Database License: https://github.com/CheyneyComputerScience/CREMA-D

    python scripts/download_cremad.py            # resumable; already-downloaded files are skipped
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = 'CheyneyComputerScience/CREMA-D'
MEDIA = f'https://media.githubusercontent.com/media/{REPO}/master/AudioWAV/'
RAW = f'https://raw.githubusercontent.com/{REPO}/master/'


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def list_files() -> list[str]:
    root = json.loads(_get(f'https://api.github.com/repos/{REPO}/contents/'))
    sha = next(x['sha'] for x in root if x['name'] == 'AudioWAV')
    tree = json.loads(_get(f'https://api.github.com/repos/{REPO}/git/trees/{sha}'))
    return [x['path'] for x in tree['tree'] if x['path'].endswith('.wav')]


def fetch(name: str, out: Path, retries: int = 3) -> bool:
    dst = out / name
    if dst.exists() and dst.stat().st_size > 1000:
        return True
    for _ in range(retries):
        try:
            data = _get(MEDIA + name)
            if data[:4] == b'RIFF':  # not a Git LFS pointer or an error page
                dst.write_bytes(data)
                return True
        except OSError:
            pass
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--out', default='data/CREMA-D')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'AudioWAV').mkdir(parents=True, exist_ok=True)
    (out / 'VideoDemographics.csv').write_bytes(_get(RAW + 'VideoDemographics.csv'))

    names = list_files()
    print(f'Downloading {len(names)} files to {out / "AudioWAV"} ...')
    with ThreadPoolExecutor(args.workers) as pool:
        ok = list(pool.map(lambda n: fetch(n, out / 'AudioWAV'), names))
    failed = [n for n, k in zip(names, ok) if not k]
    print(f'Done: {sum(ok)}/{len(names)} files.' + (f' Failed (re-run to retry): {failed[:10]}' if failed else ''))


if __name__ == '__main__':
    main()
