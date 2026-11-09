"""Verify upstream files without writing to the source tree."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_original():
    manifest_path = ROOT / 'source_baseline/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['original_vllm_source'])
    expected = manifest['source_sha256']
    changed = []
    for relative, digest in expected.items():
        path = source / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(relative)
    if changed:
        raise RuntimeError(f'Original vLLM source differs from migration baseline: {changed}')
    return dict(source=str(source), files_checked=len(expected), changed_files=[],
                baseline_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())


if __name__ == '__main__':
    print(json.dumps(verify_original(), ensure_ascii=False, indent=2))
