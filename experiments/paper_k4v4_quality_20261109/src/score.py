"""Use the unchanged pinned scoring and extraction rules for all methods."""
import argparse
import importlib.util
import json

from common import ROOT, PARENT, LEGACY, atomic, cases, sha


def scorers():
    # Parent loader verifies the hashes of official grading sources.
    spec = importlib.util.spec_from_file_location('parent_quality_scorers', PARENT / 'src/score.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_scorers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    args = parser.parse_args()
    grader = scorers()
    rows = {r['index']:r for r in cases()}
    folder = ROOT / 'results' / args.method
    for path in sorted(folder.glob('[0-9]*.json')):
        record = json.loads(path.read_text())
        row = rows[record['index']]
        assert record['input_ids_sha256'] == row['input_ids_sha256']
        output = folder / 'scores' / path.name
        if output.exists():
            assert json.loads(output.read_text())['generation_sha256'] == sha(path)
            continue
        if row['dataset'] == 'humaneval':
            result = grader.human_score(record['text'], row['source_row'])
        elif row['dataset'] == 'math500':
            result = grader.math_score(record['text'], row['source_row'])
        else:
            result = grader.longbench_score(record['text'], row['source_row'], row['dataset'])
        atomic(output, dict(index=row['index'], case_id=row['case_id'], dataset=row['dataset'],
                            generation_sha256=sha(path), natural_completion=record['natural_completion'],
                            thinking_complete=record['thinking_complete'], **result))


if __name__ == '__main__':
    main()
