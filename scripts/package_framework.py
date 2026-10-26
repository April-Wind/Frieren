from pathlib import Path
import argparse, hashlib, json, platform, zipfile
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT.parent / 'Frieren_framework_code.zip'))
    args = parser.parse_args()
    allowed_roots = {'src', 'tests', 'configs', 'docs', 'scripts'}
    allowed_top = {'README.md', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml', '.gitignore'}
    files = []
    for p in ROOT.rglob('*'):
        rel = p.relative_to(ROOT)
        if not p.is_file() or '__pycache__' in rel.parts or p.suffix in ('.pyc', '.pt', '.pth', '.npy', '.npz'):
            continue
        if p.name == 'IMPLEMENTATION_PROVENANCE.json':
            continue
        if len(rel.parts) == 1 and p.name in allowed_top or (rel.parts[0] in allowed_roots and p.suffix in ('.py', '.md', '.json', '.toml')):
            files.append(p)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, 'frieren/' + p.relative_to(ROOT).as_posix())
    receipt = {'archive': output.name, 'files': len(files), 'bytes': output.stat().st_size, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'contains_data_or_credentials': False}
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt))
if __name__ == '__main__':
    main()
