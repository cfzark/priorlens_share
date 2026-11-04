"""Read-only release checks, bounded to this repository; no network or fits."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','.venv','__pycache__','build','dist','results','figures','.pytest_cache'}


def files():
    for base,dirs,names in os.walk(ROOT,followlinks=False):
        depth=len(Path(base).relative_to(ROOT).parts)
        if any((Path(base)/d).is_symlink() for d in dirs):
            raise ValueError('Symlink directory is not a portable release')
        dirs[:]=sorted(d for d in dirs if d not in SKIP and not d.endswith('.egg-info')) if depth<8 else []
        for name in sorted(names):
            p=Path(base)/name
            if p.suffix in {'.pyc','.so','.o'}:continue
            if p.is_symlink():raise ValueError('Symlink is not a portable release file: '+str(p))
            yield p


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check(check_manifest=True):
    paths=list(files());names={p.relative_to(ROOT).as_posix() for p in paths}
    errors=[]
    for p in paths:
        rel=p.relative_to(ROOT).as_posix()
        text=p.read_text(encoding='utf-8')
        # Split literals keep the check itself free of the forbidden paths.
        forbidden=['/ho'+'me/','/nf'+'s/','stats_'+'dept','/Users'+'/']
        if any(x in text for x in forbidden):errors.append('Machine-specific text: '+rel)
        if p.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in target or target.startswith('#'):continue
                target=target.split('#')[0]
                if not (p.parent/target).exists():errors.append(f'Broken link: {rel} -> {target}')
    source=json.loads((ROOT/'source_map.json').read_text())
    for name,meta in source['files'].items():
        if sha(ROOT/name)!=meta['release_sha256']:errors.append('Changed mapped kernel: '+name)
    data=json.loads((ROOT/'data/manifest.json').read_text())
    for name,meta in data['tables'].items():
        if sha(ROOT/'data'/name)!=meta['sha256']:errors.append('Changed data table: '+name)
    tree=ast.parse((ROOT/'src/prioreffect_v2/methods.py').read_text())
    resolver=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_method')
    allowed=next(ast.literal_eval(n.value) for n in resolver.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='allowed' for t in n.targets))
    catalog=(ROOT/'docs/options.md').read_text()
    for option in allowed:
        if '`'+option+'`' not in catalog:errors.append('Undocumented accepted option: '+option)
    if check_manifest:
        manifest=json.loads((ROOT/'release_manifest.json').read_text())['files']
        actual={p.relative_to(ROOT).as_posix():sha(p) for p in paths if p.name!='release_manifest.json'}
        if manifest!=actual:errors.append('Release manifest differs: regenerate intentionally after review')
    if errors:raise ValueError('\n'.join(errors))
    return dict(status='pass',files=len(names),method_keys=len(allowed),mapped_sources=len(source['files']),tables=len(data['tables']))


if __name__=='__main__':print(json.dumps(check(),indent=2))
