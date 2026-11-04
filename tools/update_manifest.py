"""Explicitly refresh the release inventory after reviewing intentional edits."""
import json
from check_release import ROOT,files,sha,check

if __name__=='__main__':
    check(check_manifest=False)
    record=dict(release='0.1.0',precision_certified=False,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in files() if p.name!='release_manifest.json'})
    (ROOT/'release_manifest.json').write_text(json.dumps(record,indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(check(),indent=2))
