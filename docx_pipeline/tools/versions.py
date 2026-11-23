"""File-copy version control (no git). Every version is an immutable copy: vNNN_<time>_<label>.docx
+ manifest.jsonl. versions/ carries its own .gitignore so it never enters git."""
import json, re, time
from pathlib import Path
from docxlib import sha256

class Versions:
    def __init__(self, vdir):
        self.dir = Path(vdir); self.dir.mkdir(parents=True, exist_ok=True)
        gi = self.dir / '.gitignore'
        if not gi.exists(): gi.write_text('*\n!.gitignore\n', encoding='utf-8')
        self.manifest = self.dir / 'manifest.jsonl'

    def entries(self):
        if not self.manifest.exists(): return []
        return [json.loads(l) for l in self.manifest.read_text(encoding='utf-8').splitlines() if l.strip()]

    def latest(self):
        e = self.entries(); return e[-1] if e else None

    def add(self, data: bytes, label: str, note: str = ''):
        es = self.entries(); n = len(es) + 1
        slug = re.sub(r'[^\w\-]+', '-', label, flags=re.UNICODE).strip('-')[:40] or 'version'
        name = f"v{n:03d}_{time.strftime('%Y%m%d-%H%M%S')}_{slug}.docx"
        (self.dir / name).write_bytes(data)
        ent = dict(version=f'v{n:03d}', file=name, sha256=sha256(data), parent=(es[-1]['version'] if es else None),
                   label=label, note=note, created=time.strftime('%Y-%m-%dT%H:%M:%S'))
        with open(self.manifest, 'a', encoding='utf-8') as f: f.write(json.dumps(ent, ensure_ascii=False) + '\n')
        return ent

    def sync(self, work_bytes: bytes):
        """If the working file differs from the newest version (e.g. you edited it in Word), snapshot it first."""
        lat = self.latest()
        if lat and lat['sha256'] == sha256(work_bytes): return None
        return self.add(work_bytes, 'manual-edit' if lat else 'baseline', 'auto-snapshot of working file before pipeline operation')

    def get(self, version):
        for e in self.entries():
            if e['version'] == version: return e, (self.dir / e['file']).read_bytes()
        raise KeyError(version)
