"""usage (always run with -I from anywhere):
  python -I docx_cli.py [--config PATH] status
  python -I docx_cli.py index
  python -I docx_cli.py show <section> [--title T]
  python -I docx_cli.py snapshot [--label L]
  python -I docx_cli.py apply spec.json [--dry-run] [--no-track] [--label L] [--no-smoke]
  python -I docx_cli.py restore vNNN
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docxlib import Package, index_sections, resolve_section, ptext, para_style_id, block_flags, ln, sha256
from versions import Versions
from edit_engine import apply_spec, EditError
from audit import audit, signature, smoke_open

def load_cfg(path):
    path = Path(path).resolve(); base = path.parent; c = json.loads(path.read_text(encoding='utf-8'))
    c['_base'] = base
    c['work'] = (base / c['work_file']).resolve(); c['vdir'] = (base / c['versions_dir']).resolve()
    c['rdir'] = (base / c['reports_dir']).resolve(); c['sm'] = json.loads((base / c['style_map']).read_text(encoding='utf-8'))
    return c

def intended_signature(before, head, log, base_mode='reject'):
    """expected Accept-All result computed from the ops log + untouched original (independent of the edited tree)"""
    base = signature(before, base_mode); sec0 = head['index'] + 1
    repl, ins = {}, {}
    for kind, k, b, n in log:
        idx = head['index'] if k == -1 else sec0 + k
        if kind == 'insert': ins.setdefault(idx, []).append(('p', n))
        elif kind == 'delete_paragraph': repl[idx] = None
        else: repl[idx] = ('p', n)
    out = []
    for i, sg in enumerate(base):
        if i in repl:
            if repl[i] is not None: out.append(repl[i])
        else: out.append(sg)
        out += ins.get(i, [])
    return out

def atomic_write(path, data):
    tmp = Path(str(path) + '.tmp'); tmp.write_bytes(data); os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default=str(Path(__file__).resolve().parent.parent / 'config.json'))
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('status'); sp.add_parser('index')
    s = sp.add_parser('show'); s.add_argument('section'); s.add_argument('--title')
    s = sp.add_parser('snapshot'); s.add_argument('--label', default='manual-snapshot')
    s = sp.add_parser('apply'); s.add_argument('spec'); s.add_argument('--dry-run', action='store_true'); s.add_argument('--no-track', action='store_true'); s.add_argument('--label'); s.add_argument('--no-smoke', action='store_true')
    s = sp.add_parser('restore'); s.add_argument('version')
    a = ap.parse_args(); cfg = load_cfg(a.config); V = Versions(cfg['vdir'])
    work = cfg['work'].read_bytes()

    if a.cmd == 'status':
        lat = V.latest(); print('working file:', cfg['work']); print('sha256     :', sha256(work)[:16])
        print('latest ver :', lat['version'] if lat else None, '| in sync' if lat and lat['sha256'] == sha256(work) else '| WORKING FILE DIFFERS from latest version (will be auto-snapshotted before any operation)')
        for e in V.entries(): print(' ', e['version'], e['created'], e['label'], '-', e['file'])
    elif a.cmd == 'index':
        pkg = Package(work); kids, heads = index_sections(pkg)
        for h in heads: print(f"{'  ' * (h['level'] - 1 if h['level'] < 5 else 0)}{h['number'] or '·'} {h['title']}  [body {h['index']}-{h['end']}]")
    elif a.cmd == 'show':
        pkg = Package(work); kids, h = resolve_section(pkg, a.section, a.title)
        print(f"# section {h['number'] or '·'} {h['title']}  (para -1 = the heading itself)")
        for k, el in enumerate(kids[h['index'] + 1: h['end']]):
            st = pkg.style_name.get(para_style_id(el), 'Normal') if ln(el) == 'p' else ln(el)
            print(f"[{k:3}] {st[:14]:14} {block_flags(el):12} {ptext(el).strip()[:70] if ln(el) == 'p' else ''}")
    elif a.cmd == 'snapshot':
        e = V.sync(work)
        if e is None: e = V.add(work, a.label, 'explicit snapshot (identical to previous version)')
        print('saved', e['version'], e['file'])
    elif a.cmd == 'restore':
        V.sync(work); ent, data = V.get(a.version)
        new = V.add(data, f'restore-{a.version}', f'working file restored from {a.version}')
        atomic_write(cfg['work'], data); print('restored', a.version, '->', new['version'])
    elif a.cmd == 'apply':
        spec = json.loads(Path(a.spec).read_text(encoding='utf-8'))
        track = cfg['track_changes'] and not a.no_track
        before = Package(work); after = Package(work)
        try: head, log = apply_spec(after, spec, cfg['sm'], track, cfg['author'])
        except (EditError, ValueError, KeyError) as ex: print('ABORTED (nothing written):', ex); sys.exit(2)
        data = after.to_bytes(); after2 = Package(data)            # re-parse the serialized result
        has_image = any(b.get('type') == 'figure_image' for op in spec['ops'] if op.get('op') == 'insert_after'
                        for b in op.get('blocks', []))
        res = audit(before, after2, head, set(cfg['sm'].values()), track,
                    intended_signature(before, head, log, 'accept' if has_image else 'reject'), spec)
        if not a.no_smoke: res += smoke_open(data)
        ok = all(r[1] for r in res)
        print(f"section {head['number']} {head['title']} | track_changes={track} | {len(log)} change(s)")
        for n, o, d in res: print(('  PASS ' if o else '  FAIL ') + n + (f'  [{d}]' if d and not o else (f'  [{d}]' if d else '')))
        if not ok: print('ABORTED (audit failed, nothing written)'); sys.exit(3)
        if a.dry_run: print('dry-run: nothing written'); return
        pre = V.sync(work)
        ent = V.add(data, a.label or f"edit-{head['number'] or head['title']}", json.dumps(spec, ensure_ascii=False)[:2000])
        cfg['rdir'].mkdir(parents=True, exist_ok=True)
        lines = [f"# {ent['version']} — section {head['number']} {head['title']}", f"- parent: {ent['parent']}  | tracked changes: {track}", f"- file: versions/{ent['file']}", '']
        for kind, k, b, n in log:
            lines += [f'## {kind} (para {k})', f'- before: {b}', f'- after : {n}', '']
        lines += ['## audit'] + [f"- {'PASS' if o else 'FAIL'} {n}" for n, o, d in res]
        lines += ['', '> 欄位暫存數字、圖目錄與表目錄尚未由管線更新；需在 Word 按 Ctrl+A、F9 更新。']
        (cfg['rdir'] / f"changes_{ent['version']}.md").write_text('\n'.join(lines), encoding='utf-8')
        try: atomic_write(cfg['work'], data); print('working file updated')
        except OSError as ex: print('WARNING: could not overwrite the working file (open in Word?):', ex, '\n-> result is saved as', ent['file'])
        print('saved', ent['version'], ent['file'])

if __name__ == '__main__': main()
