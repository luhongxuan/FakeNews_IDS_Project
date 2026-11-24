"""Core helpers: read a .docx as a package, index headings/sections, test paragraph editability.
Rule: never re-save through a library that rewrites unrelated parts. Image insertion may
add media and update document relationships; all other zip entries retain their bytes."""
import io, re, zipfile, hashlib
from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
XML_NS = 'http://www.w3.org/XML/1998/namespace'
NS = {'w': W}
def q(t): return '{%s}%s' % (W, t)
def ln(e): return etree.QName(e).localname
def sha256(b): return hashlib.sha256(b).hexdigest()

class Package:
    def __init__(self, src):
        self.raw = bytes(src) if isinstance(src, (bytes, bytearray)) else open(src, 'rb').read()
        self.zip = zipfile.ZipFile(io.BytesIO(self.raw))
        self.doc = etree.fromstring(self.zip.read('word/document.xml'))
        self.styles = etree.fromstring(self.zip.read('word/styles.xml'))
        self.body = self.doc.find(q('body'))
        self.part_updates = {}  # image insertion only: added media / document relationships
        self.style_name, self.default_para_style = {}, None
        for s in self.styles.findall(q('style')):
            n = s.find(q('name'))
            self.style_name[s.get(q('styleId'))] = n.get(q('val')) if n is not None else ''
            if s.get(q('type')) == 'paragraph' and s.get(q('default')) == '1':
                self.default_para_style = s.get(q('styleId'))

    def to_bytes(self):
        new_doc = etree.tostring(self.doc, xml_declaration=True, encoding='UTF-8', standalone=True)
        out = io.BytesIO()
        with zipfile.ZipFile(out, 'w') as zo:
            for info in self.zip.infolist():
                data = new_doc if info.filename == 'word/document.xml' else self.part_updates.get(info.filename)
                if data is None: data = self.zip.read(info.filename)
                zi = zipfile.ZipInfo(info.filename, info.date_time)
                zi.compress_type = info.compress_type
                zi.external_attr = info.external_attr
                zo.writestr(zi, data)
            for name, data in self.part_updates.items():
                if name not in self.zip.namelist(): zo.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
        return out.getvalue()

def ptext(p):
    return ''.join(t.text or '' for t in p.iter(q('t')))

def para_style_id(p):
    e = p.find('w:pPr/w:pStyle', NS)
    return e.get(q('val')) if e is not None else None

def heading_level(pkg, p):
    sid = para_style_id(p)
    m = re.fullmatch(r'heading (\d)', pkg.style_name.get(sid, '')) if sid else None
    return int(m.group(1)) if m else None

def index_sections(pkg):
    """Headings 1-4 are auto-numbered (numId 3); heading 5 = unnumbered front-matter titles."""
    kids = list(pkg.body)
    heads, cnt = [], [0, 0, 0, 0]
    for i, c in enumerate(kids):
        if ln(c) != 'p': continue
        lv = heading_level(pkg, c)
        if lv is None or not ptext(c).strip(): continue
        if lv <= 4:
            cnt[lv - 1] += 1
            for k in range(lv, 4): cnt[k] = 0
            num = '.'.join(str(x) for x in cnt[:lv])
        else:
            num = ''
        heads.append(dict(index=i, level=lv, number=num, title=ptext(c).strip(), elem=c))
    for k, h in enumerate(heads):
        h['end'] = len(kids) - 1  # last child = body-level sectPr
        for h2 in heads[k + 1:]:
            if h2['level'] <= h['level']:
                h['end'] = h2['index']; break
    return kids, heads

def resolve_section(pkg, key, title=None):
    kids, heads = index_sections(pkg)
    key = str(key).strip()
    cand = [h for h in heads if (h['number'] == key) or (h['number'] == '' and h['title'] == key)]
    if title is not None:
        cand = [h for h in cand if h['title'] == title.strip()]
    if len(cand) != 1:
        raise ValueError(f'section {key!r} title={title!r}: expected 1 match, found {len(cand)}')
    return kids, cand[0]

def is_plain(p):
    """Editable by the engine: only simple text runs (+bookmarks at the ends). No fields, math,
    drawings, tabs, breaks, hyperlinks, existing tracked changes, or section break."""
    if ln(p) != 'p': return False
    ok_child = {'pPr', 'r', 'bookmarkStart', 'bookmarkEnd', 'proofErr'}
    ok_run = {'rPr', 't', 'lastRenderedPageBreak'}
    run_idx = []
    for i, c in enumerate(p):
        if ln(c) not in ok_child: return False
        if ln(c) == 'r':
            if any(ln(x) not in ok_run for x in c): return False
            run_idx.append(i)
    pp = p.find(q('pPr'))
    if pp is not None and pp.find(q('sectPr')) is not None: return False
    if run_idx:
        for i, c in enumerate(p):
            if ln(c) in ('bookmarkStart', 'bookmarkEnd') and run_idx[0] < i < run_idx[-1]: return False
    return True

def block_flags(p):
    if ln(p) != 'p': return ln(p)
    f = []
    if p.find('.//' + q('drawing')) is not None or p.find('.//' + q('pict')) is not None: f.append('img')
    if p.find('.//' + q('instrText')) is not None or p.find('.//' + q('fldSimple')) is not None: f.append('field')
    if p.find('.//{%s}oMath' % M) is not None: f.append('math')
    if p.find('w:pPr/w:sectPr', NS) is not None: f.append('SECTION-BREAK')
    return ','.join(f) if f else ('plain' if is_plain(p) else 'complex')
