"""Section-scoped edit operations on word/document.xml (tracked changes by default).
Ops address a section (e.g. "3.2") and a block ordinal inside it; every op carries `expect` (text prefix
of the target) so a stale spec aborts instead of editing the wrong paragraph.
Supported ops: replace_paragraph, replace_text, insert_after, delete_paragraph.
Existing tables/figures/math/fields are never rewritten. insert_after can clone caption fields."""
import copy, datetime, difflib, math, re, struct
from pathlib import Path
from lxml import etree
from docxlib import q, ln, NS, XML_NS, M, ptext, is_plain, resolve_section

TOK = re.compile(r'[　-〿㐀-鿿豈-﫿＀-￯]|[A-Za-z0-9_]+|\s+|.', re.S)
CJK = re.compile(r'[　-〿㐀-鿿豈-﫿＀-￯]')
BLOCK_TYPES = {'paragraph': 'paragraph', 'h1': 'h1', 'h2': 'h2', 'h3': 'h3', 'h4': 'h4',
               'list_number': 'list_number', 'list_para': 'list_para'}
CAPTION_TYPES = {'caption_figure': '圖', 'caption_table': '表'}
WP = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
PIC = 'http://schemas.openxmlformats.org/drawingml/2006/picture'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
RELPKG = 'http://schemas.openxmlformats.org/package/2006/relationships'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
WP14 = 'http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing'
IMAGE_REL = R + '/image'

class EditError(Exception): pass

class Ctx:
    def __init__(self, pkg, style_map, track, author):
        self.pkg, self.sm, self.track, self.author = pkg, style_map, track, author
        ids = [int(a) for a in pkg.doc.xpath('//@w:id', namespaces=NS) if str(a).lstrip('-').isdigit()]
        self.next_id = max(ids + [0]) + 1000
        names = [e.get(q('name'), '') for e in pkg.doc.iter(q('bookmarkStart'))]
        self.next_toc = max([int(n[4:]) for n in names if re.fullmatch(r'_Toc\d+', n)] + [0]) + 1
        self.now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    def mark(self, tag):
        e = etree.Element(q(tag))
        e.set(q('id'), str(self.next_id)); self.next_id += 1
        e.set(q('author'), self.author); e.set(q('date'), self.now)
        return e

    def bookmark(self):
        bid = str(self.next_id); self.next_id += 1
        name = f'_Toc{self.next_toc}'; self.next_toc += 1
        return bid, name

def caption_kind(p):
    instr = ''.join(e.text or '' for e in p.iter(q('instrText')))
    for typ, label in CAPTION_TYPES.items():
        if f'SEQ {label} \\* ARABIC \\s 1' in instr and 'STYLEREF 1 \\s' in instr:
            return typ
    return None

def caption_template(pkg, typ):
    """Use an existing bookmarked caption of the same kind, with one description run."""
    for p in pkg.body.iterchildren(q('p')):
        if caption_kind(p) != typ: continue
        if typ == 'caption_figure':
            style = p.find('w:pPr/w:pStyle', NS)
            if style is None or style.get(q('val')) != 'af4': continue
        if len(list(p.iter(q('bookmarkStart')))) != 1 or len(list(p.iter(q('bookmarkEnd')))) != 1: continue
        if len(_description_runs(p)) == 1: return p
    raise EditError(f'no suitable existing {typ} caption template')

def _description_runs(p):
    ends = list(p.iter(q('fldChar')))
    if len(ends) != 6 or [e.get(q('fldCharType')) for e in ends] != ['begin', 'separate', 'end'] * 2:
        return []
    last = ends[-1].getparent()
    return [r for r in last.itersiblings() if ln(r) == 'r' and r.find(q('t')) is not None]

def new_caption(ctx, block, anchor):
    typ = block['type']; text = block.get('text')
    if not isinstance(text, str) or not text.strip(): raise EditError('empty caption text')
    if text != text.strip() or re.match(r'^(?:[圖表]\s*\d|\d)', text):
        raise EditError('caption text must be a description without a number or surrounding space')
    template = caption_template(ctx.pkg, typ)
    p = copy.deepcopy(template)
    w14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
    para_id_attr = f'{{{w14}}}paraId'
    text_id_attr = f'{{{w14}}}textId'
    used = {value.upper() for el in ctx.pkg.doc.iter(q('p'))
            if (value := el.get(para_id_attr)) is not None}
    para_id = next((f'{value:08X}' for value in range(1, 0x80000000)
                    if f'{value:08X}' not in used), None)
    if para_id is None: raise EditError('no unused w14:paraId below 0x80000000')
    p.set(para_id_attr, para_id)
    if p.get(text_id_attr) is not None: p.set(text_id_attr, '77777777')
    bid, name = ctx.bookmark()
    p.find('.//' + q('bookmarkStart')).set(q('id'), bid)
    p.find('.//' + q('bookmarkStart')).set(q('name'), name)
    p.find('.//' + q('bookmarkEnd')).set(q('id'), bid)
    _description_runs(p)[0].find(q('t')).text = text
    # The displayed field results are estimates. Word recalculates both fields on F9.
    count = 0
    from docxlib import heading_level
    # Heading numbers are fields/numbering, so use the section number passed by apply_spec.
    chapter = ctx.chapter
    for el in ctx.pkg.body:
        if ln(el) == 'p' and heading_level(ctx.pkg, el) == 1: count = 0
        if ln(el) == 'p' and caption_kind(el) == typ: count += 1
        if el is anchor: break
    results = [r.find(q('t')) for r in p.iter(q('r')) if r.find(q('rPr')) is not None and r.find(q('rPr')).find(q('noProof')) is not None and r.find(q('t')) is not None]
    if len(results) != 2: raise EditError('caption template has unexpected field results')
    results[0].text, results[1].text = str(chapter), str(count + 1)
    if not ctx.track: raise EditError('caption insertion requires tracked changes')
    pPr = p.find(q('pPr'))
    if pPr is None: pPr = etree.Element(q('pPr')); p.insert(0, pPr)
    rp = pPr.find(q('rPr'))
    if rp is None: rp = etree.SubElement(pPr, q('rPr'))
    rp.append(ctx.mark('ins'))
    ins = ctx.mark('ins')
    for child in list(p):
        if child is not pPr:
            p.remove(child); ins.append(child)
    p.append(ins)
    return p

def figure_template(pkg):
    """First inline body image immediately before an af4 SEQ 圖 caption."""
    kids = list(pkg.body)
    for i in range(1, len(kids)):
        caption = kids[i]
        if ln(caption) != 'p' or caption_kind(caption) != 'caption_figure': continue
        style = caption.find('w:pPr/w:pStyle', NS)
        if style is None or style.get(q('val')) != 'af4': continue
        image = kids[i - 1]
        if ln(image) == 'p' and len(list(image.iter(f'{{{WP}}}inline'))) == 1:
            return image, caption, i - 1
    raise EditError('no inline body image immediately before an af4 SEQ 圖 caption')

def usable_width_emu(anchor):
    """The current section ends at its next sectPr; page measurements are in twips."""
    for el in (anchor, *anchor.itersiblings()):
        sect = el if el.tag == q('sectPr') else el.find('w:pPr/w:sectPr', NS)
        if sect is None: continue
        size, margins = sect.find(q('pgSz')), sect.find(q('pgMar'))
        if size is None or margins is None: raise EditError('section page size or margins missing')
        twips = int(size.get(q('w'))) - int(margins.get(q('left'))) - int(margins.get(q('right')))
        if twips <= 0: raise EditError('section usable width is not positive')
        return twips * 635  # 1 twip = 635 EMU
    raise EditError('no section properties after image insertion point')

def new_figure_image(ctx, block, anchor):
    if not ctx.track: raise EditError('figure_image requires tracked changes')
    source = block.get('path')
    if not isinstance(source, str) or not Path(source).is_absolute() or not Path(source).is_file():
        raise EditError('figure_image path must be an existing absolute PNG path')
    data = Path(source).read_bytes()
    if len(data) < 33 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR' or struct.unpack('>I', data[8:12])[0] != 13:
        raise EditError('figure_image source is not a valid PNG IHDR')
    pixel_w, pixel_h = struct.unpack('>II', data[16:24])
    if not pixel_w or not pixel_h: raise EditError('PNG dimensions must be positive')
    alt = block.get('alt')
    if not isinstance(alt, str) or not alt.strip(): raise EditError('figure_image alt is required')
    width_cm = block.get('width_cm', 15.0)
    if isinstance(width_cm, bool) or not isinstance(width_cm, (int, float)) or not math.isfinite(width_cm) or width_cm <= 0:
        raise EditError('figure_image width_cm must be a positive finite number')
    cx = round(width_cm * 360000)
    if cx > usable_width_emu(anchor): raise EditError('figure_image width_cm exceeds usable page width')
    cy = round(cx * pixel_h / pixel_w)
    template, _, _ = figure_template(ctx.pkg)
    p = copy.deepcopy(template)
    inline, = list(p.iter(f'{{{WP}}}inline'))
    blips = list(inline.iter(f'{{{A}}}blip'))
    docprs = list(inline.iter(f'{{{WP}}}docPr'))
    cnvprs = list(inline.iter(f'{{{PIC}}}cNvPr'))
    xfrm_exts = inline.findall(f'.//{{{PIC}}}spPr/{{{A}}}xfrm/{{{A}}}ext')
    extents = list(inline.iter(f'{{{WP}}}extent'))
    if any(len(items) != 1 for items in (blips, docprs, cnvprs, xfrm_exts, extents)):
        raise EditError('figure template must have one blip, docPr, cNvPr and extent pair')
    used_para_ids = {value.upper() for el in ctx.pkg.doc.iter(q('p'))
                     if (value := el.get(f'{{{W14}}}paraId')) is not None}
    para_id = next((f'{value:08X}' for value in range(1, 0x80000000)
                    if f'{value:08X}' not in used_para_ids), None)
    if para_id is None: raise EditError('no unused w14:paraId below 0x80000000')
    p.set(f'{{{W14}}}paraId', para_id)
    if p.get(f'{{{W14}}}textId') is not None: p.set(f'{{{W14}}}textId', '77777777')
    for attr in ('anchorId', 'editId'): inline.attrib.pop(f'{{{WP14}}}{attr}', None)
    for el in inline.iter(f'{{{WP}}}effectExtent'):
        for attr in ('l', 't', 'r', 'b'): el.set(attr, '0')
    for el in (extents[0], xfrm_exts[0]): el.set('cx', str(cx)); el.set('cy', str(cy))
    docpr_ids = [int(el.get('id')) for el in ctx.pkg.doc.iter(f'{{{WP}}}docPr')]
    for name in ctx.pkg.zip.namelist():
        if name.startswith('word/') and name.endswith('.xml') and name != 'word/document.xml':
            root = etree.fromstring(ctx.pkg.zip.read(name))
            docpr_ids.extend(int(el.get('id')) for el in root.iter(f'{{{WP}}}docPr'))
    docpr_id = max(docpr_ids + [0]) + 1
    docprs[0].set('id', str(docpr_id)); docprs[0].set('name', f'圖片 {docpr_id}')
    docprs[0].set('descr', alt); cnvprs[0].set('descr', alt)
    existing_media = set(ctx.pkg.zip.namelist()) | set(ctx.pkg.part_updates)
    media_numbers = [int(m.group(1)) for name in existing_media
                     if (m := re.fullmatch(r'word/media/image(\d+)\.png', name))]
    media_name = f'word/media/image{max(media_numbers + [0]) + 1}.png'
    rel_name = 'word/_rels/document.xml.rels'
    rels = etree.fromstring(ctx.pkg.part_updates.get(rel_name, ctx.pkg.zip.read(rel_name)))
    rel_numbers = [int(m.group(1)) for el in rels
                   if (m := re.fullmatch(r'rId(\d+)', el.get('Id', '')))]
    rid = f'rId{max(rel_numbers + [0]) + 1}'
    etree.SubElement(rels, f'{{{RELPKG}}}Relationship',
                     Id=rid, Type=IMAGE_REL, Target=f'media/{Path(media_name).name}')
    blips[0].set(f'{{{R}}}embed', rid)
    content_name = '[Content_Types].xml'
    content = etree.fromstring(ctx.pkg.part_updates.get(content_name, ctx.pkg.zip.read(content_name)))
    ct = 'http://schemas.openxmlformats.org/package/2006/content-types'
    if not any(el.get('Extension', '').lower() == 'png' for el in content.iter(f'{{{ct}}}Default')):
        etree.SubElement(content, f'{{{ct}}}Default', Extension='png', ContentType='image/png')
        ctx.pkg.part_updates[content_name] = etree.tostring(content, xml_declaration=True, encoding='UTF-8', standalone=True)
    ctx.pkg.part_updates[rel_name] = etree.tostring(rels, xml_declaration=True, encoding='UTF-8', standalone=True)
    ctx.pkg.part_updates[media_name] = data
    pPr = p.find(q('pPr'))
    if pPr is None: pPr = etree.Element(q('pPr')); p.insert(0, pPr)
    rp = pPr.find(q('rPr'))
    if rp is None: rp = etree.SubElement(pPr, q('rPr'))
    rp.append(ctx.mark('ins'))
    ins = ctx.mark('ins')
    for child in list(p):
        if child is not pPr:
            p.remove(child); ins.append(child)
    p.append(ins)
    return p

def _min_rpr(text):
    if not CJK.search(text): return None
    rpr = etree.Element(q('rPr')); rf = etree.SubElement(rpr, q('rFonts')); rf.set(q('hint'), 'eastAsia')
    return rpr

def _run(text, rpr, deleted=False):
    r = etree.Element(q('r'))
    if rpr is not None: r.append(copy.deepcopy(rpr))
    t = etree.SubElement(r, q('delText' if deleted else 't')); t.text = text
    t.set('{%s}space' % XML_NS, 'preserve')
    return r

def _group(chars, s, e):
    """consecutive chars sharing the same source rPr -> [(text, rpr)]"""
    out = []
    for ch, rpr in chars[s:e]:
        if out and out[-1][1] is rpr: out[-1][0].append(ch)
        else: out.append(([ch], rpr))
    return [(''.join(t), r) for t, r in out]

def rebuild_paragraph(ctx, p, new_text):
    if not is_plain(p): raise EditError('paragraph is not plain text (contains fields/math/images/etc.)')
    chars = [(ch, r.find(q('rPr'))) for r in p.findall(q('r')) for t in r.findall(q('t')) for ch in (t.text or '')]
    old = ''.join(c for c, _ in chars)
    if old == new_text: raise EditError('new text identical to current text')
    ot, nt = TOK.findall(old), TOK.findall(new_text)
    off = [0]
    for t in ot: off.append(off[-1] + len(t))
    new_children = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ot, nt, autojunk=False).get_opcodes():
        if tag == 'equal':
            for txt, rpr in _group(chars, off[i1], off[i2]): new_children.append(_run(txt, rpr))
            continue
        if tag in ('delete', 'replace') and ctx.track:
            for txt, rpr in _group(chars, off[i1], off[i2]):
                d = ctx.mark('del'); d.append(_run(txt, rpr, deleted=True)); new_children.append(d)
        if tag in ('insert', 'replace'):
            txt = ''.join(nt[j1:j2]); run = _run(txt, _min_rpr(txt))
            if ctx.track:
                i = ctx.mark('ins'); i.append(run); new_children.append(i)
            else: new_children.append(run)
    kids = list(p); pPr = [c for c in kids if ln(c) == 'pPr']
    runs = [i for i, c in enumerate(kids) if ln(c) == 'r']
    lead = [c for c in kids[:runs[0]] if ln(c) != 'pPr'] if runs else []
    trail = kids[runs[-1] + 1:] if runs else []
    for c in kids: p.remove(c)
    for c in pPr + lead + new_children + trail: p.append(c)

def new_paragraph(ctx, block):
    typ = block.get('type', 'paragraph')
    if typ not in BLOCK_TYPES: raise EditError(f'unsupported block type {typ!r}; allowed: {sorted(BLOCK_TYPES)}')
    text = block.get('text', '')
    if not text.strip(): raise EditError('empty block text')
    if typ.startswith('h') and re.match(r'^\s*\d+([.、．]\d*)*\s', text):
        raise EditError(f'heading text {text[:20]!r} starts with a number; headings are auto-numbered')
    sid = ctx.sm[BLOCK_TYPES[typ]]
    if sid not in ctx.pkg.style_name: raise EditError(f'styleId {sid!r} for {typ} not found in styles.xml')
    p = etree.Element(q('p')); pPr = etree.SubElement(p, q('pPr'))
    if sid != ctx.pkg.default_para_style:
        etree.SubElement(pPr, q('pStyle')).set(q('val'), sid)
    run = _run(text, _min_rpr(text))
    if ctx.track:
        rp = etree.SubElement(pPr, q('rPr')); rp.append(ctx.mark('ins'))
        i = ctx.mark('ins'); i.append(run); p.append(i)
    else:
        p.append(run)
    if len(pPr) == 0: p.remove(pPr)
    return p

def delete_paragraph(ctx, p):
    if not is_plain(p): raise EditError('paragraph is not plain text; refusing to delete')
    if not ctx.track:
        p.getparent().remove(p); return
    pPr = p.find(q('pPr'))
    if pPr is None: pPr = etree.Element(q('pPr')); p.insert(0, pPr)
    rp = pPr.find(q('rPr'))
    if rp is None:
        rp = etree.SubElement(pPr, q('rPr'))
    rp.insert(0, ctx.mark('del'))
    for r in p.findall(q('r')):
        d = ctx.mark('del'); r.addprevious(d); d.append(r)
        for t in r.findall(q('t')):
            t.tag = q('delText'); t.set('{%s}space' % XML_NS, 'preserve')

def apply_spec(pkg, spec, style_map, track=True, author='Claude'):
    """Mutates pkg.doc. Returns (heading_info, change_log)."""
    kids, head = resolve_section(pkg, spec['section'], spec.get('title'))
    sec = kids[head['index'] + 1: head['end']]
    ctx = Ctx(pkg, style_map, track, author)
    ctx.chapter = int(str(head['number']).split('.')[0]) if str(head['number']).split('.')[0].isdigit() else 0
    touched, tails, log = set(), {}, []
    def target(op):
        k = op.get('para')
        if k == -1: el = head['elem']
        else:
            if not isinstance(k, int) or not (0 <= k < len(sec)): raise EditError(f'op {op}: para {k} out of range 0..{len(sec)-1}')
            el = sec[k]
        exp = (op.get('expect') or '').strip()
        if len(exp) < 4: raise EditError(f'op {op.get("op")} para {k}: `expect` (>=4 chars of current text) is required')
        cur = ptext(el).strip() if ln(el) == 'p' else ''
        if not cur.startswith(exp): raise EditError(f'para {k}: expect {exp!r} != current {cur[:30]!r} (stale spec?)')
        return k, el
    for op in spec['ops']:
        kind = op['op']; k, el = target(op)
        if kind in ('replace_paragraph', 'replace_text', 'delete_paragraph'):
            if k == -1: raise EditError('cannot modify the heading itself')
            if id(el) in touched: raise EditError(f'para {k} targeted by more than one modifying op')
            touched.add(id(el))
        before = ptext(el)
        if kind == 'replace_paragraph':
            rebuild_paragraph(ctx, el, op['new_text']); log.append((kind, k, before, op['new_text']))
        elif kind == 'replace_text':
            if before.count(op['find']) != 1: raise EditError(f'para {k}: `find` must occur exactly once (found {before.count(op["find"])})')
            new = before.replace(op['find'], op['replace'], 1)
            rebuild_paragraph(ctx, el, new); log.append((kind, k, before, new))
        elif kind == 'delete_paragraph':
            delete_paragraph(ctx, el); log.append((kind, k, before, ''))
        elif kind == 'insert_after':
            if ln(el) == 'p' and el.find('w:pPr/w:sectPr', NS) is not None: raise EditError('cannot insert after a section-break paragraph')
            anchor = tails.get(id(el), el)
            for b in op['blocks']:
                typ = b.get('type')
                np_ = (new_caption(ctx, b, anchor) if typ in CAPTION_TYPES else
                       new_figure_image(ctx, b, anchor) if typ == 'figure_image' else new_paragraph(ctx, b))
                anchor.addnext(np_); anchor = np_
                log.append(('insert', k, '', ptext(np_) if typ in CAPTION_TYPES or typ == 'figure_image' else b['text']))
            tails[id(el)] = anchor
        else:
            raise EditError(f'unknown op {kind!r}')
    return head, log
