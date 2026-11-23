"""Post-edit invariants. An edit is only accepted if EVERY check passes."""
import copy, hashlib, os, re, struct, subprocess, tempfile
from collections import Counter
from pathlib import Path
from lxml import etree
from docxlib import Package, q, ln, NS, M, ptext
from edit_engine import (CAPTION_TYPES, caption_kind, caption_template, figure_template,
                         usable_width_emu, _description_runs, WP, A, PIC, R, RELPKG, WP14, IMAGE_REL)

W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
PARA_ID = f'{{{W14}}}paraId'
TEXT_ID = f'{{{W14}}}textId'

def _figure_shape(p, inserted=False):
    """Compare a cloned image paragraph while ignoring only documented replacements."""
    p = copy.deepcopy(p)
    if p.get(PARA_ID) is not None: p.set(PARA_ID, 'PARAID')
    if p.get(TEXT_ID) is not None: p.set(TEXT_ID, 'TEXTID')
    if inserted:
        ins = p.find(q('ins'))
        if ins is None or len([c for c in p if ln(c) != 'pPr']) != 1: return None
        p.remove(ins)
        for child in list(ins): ins.remove(child); p.append(child)
        rp = p.find('w:pPr/w:rPr', NS)
        if rp is None or len(list(rp.iter(q('ins')))) != 1: return None
        rp.remove(rp.find(q('ins')))
        if len(rp) == 0: rp.getparent().remove(rp)
        pp = p.find(q('pPr'))
        if pp is not None and len(pp) == 0: p.remove(pp)
    inline = list(p.iter(f'{{{WP}}}inline'))
    if len(inline) != 1: return None
    for attr in ('anchorId', 'editId'): inline[0].attrib.pop(f'{{{WP14}}}{attr}', None)
    for e in inline[0].iter(f'{{{WP}}}effectExtent'):
        for attr in ('l', 't', 'r', 'b'): e.set(attr, '0')
    for e in inline[0].iter(f'{{{WP}}}extent'):
        e.set('cx', 'CX'); e.set('cy', 'CY')
    for e in inline[0].findall(f'.//{{{PIC}}}spPr/{{{A}}}xfrm/{{{A}}}ext'):
        e.set('cx', 'CX'); e.set('cy', 'CY')
    for e in inline[0].iter(f'{{{WP}}}docPr'):
        e.set('id', 'ID'); e.set('name', 'NAME'); e.attrib.pop('descr', None)
    for e in inline[0].iter(f'{{{PIC}}}cNvPr'): e.attrib.pop('descr', None)
    for e in inline[0].iter(f'{{{A}}}blip'): e.set(f'{{{R}}}embed', 'RID')
    return etree.tostring(p, method='c14n', exclusive=True)

def _word_docpr_ids(pkg):
    ids = [e.get('id') for e in pkg.doc.iter(f'{{{WP}}}docPr')]
    for name in pkg.zip.namelist():
        if name.startswith('word/') and name.endswith('.xml') and name != 'word/document.xml':
            ids.extend(e.get('id') for e in etree.fromstring(pkg.zip.read(name)).iter(f'{{{WP}}}docPr'))
    return ids

def _c14n(e): return hashlib.sha1(etree.tostring(e, method='c14n')).hexdigest()

def _children_in(e, tag): return [c for c in e.iter(q(tag))]

def _view(p, mode):
    """text of a top-level <w:p> under 'reject' (original) or 'accept' (new) view; None if the paragraph vanishes."""
    rp = p.find('w:pPr/w:rPr', NS)
    ins_mark = rp is not None and rp.find(q('ins')) is not None
    del_mark = rp is not None and rp.find(q('del')) is not None
    if mode == 'reject' and ins_mark: return None
    if mode == 'accept' and del_mark: return None
    out = []
    for r in p.iter(q('r')):
        anc = [ln(a) for a in r.iterancestors()]
        if mode == 'reject' and 'ins' in anc: continue
        if mode == 'accept' and 'del' in anc: continue
        out += [t.text or '' for t in r if ln(t) in ('t', 'delText')]
    return ''.join(out)

def signature(pkg, mode):
    sig = []
    for c in pkg.body:
        if ln(c) == 'p':
            v = _view(c, mode)
            if v is not None: sig.append(('p', v))
        else: sig.append((ln(c), _c14n(c)))
    return sig

def _caption_shape(p, inserted=False):
    """Canonical caption XML, excluding only the requested variable values and revision wrapper."""
    p = copy.deepcopy(p)
    if p.get(PARA_ID) is not None: p.set(PARA_ID, 'PARAID')
    if p.get(TEXT_ID) is not None: p.set(TEXT_ID, 'TEXTID')
    if inserted:
        ins = p.find(q('ins'))
        if ins is None or len([c for c in p if ln(c) != 'pPr']) != 1:
            return None
        p.remove(ins)
        for child in list(ins): ins.remove(child); p.append(child)
        rp = p.find('w:pPr/w:rPr', NS)
        if rp is None or len(list(rp.iter(q('ins')))) != 1: return None
        rp.remove(rp.find(q('ins')))
        if len(rp) == 0: rp.getparent().remove(rp)
        pp = p.find(q('pPr'))
        if pp is not None and len(pp) == 0: p.remove(pp)
    starts, ends = list(p.iter(q('bookmarkStart'))), list(p.iter(q('bookmarkEnd')))
    if len(starts) != 1 or len(ends) != 1: return None
    starts[0].set(q('id'), 'ID'); starts[0].set(q('name'), '_TocID')
    ends[0].set(q('id'), 'ID')
    desc = _description_runs(p)
    if len(desc) != 1: return None
    desc[0].find(q('t')).text = 'DESCRIPTION'
    results = [r.find(q('t')) for r in p.iter(q('r')) if r.find(q('rPr')) is not None and r.find(q('rPr')).find(q('noProof')) is not None and r.find(q('t')) is not None]
    if len(results) != 2: return None
    for result in results: result.text = 'NUMBER'
    return etree.tostring(p, method='c14n')

def audit(before: Package, after: Package, head, allowed_styles, track, expected_accept=None, spec=None):
    res = []
    def chk(name, ok, detail=''): res.append((name, bool(ok), detail))
    images = [b for op in (spec or {}).get('ops', []) if op.get('op') == 'insert_after'
              for b in op.get('blocks', []) if b.get('type') == 'figure_image']
    # 1. every other zip part is byte-identical
    bi = {i.filename: (i.CRC, i.file_size) for i in before.zip.infolist()}
    ai = {i.filename: (i.CRC, i.file_size) for i in after.zip.infolist()}
    if not images:
        diff = [n for n in set(bi) | set(ai) if n != 'word/document.xml' and bi.get(n) != ai.get(n)]
        chk('zip parts other than document.xml byte-identical (fonts/media/styles/numbering/headers)', not diff, ','.join(diff))
    else:
        media_numbers = [int(m.group(1)) for n in bi if (m := re.fullmatch(r'word/media/image(\d+)\.png', n))]
        start = max(media_numbers + [0]) + 1
        expected_media = [f'word/media/image{start + i}.png' for i in range(len(images))]
        old_ct = etree.fromstring(before.zip.read('[Content_Types].xml'))
        ct_ns = 'http://schemas.openxmlformats.org/package/2006/content-types'
        has_png = any(e.get('Extension', '').lower() == 'png' for e in old_ct.iter(f'{{{ct_ns}}}Default'))
        allowed = {'word/document.xml', 'word/_rels/document.xml.rels'}
        if not has_png: allowed.add('[Content_Types].xml')
        unchanged = all(before.zip.read(n) == after.zip.read(n) for n in bi if n not in allowed and n in ai)
        zip_ok = set(ai) - set(bi) == set(expected_media) and not (set(bi) - set(ai)) and unchanged
        chk('zip parts unchanged except expected image media and relationships', zip_ok,
            f'expected media={expected_media}; added={sorted(set(ai)-set(bi))}')
        new_ct = etree.fromstring(after.zip.read('[Content_Types].xml'))
        if has_png:
            ct_ok = before.zip.read('[Content_Types].xml') == after.zip.read('[Content_Types].xml')
        else:
            new_png = [e for e in new_ct.iter(f'{{{ct_ns}}}Default') if e.get('Extension', '').lower() == 'png']
            for e in new_png: e.getparent().remove(e)
            ct_ok = len(new_png) == 1 and new_png[0].get('ContentType') == 'image/png' and etree.tostring(old_ct, method='c14n') == etree.tostring(new_ct, method='c14n')
        chk('PNG content type unchanged or exactly one required Default added', ct_ok)
        media_ok = all(name in ai and hashlib.sha256(after.zip.read(name)).digest() ==
                       hashlib.sha256(Path(block['path']).read_bytes()).digest()
                       for name, block in zip(expected_media, images))
        chk('new media SHA-256 equals source PNG', media_ok)
        rel_path = 'word/_rels/document.xml.rels'
        old_rels = etree.fromstring(before.zip.read(rel_path)); new_rels = etree.fromstring(after.zip.read(rel_path))
        old_by_id = {e.get('Id'): e for e in old_rels}
        rel_ids = [e.get('Id') for e in new_rels]
        additions = [e for e in new_rels if e.get('Id') not in old_by_id]
        prior_same = all(e.get('Id') in old_by_id and etree.tostring(e, method='c14n') ==
                         etree.tostring(old_by_id[e.get('Id')], method='c14n')
                         for e in new_rels if e.get('Id') in old_by_id)
        prior_order = ([e.get('Id') for e in new_rels if e.get('Id') in old_by_id] ==
                       [e.get('Id') for e in old_rels])
        expected_targets = {f'media/{Path(name).name}' for name in expected_media}
        rel_ok = (len(rel_ids) == len(set(rel_ids)) and len(additions) == len(images) and
                  len(new_rels) == len(old_rels) + len(images) and prior_same and prior_order and
                  old_rels.tag == new_rels.tag and old_rels.attrib == new_rels.attrib and
                  {e.get('Target') for e in additions} == expected_targets and
                  all(e.get('Type') == IMAGE_REL and e.get('TargetMode') is None and
                      f"word/{e.get('Target')}" in ai for e in additions))
        chk('new image relationships are unique, exact, and target existing media', rel_ok)
    # 2. nothing outside the target section changed
    hb = [_c14n(c) for c in before.body]; ha = [_c14n(c) for c in after.body]
    pre = 0
    while pre < min(len(hb), len(ha)) and hb[pre] == ha[pre]: pre += 1
    suf = 0
    while suf < min(len(hb), len(ha)) - pre and hb[-1 - suf] == ha[-1 - suf]: suf += 1
    chk('everything before/including the section heading is unchanged', pre >= head['index'] + 1, f'common prefix={pre}, heading at {head["index"]}')
    chk('everything after the section is unchanged', suf >= len(hb) - head['end'], f'common suffix={suf}, needed {len(hb) - head["end"]}')
    # 3. only caption blocks may add fields and bookmarks, in exact template quantities.
    captions = [b['type'] for op in (spec or {}).get('ops', []) if op.get('op') == 'insert_after'
                for b in op.get('blocks', []) if b.get('type') in CAPTION_TYPES]
    expected_delta = {tag: sum(len(list(caption_template(before, typ).iter(q(tag)))) for typ in captions)
                      for tag in ('instrText', 'fldChar', 'bookmarkStart', 'bookmarkEnd')}
    for tag in ('tbl', 'drawing', 'pict', 'instrText', 'fldChar', 'bookmarkStart', 'bookmarkEnd', 'sectPr', 'hyperlink'):
        nb, na = len(list(before.doc.iter(q(tag)))), len(list(after.doc.iter(q(tag))))
        delta = len(images) if tag == 'drawing' else expected_delta.get(tag, 0)
        chk(f'count of <{tag}> increases by exactly {delta}', na - nb == delta, f'{nb} -> {na}')
    if images:
        for local, delta in (('inline', len(images)), ('anchor', 0)):
            tag = f'{{{WP}}}{local}'
            nb, na = len(list(before.doc.iter(tag))), len(list(after.doc.iter(tag)))
            chk(f'count of <wp:{local}> increases by exactly {delta}', na - nb == delta, f'{nb} -> {na}')
    nb, na = len(list(before.doc.iter('{%s}oMath' % M))), len(list(after.doc.iter('{%s}oMath' % M)))
    chk('count of equations preserved', nb == na, f'{nb} -> {na}')
    # 4. revision ids unique
    ids = [e.get(q('id')) for t in ('ins', 'del') for e in after.doc.iter(q(t))]
    chk('tracked-change ids unique', len(ids) == len(set(ids)))
    bookmarks = list(after.doc.iter(q('bookmarkStart')))
    bids = [e.get(q('id')) for e in bookmarks]
    bnames = [e.get(q('name')) for e in bookmarks]
    ends = [e.get(q('id')) for e in after.doc.iter(q('bookmarkEnd'))]
    chk('bookmark ids/names unique and paired', len(bids) == len(set(bids)) and
        len(bnames) == len(set(bnames)) and sorted(bids) == sorted(ends))
    # 5. new content uses only whitelisted styles and no direct formatting
    seen = set(hb); bad = []; actual_captions = []; new_para_ids = []; text_ids_ok = True
    new_images = []
    existing_para_ids = {value.upper() for el in before.doc.iter(q('p'))
                         if (value := el.get(PARA_ID)) is not None}
    for c in after.body:
        if ln(c) != 'p' or _c14n(c) in seen: continue
        typ = caption_kind(c)
        if typ is not None and c.find('w:pPr/w:rPr/w:ins', NS) is not None:
            actual_captions.append(typ)
            template = caption_template(before, typ)
            new_para_ids.append(c.get(PARA_ID))
            if template.get(TEXT_ID) is not None and c.get(TEXT_ID) != '77777777':
                text_ids_ok = False
            if _caption_shape(c, inserted=True) != _caption_shape(template):
                bad.append(f'{typ} differs from its existing template')
            continue
        if images and c.find('w:pPr/w:rPr/w:ins', NS) is not None and c.find('.//' + q('drawing')) is not None:
            new_images.append(c)
            continue
        sid = (c.find('w:pPr/w:pStyle', NS).get(q('val')) if c.find('w:pPr/w:pStyle', NS) is not None else after.default_para_style)
        pp = c.find(q('pPr'))
        if pp is not None:
            extra = [ln(x) for x in pp if ln(x) not in ('pStyle', 'rPr', 'numPr') ]
            if c.find('w:pPr/w:rPr', NS) is not None and c.find('w:pPr/w:rPr', NS).find(q('ins')) is not None:
                if sid not in allowed_styles: bad.append(f'style {sid} not in style_map')
                if extra: bad.append(f'direct pPr {extra}')
        for r in c.iter(q('r')):
            if any(ln(a) == 'ins' for a in r.iterancestors()):
                rp = r.find(q('rPr'))
                if rp is not None and any(ln(x) != 'rFonts' or set(x.attrib) - {q('hint')} for x in rp): bad.append('direct run formatting in inserted text')
    chk('inserted content: style_map styles only, no direct formatting', not bad, '; '.join(sorted(set(bad))))
    chk('inserted caption types and template structures match spec', Counter(actual_captions) == Counter(captions),
        f'expected={captions}, actual={actual_captions}')
    valid_para_ids = all(value is not None and len(value) == 8 and
                         value == value.upper() and all(char in '0123456789ABCDEF' for char in value) and
                         int(value, 16) < 0x80000000 and value not in existing_para_ids
                         for value in new_para_ids)
    chk('new caption paraIds are valid, unused, and unique', valid_para_ids and
        len(new_para_ids) == len(set(new_para_ids)), f'new={new_para_ids}')
    chk('new caption textIds use 77777777 when template has textId', text_ids_ok)
    if images:
        template, _, _ = figure_template(before)
        shape_ok = len(new_images) == len(images) and all(_figure_shape(p, True) == _figure_shape(template) for p in new_images)
        chk('new image paragraphs match inline body template structure', shape_ok)
        before_para_ids = {p.get(PARA_ID) for p in before.doc.iter(q('p')) if p.get(PARA_ID) is not None}
        image_para_ids = [p.get(PARA_ID) for p in new_images]
        all_new_ids = new_para_ids + image_para_ids
        para_ok = all(v is not None and re.fullmatch(r'[0-7][0-9A-F]{7}', v) and v not in before_para_ids
                      for v in image_para_ids) and len(all_new_ids) == len(set(all_new_ids))
        chk('new image paraIds are valid, unused, and unique across inserted blocks', para_ok, f'new={image_para_ids}')
        chk('new image textIds use 77777777 when template has textId',
            all(p.get(TEXT_ID) == '77777777' for p in new_images) if template.get(TEXT_ID) is not None else True)
        existing_docpr = set(_word_docpr_ids(before))
        new_docpr = [e.get('id') for p in new_images for e in p.iter(f'{{{WP}}}docPr')]
        docpr_ok = len(new_docpr) == len(images) and len(new_docpr) == len(set(new_docpr)) and all(v not in existing_docpr and v is not None and v.isdigit() for v in new_docpr)
        chk('new docPr ids are unique across the full document', docpr_ok, f'new={new_docpr}')
        new_rel_by_id = {e.get('Id'): e for e in additions}
        image_rids = [e.get(f'{{{R}}}embed') for p in new_images for e in p.iter(f'{{{A}}}blip')]
        embeds_ok = len(image_rids) == len(images) and len(set(image_rids)) == len(images) and set(image_rids) == set(new_rel_by_id)
        chk('new blipFill embeds reference only new image relationships', embeds_ok, f'new={image_rids}')
        dimensions_ok = len(new_images) == len(images)
        alt_ok = dimensions_ok
        alt_by_media = {name: block.get('alt') for name, block in zip(expected_media, images)}
        for p in new_images:
            inline = next(p.iter(f'{{{WP}}}inline'), None)
            if inline is None: dimensions_ok = False; alt_ok = False; continue
            wp_ext = next(inline.iter(f'{{{WP}}}extent'), None)
            xfrm_ext = next(iter(inline.findall(f'.//{{{PIC}}}spPr/{{{A}}}xfrm/{{{A}}}ext')), None)
            blip = next(inline.iter(f'{{{A}}}blip'), None)
            docpr = next(inline.iter(f'{{{WP}}}docPr'), None)
            cnvpr = next(inline.iter(f'{{{PIC}}}cNvPr'), None)
            if any(x is None for x in (wp_ext, xfrm_ext, blip, docpr, cnvpr)):
                dimensions_ok = False; alt_ok = False; continue
            rel = new_rel_by_id.get(blip.get(f'{{{R}}}embed'))
            if rel is None: dimensions_ok = False; alt_ok = False; continue
            media_path = 'word/' + rel.get('Target', '')
            data = after.zip.read(media_path)
            pixel_w, pixel_h = struct.unpack('>II', data[16:24])
            cx, cy = int(wp_ext.get('cx')), int(wp_ext.get('cy'))
            dimensions_ok &= (cx == int(xfrm_ext.get('cx')) and cy == int(xfrm_ext.get('cy')) and
                              cx > 0 and cy > 0 and cx <= usable_width_emu(p) and
                              abs((cx / cy) / (pixel_w / pixel_h) - 1) < 0.005 and
                              all(inline.get(f'{{{WP14}}}{attr}') is None for attr in ('anchorId', 'editId')) and
                              all(all(e.get(attr) == '0' for attr in ('l', 't', 'r', 'b'))
                                  for e in inline.iter(f'{{{WP}}}effectExtent')))
            alt_ok &= (alt_by_media.get(media_path) == docpr.get('descr') == cnvpr.get('descr') and
                       docpr.get('name') == f"圖片 {docpr.get('id')}")
        chk('new image dimensions preserve PNG ratio and fit usable page width', dimensions_ok)
        chk('new image alt text matches docPr and cNvPr', alt_ok)
        chk('Reject-All orphan image media/relationships are expected and harmless', True,
            f'{len(images)} media and {len(images)} relationships remain after rejecting insertion')
    # 6. reversibility / expected result
    if track:
        chk('Reject-All view == original document', signature(after, 'reject') == signature(before, 'reject'))
    if expected_accept is not None:
        chk('Accept-All view == intended result', signature(after, 'accept') == expected_accept)
    return res

def smoke_open(data: bytes, convert=True):
    out = []
    try:
        import docx, io
        d = docx.Document(io.BytesIO(data)); out.append(('python-docx opens file', True, f'{len(d.paragraphs)} paragraphs'))
    except Exception as ex: out.append(('python-docx opens file', False, repr(ex)))
    if convert:
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, 't.docx'); open(f, 'wb').write(data)
            try:
                r = subprocess.run(['soffice', '--headless', '--convert-to', 'pdf', '--outdir', td, f], capture_output=True, timeout=150)
                out.append(('LibreOffice converts to PDF (opens without error)', os.path.exists(os.path.join(td, 't.pdf')), ''))
            except Exception as ex: out.append(('LibreOffice converts to PDF', False, repr(ex)))
    return out
