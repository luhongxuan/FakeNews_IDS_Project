import sys, re, collections, zipfile
from lxml import etree
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ns={'w':W}
q=lambda t:'{%s}%s'%(W,t)
z=zipfile.ZipFile(sys.argv[1])
doc=etree.fromstring(z.read('word/document.xml'))
sty=etree.fromstring(z.read('word/styles.xml'))
body=doc.find(q('body'))
# style table
styles={}
for s in sty.findall(q('style')):
    n=s.find(q('name'));b=s.find(q('basedOn'))
    styles[s.get(q('styleId'))]=(s.get(q('type')),n.get(q('val')) if n is not None else None,b.get(q('val')) if b is not None else None,s)
dd=sty.find(q('docDefaults'))
print('docDefaults:',etree.tostring(dd).decode()[:900])
print('\n#top-level body children:',collections.Counter(etree.QName(c).localname for c in body))
# para style usage
use=collections.Counter(); direct=collections.Counter()
paras=body.iter(q('p'))
for p in paras:
    ps=p.find('w:pPr/w:pStyle',ns); sid=ps.get(q('val')) if ps is not None else '(Normal-default)'
    use[sid]+=1
    # direct formatting
    if p.find('w:pPr/w:ind',ns) is not None or p.find('w:pPr/w:spacing',ns) is not None or p.find('w:pPr/w:jc',ns) is not None: direct[sid]+=1
print('\n#paragraph style usage (count, w/ direct pPr fmt):')
for k,v in use.most_common(): print(f'  {k:28} {styles.get(k,(0,"?"))[1]!s:28} {v:5} direct={direct[k]}')
runfmt=collections.Counter()
for r in body.iter(q('r')):
    rp=r.find(q('rPr'))
    if rp is None: runfmt['(none)']+=1; continue
    runfmt['rStyle' if rp.find(q('rStyle')) is not None else 'direct' if len(rp) else '(empty)']+=1
print('\n#run formatting:',dict(runfmt))
fonts=collections.Counter()
for rf in body.iter(q('rFonts')):
    fonts[(rf.get(q('ascii')),rf.get(q('eastAsia')))]+=1
print('\n#direct rFonts (ascii,eastAsia):',fonts.most_common(8))
sz=collections.Counter(s.get(q('val')) for s in body.iter(q('sz')))
print('#direct sz:',sz.most_common(8))
print('\n#sectPr:',len(list(body.iter(q('sectPr')))))
for sp in body.iter(q('sectPr')):
    pg=sp.find(q('pgSz'));mg=sp.find(q('pgMar'));t=sp.find(q('type'));pn=sp.find(q('pgNumType'))
    print('  ',dict(pg.attrib) if pg is not None else None, {k.split('}')[1]:v for k,v in mg.attrib.items()} if mg is not None else None, t.get(q('val')) if t is not None else None, dict(pn.attrib) if pn is not None else None, [ (etree.QName(x).localname,x.get(q('type'))) for x in sp if etree.QName(x).localname in('headerReference','footerReference')])
instr=collections.Counter()
for i in body.iter(q('instrText')): instr[(i.text or '').strip().split(' ')[0]]+=1
for f in body.iter(q('fldSimple')): instr['simple:'+f.get(q('instr')).strip().split(' ')[0]]+=1
print('\n#fields:',dict(instr))
print('#bookmarks:',len(list(body.iter(q('bookmarkStart')))),' names sample:',[b.get(q('name')) for b in body.iter(q('bookmarkStart'))][:12])
print('#sdt:',len(list(body.iter(q('sdt')))),' tables:',len(list(body.iter(q('tbl')))),' drawings:',len(list(body.iter(q('drawing')))),' pict:',len(list(body.iter(q('pict')))),' oMath:',len(list(body.iter('{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'))))
print('#tracked ins/del:',len(list(body.iter(q('ins')))),len(list(body.iter(q('del')))),' comments refs:',len(list(body.iter(q('commentReference')))),' hyperlinks:',len(list(body.iter(q('hyperlink')))),' footnoteRef:',len(list(body.iter(q('footnoteReference')))))
print('#page breaks:',len([b for b in body.iter(q('br')) if b.get(q('type'))=='page']),' pageBreakBefore:',len(list(body.iter(q('pageBreakBefore')))))
print('#empty paragraphs:',sum(1 for p in body.findall(q('p')) if not ''.join(p.itertext()).strip() and p.find('.//'+q('drawing')) is None))
