"""PNG insertion checks against FIGURE_TEST_DOCX, a disposable report copy."""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docxlib import Package, q, ptext, resolve_section
from edit_engine import (A, PIC, R, WP, EditError, apply_spec, figure_template)
from audit import audit, signature
from docx_cli import intended_signature


ROOT = Path(__file__).resolve().parents[2]
PNG1 = ROOT / 'analysis/20261122_report_figures/fig_rules_vs_models.png'
PNG2 = ROOT / 'analysis/20261122_report_figures/fig_two_stage_assumed_accuracy.png'


class FigureImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('FIGURE_TEST_DOCX')
        if not path or not Path(path).is_file():
            raise RuntimeError('FIGURE_TEST_DOCX must point to a disposable docx copy')
        cls.raw = Path(path).read_bytes()
        cls.styles = json.loads((ROOT / 'docx_pipeline/style_map.json').read_text(encoding='utf-8'))

    def setUp(self):
        self.before = Package(self.raw)
        self.after = Package(self.raw)
        kids, head = resolve_section(self.before, '3.3')
        self.expect = ptext(kids[head['index'] + 1])[:12]

    def image(self, path=PNG1, **kwargs):
        return {'type': 'figure_image', 'path': str(path), 'width_cm': 15.0,
                'alt': '測試圖片替代文字', **kwargs}

    def apply(self, blocks):
        spec = {'section': '3.3', 'ops': [{'op': 'insert_after', 'para': 0,
                'expect': self.expect, 'blocks': blocks}]}
        head, log = apply_spec(self.after, spec, self.styles, True, 'Test')
        parsed = Package(self.after.to_bytes())
        checks = audit(self.before, parsed, head, set(self.styles.values()), True,
                       intended_signature(self.before, head, log), spec)
        self.assertEqual([], [(name, detail) for name, ok, detail in checks if not ok])
        return parsed, checks

    def recheck(self, pkg, blocks):
        _, head = resolve_section(self.before, '3.3')
        spec = {'section': '3.3', 'ops': [{'op': 'insert_after', 'para': 0,
                'expect': self.expect, 'blocks': blocks}]}
        return audit(self.before, pkg, head, set(self.styles.values()), True, spec=spec)

    def test_image_and_caption(self):
        pkg, checks = self.apply([self.image(), {'type': 'caption_figure', 'text': '測試圖說'}])
        images = [p for p in pkg.body if p.tag == q('p') and p.find('w:pPr/w:rPr/w:ins',
                  {'w': q('p').split('}')[0][1:]}) is not None and list(p.iter(f'{{{WP}}}inline'))]
        self.assertEqual(1, len(images))
        self.assertIn('word/media/image32.png', pkg.zip.namelist())
        self.assertEqual(len(list(self.before.doc.iter(q('drawing')))) + 1,
                         len(list(pkg.doc.iter(q('drawing')))))
        self.assertEqual(signature(self.before, 'reject'), signature(pkg, 'reject'))
        self.assertTrue(any(name == 'Accept-All view == intended result' and ok for name, ok, _ in checks))
        inline, = list(images[0].iter(f'{{{WP}}}inline'))
        extent, = list(inline.iter(f'{{{WP}}}extent'))
        xfrm, = inline.findall(f'.//{{{PIC}}}spPr/{{{A}}}xfrm/{{{A}}}ext')
        self.assertEqual((extent.get('cx'), extent.get('cy')), (xfrm.get('cx'), xfrm.get('cy')))
        self.assertLess(abs((int(extent.get('cx')) / int(extent.get('cy'))) / (2391 / 1093) - 1), .005)

    def test_two_images(self):
        pkg, _ = self.apply([self.image(), self.image(PNG2, alt='第二張圖片')])
        rels = [x.get(f'{{{R}}}embed') for x in pkg.doc.iter(f'{{{A}}}blip')]
        self.assertEqual(len(rels), len(set(rels)))
        self.assertIn('word/media/image32.png', pkg.zip.namelist())
        self.assertIn('word/media/image33.png', pkg.zip.namelist())

    def test_second_image_with_prior_unaccepted_revision(self):
        first, _ = self.apply([self.image(), {'type': 'caption_figure', 'text': '第一張圖說'}])
        self.before = first
        self.after = Package(first.raw)
        kids, head = resolve_section(self.before, '3.4.4')
        spec = {'section': '3.4.4', 'ops': [{'op': 'insert_after', 'para': 3,
                'expect': ptext(kids[head['index'] + 4])[:12],
                'blocks': [self.image(PNG2, alt='第二張圖片'),
                           {'type': 'caption_figure', 'text': '第二張圖說'}]}]}
        head, log = apply_spec(self.after, spec, self.styles, True, 'Test')
        parsed = Package(self.after.to_bytes())
        checks = audit(self.before, parsed, head, set(self.styles.values()), True,
                       intended_signature(self.before, head, log, 'accept'), spec)
        self.assertEqual([], [(name, detail) for name, ok, detail in checks if not ok])

    def test_changed_media_fails_audit(self):
        block = self.image()
        self.apply([block])
        media = 'word/media/image32.png'
        self.after.part_updates[media] = self.after.part_updates[media] + b'altered'
        checks = self.recheck(Package(self.after.to_bytes()), [block])
        self.assertTrue(any(name == 'new media SHA-256 equals source PNG' and not ok
                            for name, ok, _ in checks))

    def test_distorted_extent_fails_audit(self):
        block = self.image()
        pkg, _ = self.apply([block])
        image = next(p for p in pkg.body if p.tag == q('p') and
                     p.find('w:pPr/w:rPr/w:ins', {'w': q('p').split('}')[0][1:]}) is not None and
                     list(p.iter(f'{{{WP}}}inline')))
        extent = next(image.iter(f'{{{WP}}}extent'))
        extent.set('cy', str(int(extent.get('cy')) * 2))
        checks = self.recheck(pkg, [block])
        self.assertTrue(any(name == 'new image dimensions preserve PNG ratio and fit usable page width' and not ok
                            for name, ok, _ in checks))

    def test_missing_file_rejected(self):
        with self.assertRaises(EditError):
            self.apply([self.image(ROOT / 'docx_pipeline/figure_validation_20261122/no-such.png')])

    def test_non_png_rejected(self):
        with self.assertRaises(EditError):
            self.apply([self.image(ROOT / 'docx_pipeline/TASK_figure_image.md')])

    def test_too_wide_rejected(self):
        with self.assertRaises(EditError):
            self.apply([self.image(width_cm=30.0)])

    def test_missing_alt_rejected(self):
        block = self.image(); block.pop('alt')
        with self.assertRaises(EditError): self.apply([block])

    def test_missing_template_rejected(self):
        for p in self.after.doc.iter(q('p')):
            for inline in list(p.iter(f'{{{WP}}}inline')):
                inline.getparent().remove(inline)
        with self.assertRaisesRegex(EditError, 'no inline body image'):
            self.apply([self.image()])

    def test_no_image_regression(self):
        pkg, checks = self.apply([{'type': 'caption_figure', 'text': '回歸測試圖說'}])
        names = [name for name, _, _ in checks]
        self.assertEqual([
            'zip parts other than document.xml byte-identical (fonts/media/styles/numbering/headers)',
            'everything before/including the section heading is unchanged',
            'everything after the section is unchanged',
            *[f'count of <{tag}> increases by exactly {delta}' for tag, delta in (
                ('tbl', 0), ('drawing', 0), ('pict', 0), ('instrText', 8), ('fldChar', 6),
                ('bookmarkStart', 1), ('bookmarkEnd', 1), ('sectPr', 0), ('hyperlink', 0))],
            'count of equations preserved', 'tracked-change ids unique',
            'bookmark ids/names unique and paired',
            'inserted content: style_map styles only, no direct formatting',
            'inserted caption types and template structures match spec',
            'new caption paraIds are valid, unused, and unique',
            'new caption textIds use 77777777 when template has textId',
            'Reject-All view == original document', 'Accept-All view == intended result'], names)
        self.assertEqual(set(pkg.zip.namelist()), set(self.before.zip.namelist()))
        self.assertEqual({}, self.after.part_updates)


if __name__ == '__main__':
    unittest.main()
