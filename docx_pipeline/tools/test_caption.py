"""Caption and existing edit-operation checks against CAPTION_TEST_DOCX (a disposable copy)."""
import copy
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from docxlib import Package, q, ptext, resolve_section
from edit_engine import EditError, apply_spec, caption_kind, caption_template
from audit import audit, signature, PARA_ID, TEXT_ID
from docx_cli import intended_signature


class CaptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('CAPTION_TEST_DOCX')
        if not path or not Path(path).is_file():
            raise RuntimeError('CAPTION_TEST_DOCX must point to a disposable docx copy')
        cls.raw = Path(path).read_bytes()
        cls.styles = json.loads((Path(__file__).resolve().parent.parent / 'style_map.json').read_text(encoding='utf-8'))

    def setUp(self):
        self.before = Package(self.raw)
        self.after = Package(self.raw)
        kids, head = resolve_section(self.before, '3.3')
        self.expect = ptext(kids[head['index'] + 1])[:12]

    def apply(self, blocks=None, ops=None):
        spec = {'section': '3.3', 'ops': ops or [
            {'op': 'insert_after', 'para': 0, 'expect': self.expect, 'blocks': blocks}]}
        head, log = apply_spec(self.after, spec, self.styles, True, 'Test')
        parsed = Package(self.after.to_bytes())
        checks = audit(self.before, parsed, head, set(self.styles.values()), True,
                       intended_signature(self.before, head, log), spec)
        self.assertEqual([], [(name, detail) for name, ok, detail in checks if not ok])
        return parsed, spec, head, log

    def inserted(self, pkg):
        return [p for p in pkg.body if p.tag == q('p') and
                p.find('w:pPr/w:rPr/w:ins', {'w': q('p').split('}')[0][1:]}) is not None and
                caption_kind(p) is not None]

    def test_figure(self):
        pkg, _, _, _ = self.apply([{'type': 'caption_figure', 'text': '測試圖說'}])
        p, = self.inserted(pkg)
        new_id = p.get(PARA_ID)
        self.assertNotEqual(caption_template(self.before, 'caption_figure').get(PARA_ID), new_id)
        self.assertEqual(1, sum(el.get(PARA_ID) == new_id for el in pkg.doc.iter(q('p'))))
        self.assertLess(int(new_id, 16), 0x80000000)
        self.assertEqual('77777777', p.get(TEXT_ID))
        self.assertEqual('caption_figure', caption_kind(p))
        self.assertEqual(1, len(list(p.iter(q('bookmarkStart')))))
        self.assertIn('STYLEREF', ''.join(x.text or '' for x in p.iter(q('instrText'))))
        self.assertEqual(['begin', 'separate', 'end'] * 2,
                         [x.get(q('fldCharType')) for x in p.iter(q('fldChar'))])
        self.assertEqual(1, len(p.findall(q('ins'))))
        self.assertEqual(signature(self.before, 'reject'), signature(pkg, 'reject'))

    def test_figure_and_table(self):
        pkg, _, _, _ = self.apply([{'type': 'caption_figure', 'text': '測試圖說'},
                                   {'type': 'caption_table', 'text': '測試表說'}])
        caps = self.inserted(pkg)
        self.assertEqual(['caption_figure', 'caption_table'], [caption_kind(p) for p in caps])
        self.assertEqual(2, len({p.get(PARA_ID) for p in caps}))
        ids = [e.get(q('id')) for e in pkg.doc.iter(q('bookmarkStart'))]
        self.assertEqual(len(ids), len(set(ids)))

    def test_two_figures_have_unique_names(self):
        pkg, _, _, _ = self.apply([{'type': 'caption_figure', 'text': '第一張測試'},
                                   {'type': 'caption_figure', 'text': '第二張測試'}])
        names = [p.find('.//' + q('bookmarkStart')).get(q('name')) for p in self.inserted(pkg)]
        self.assertEqual(2, len(set(names)))

    def test_five_caption_para_ids_unique_and_in_range(self):
        blocks = [{'type': 'caption_figure' if i % 2 == 0 else 'caption_table',
                   'text': f'測試說明{i}'} for i in range(5)]
        pkg, _, _, _ = self.apply(blocks)
        values = [p.get(PARA_ID) for p in self.inserted(pkg)]
        self.assertEqual(5, len(set(values)))
        self.assertTrue(all(len(value) == 8 and value == value.upper() and
                            int(value, 16) < 0x80000000 for value in values))

    def test_caption_para_ids_deterministic(self):
        blocks = [{'type': 'caption_figure', 'text': '決定性測試'},
                  {'type': 'caption_table', 'text': '決定性表說'}]
        first, _, _, _ = self.apply(blocks)
        self.after = Package(self.raw)
        second, _, _, _ = self.apply(blocks)
        self.assertEqual([p.get(PARA_ID) for p in self.inserted(first)],
                         [p.get(PARA_ID) for p in self.inserted(second)])

    def test_template_para_id_reuse_fails_audit(self):
        pkg, spec, head, log = self.apply([{'type': 'caption_figure', 'text': '測試圖說'}])
        p, = self.inserted(pkg)
        p.set(PARA_ID, caption_template(self.before, 'caption_figure').get(PARA_ID))
        checks = audit(self.before, pkg, head, set(self.styles.values()), True,
                       intended_signature(self.before, head, log), spec)
        self.assertTrue(any('new caption paraIds' in name and not ok for name, ok, _ in checks))

    def test_bad_descriptions_rejected(self):
        for text in ('', '圖 3-1 已編號', '3-1 已編號'):
            with self.subTest(text=text), self.assertRaises(EditError):
                pkg = Package(self.raw)
                apply_spec(pkg, {'section': '3.3', 'ops': [{'op': 'insert_after', 'para': 0,
                    'expect': self.expect, 'blocks': [{'type': 'caption_figure', 'text': text}]}]},
                    self.styles, True, 'Test')

    def test_extra_field_fails_audit(self):
        pkg, spec, head, log = self.apply([{'type': 'caption_figure', 'text': '測試圖說'}])
        p, = self.inserted(pkg)
        ins = p.find(q('ins'))
        run = copy.deepcopy(next(r for r in ins.iter(q('r')) if r.find(q('fldChar')) is not None))
        ins.append(run)
        checks = audit(self.before, pkg, head, set(self.styles.values()), True,
                       intended_signature(self.before, head, log), spec)
        self.assertTrue(any('fldChar' in name and not ok for name, ok, _ in checks))

    def test_existing_operations(self):
        for operation in ('replace_paragraph', 'replace_text', 'insert_after', 'delete_paragraph'):
            with self.subTest(operation=operation):
                self.setUp()
                base = {'op': operation, 'para': 0, 'expect': self.expect}
                if operation == 'replace_paragraph': base['new_text'] = '替換測試段落'
                if operation == 'replace_text': base.update(find=self.expect, replace='替換測試')
                if operation == 'insert_after': base['blocks'] = [{'type': 'paragraph', 'text': '新增測試段落'}]
                self.apply(ops=[base])


if __name__ == '__main__':
    unittest.main()
