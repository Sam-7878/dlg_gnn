"""Small stdlib tests; no private manuscript, GPU, model or network required."""
from __future__ import annotations
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from r03_common import sources,sha256
from r03_finalize import gates,check_zip


class CloseoutTests(unittest.TestCase):
    def test_gates_default_hold(self):
        result=gates()
        self.assertEqual(result['Gate_P']['status'],'HOLD')
        self.assertEqual(result['Gate_J']['status'],'HOLD')
        self.assertFalse(result['PUBLICATION_EXECUTED'])
        self.assertIsNone(result['DOI'])

    def test_author_approval_alone_insufficient(self):
        result=gates(author_approved=True)
        self.assertEqual(result['Gate_P']['status'],'HOLD')
        self.assertIn('actual AI-use declaration',result['Gate_P']['blockers'])

    def test_preprint_not_intrinsic_journal_condition(self):
        result=gates(True,True,True,True,True)
        self.assertEqual(result['Gate_J']['status'],'PASS')
        self.assertFalse(result['preprint_posting_is_intrinsic_journal_gate_requirement'])
        self.assertFalse(result['PUBLICATION_EXECUTED'])
        self.assertFalse(result['JOURNAL_SUBMISSION_EXECUTED'])

    def test_rights_and_portal_remain_required(self):
        result=gates(True,True,True,False,False)
        self.assertEqual(result['Gate_J']['status'],'HOLD')
        self.assertIn('author rights/posting agreement',result['Gate_J']['blockers'])

    def test_source_allowlist_ignores_stale_outputs_and_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('main.tex','references.bib','figures/a.png','build_old/copied.tex','visual_old/page.png','packages_old/main.tex','r02.zip','data.pt','secret.json'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
            self.assertEqual({str(p.relative_to(root)) for p in sources(root)},{'main.tex','references.bib','figures/a.png'})

    def test_zip_exact_members_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'candidate.zip';data=b'known'
            with zipfile.ZipFile(p,'w') as z:z.writestr('main.tex',data)
            check_zip(p,{'main.tex':hashlib.sha256(data).hexdigest()})
            with self.assertRaises(AssertionError):check_zip(p,{'main.tex':'0'*64})
            with self.assertRaises(AssertionError):check_zip(p,{'missing.tex':hashlib.sha256(data).hexdigest()})

    def test_hash_binds_bytes_not_path(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file';p.write_bytes(b'old');before=sha256(p)
            p.write_bytes(b'new');self.assertNotEqual(before,sha256(p))


if __name__=='__main__':unittest.main()
