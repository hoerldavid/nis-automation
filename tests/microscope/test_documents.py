"""
Offline tests for nis._match_opened_document (the path/title matcher
behind activate_opened_document). No NIS / microscope needed.
"""
import os
import unittest

from autofrap.microscope import nis as nis_util

A = r'C:\data\run1\fov01_cycle01_survey.nd2'
B = r'C:\data\run1\fov01_cycle01_frap.nd2'
C = r'D:\other\same_name.nd2'
DOCS = [A, B]

# the two checks that rely on Windows os.path semantics (normcase
# lowercases, '\' is the separator) cannot pass on POSIX - they would
# test the platform, not the matcher
WINDOWS = os.name == 'nt'


class TestMatchOpenedDocument(unittest.TestCase):

    def test_exact_match(self):
        """exact full-name match (the normal case)"""
        self.assertEqual(nis_util._match_opened_document(A, DOCS), A)

    @unittest.skipUnless(WINDOWS, 'Windows path semantics (normcase)')
    def test_case_insensitive_match(self):
        """case-insensitive (Windows paths)"""
        self.assertEqual(nis_util._match_opened_document(A.lower(), DOCS), A)

    def test_basename_match(self):
        """base-name match, unambiguous"""
        self.assertEqual(nis_util._match_opened_document(
            os.path.basename(A), DOCS), A)

    @unittest.skipUnless(WINDOWS, 'Windows path semantics (normcase)')
    def test_basename_match_other_dir(self):
        """base-name match across different directories"""
        self.assertEqual(nis_util._match_opened_document(
            r'X:\elsewhere\fov01_cycle01_survey.nd2', DOCS), A)

    def test_ambiguous_basename_returns_none(self):
        """base-name match is ambiguous -> None (two entries share the
        base name)"""
        self.assertIsNone(nis_util._match_opened_document(
            'same_name.nd2', [C, r'E:\more\same_name.nd2']))

    def test_no_match_returns_none(self):
        """no match at all -> None"""
        self.assertIsNone(nis_util._match_opened_document(
            r'C:\data\run2\fov02_cycle01_survey.nd2', DOCS))

    def test_unsaved_document_title(self):
        """unsaved-document title (no separators, matched as-is)"""
        self.assertEqual(nis_util._match_opened_document(
            'ND Acquisition', ['ND Acquisition', A]), 'ND Acquisition')

    def test_empty_list_returns_none(self):
        """empty list -> None"""
        self.assertIsNone(nis_util._match_opened_document(A, []))


if __name__ == '__main__':
    unittest.main()
