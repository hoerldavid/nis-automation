"""
Offline tests for nis_util._match_opened_document (the path/title matcher
behind activate_opened_document). No NIS / microscope needed.

Run: python autofrap/autofrap_bitsnpieces/test_opened_documents_match.py
"""
import os
import sys

# repo root (for nis_util) — this script lives two levels down in autofrap/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nis_util

failures = 0


def check(name, got, want):
    global failures
    ok = got == want
    if not ok:
        failures += 1
    print(('OK    ' if ok else 'FAIL  ') + f'{name}: got {got!r}, want {want!r}')


A = r'C:\data\run1\fov01_cycle01_survey.nd2'
B = r'C:\data\run1\fov01_cycle01_frap.nd2'
C = r'D:\other\same_name.nd2'
DOCS = [A, B]

# exact full-name match (the normal case)
check('exact', nis_util._match_opened_document(A, DOCS), A)

# case-insensitive (Windows paths)
check('case', nis_util._match_opened_document(A.lower(), DOCS), A)

# base-name match, unambiguous
check('basename', nis_util._match_opened_document(
    os.path.basename(A), DOCS), A)

# base-name match across different directories
check('basename-other-dir', nis_util._match_opened_document(
    r'X:\elsewhere\fov01_cycle01_survey.nd2', DOCS), A)

# base-name match is ambiguous -> None (two entries share the base name)
check('ambiguous', nis_util._match_opened_document(
    'same_name.nd2', [C, r'E:\more\same_name.nd2']), None)

# no match at all -> None
check('nomatch', nis_util._match_opened_document(
    r'C:\data\run2\fov02_cycle01_survey.nd2', DOCS), None)

# unsaved-document title (no separators, matched as-is)
check('title', nis_util._match_opened_document(
    'ND Acquisition', ['ND Acquisition', A]), 'ND Acquisition')

# empty list -> None
check('empty', nis_util._match_opened_document(A, []), None)

print()
if failures:
    print(f'{failures} failure(s)')
    sys.exit(1)
print('all passed')
