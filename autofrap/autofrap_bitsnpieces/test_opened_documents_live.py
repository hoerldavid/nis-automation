"""
Live test for the opened-document wrappers (TODO #25 prep).

Run at the microscope with NIS-Elements open and at least one image
document open:

  python autofrap/autofrap_bitsnpieces/test_opened_documents_live.py

Exercises (read-only apart from switching the current document, which
is restored at the end):

  - get_opened_documents: list the open documents, print them
  - get_current_document vs the list: is the current doc in the list,
    and in which form (full path vs title)
  - activate_document with the exact NIS spelling (list entry)
  - activate_opened_document with a full path and with a base name
  - if >= 2 documents are open: switch to another one, verify via
    get_current_document, switch back, verify

No documents are opened, saved, or closed.
"""
import os
import sys

# repo root (for nis_util) — this script lives two levels down in autofrap/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import nis_util

NIS = r'C:\Program Files\NIS-Elements\nis_ar.exe'

failures = 0


def check(name, ok, detail=''):
    global failures
    if not ok:
        failures += 1
    print(('OK    ' if ok else 'FAIL  ') + f'{name} {detail}')


def main():
    docs = nis_util.get_opened_documents(NIS)
    print(f'{len(docs)} open document(s):')
    for i, d in enumerate(docs):
        print(f'  {i}: {d}')
    if not docs:
        print('FAIL  no documents open — open one first')
        sys.exit(1)

    cur0 = nis_util.get_current_document(NIS)
    print(f'current document: {cur0!r}')
    in_list = any(os.path.normcase(d) == os.path.normcase(cur0) for d in docs)
    check('current doc is in the list', in_list)

    # activate the current doc itself with its exact NIS spelling
    doc0 = nis_util.activate_opened_document(NIS, cur0)
    check('activate(current) -> same entry',
          os.path.normcase(doc0) == os.path.normcase(cur0), f'({doc0!r})')

    # activate by base name only (the matching glue)
    base = os.path.basename(cur0)
    if base != cur0:  # only meaningful for a path, not a title
        doc1 = nis_util.activate_opened_document(NIS, base)
        check('activate(base name) -> same doc',
              os.path.normcase(doc1) == os.path.normcase(cur0), f'({doc1!r})')

    # switch to another open document, verify, switch back
    others = [d for d in docs if os.path.normcase(d) != os.path.normcase(cur0)]
    if others:
        target = others[0]
        nis_util.activate_document(NIS, target)
        cur1 = nis_util.get_current_document(NIS)
        check('current follows ActivateDocument',
              os.path.normcase(cur1) == os.path.normcase(target)
              or os.path.normcase(cur1) == os.path.normcase(os.path.basename(target)),
              f'(now: {cur1!r})')
        nis_util.activate_opened_document(NIS, cur0)
        cur2 = nis_util.get_current_document(NIS)
        check('switched back',
              os.path.normcase(cur2) == os.path.normcase(cur0),
              f'(now: {cur2!r})')
    else:
        print('NOTE  only one document open — skipping the switch test')

    print()
    if failures:
        print(f'{failures} failure(s)')
        sys.exit(1)
    print('all passed')


if __name__ == '__main__':
    main()
