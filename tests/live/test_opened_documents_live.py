"""
Live test for the opened-document wrappers.

Run at the microscope with NIS-Elements open and at least one image
document open. Exercises (read-only apart from switching the current
document, which is restored at the end):

  - get_opened_documents: list the open documents
  - get_current_document vs the list: is the current doc in the list,
    and in which form (full path vs title)
  - activate_document with the exact NIS spelling (list entry)
  - activate_opened_document with a full path and with a base name
  - if >= 2 documents are open: switch to another one, verify via
    get_current_document, switch back, verify

No documents are opened, saved, or closed.
Skipped automatically off the microscope workstation.
"""
import os
import unittest

from autofrap.microscope import nis as nis_util

NIS = r'C:\Program Files\NIS-Elements\nis_ar.exe'
LIVE = os.name == 'nt' and os.path.exists(NIS)


@unittest.skipUnless(LIVE, 'requires the microscope workstation (NIS-Elements)')
class TestOpenedDocumentsLive(unittest.TestCase):

    def test_opened_document_wrappers(self):
        docs = nis_util.get_opened_documents(NIS)
        if not docs:
            self.skipTest('no documents open — open one first')
        print(f'{len(docs)} open document(s):')
        for i, d in enumerate(docs):
            print(f'  {i}: {d}')

        cur0 = nis_util.get_current_document(NIS)
        print(f'current document: {cur0!r}')
        self.assertTrue(
            any(os.path.normcase(d) == os.path.normcase(cur0) for d in docs),
            'current doc is not in the list')

        # activate the current doc itself with its exact NIS spelling
        doc0 = nis_util.activate_opened_document(NIS, cur0)
        self.assertEqual(os.path.normcase(doc0), os.path.normcase(cur0),
                         f'activate(current) -> different entry ({doc0!r})')

        # activate by base name only (the matching glue)
        base = os.path.basename(cur0)
        if base != cur0:  # only meaningful for a path, not a title
            doc1 = nis_util.activate_opened_document(NIS, base)
            self.assertEqual(os.path.normcase(doc1), os.path.normcase(cur0),
                             f'activate(base name) -> different doc ({doc1!r})')

        # switch to another open document, verify, switch back
        others = [d for d in docs
                  if os.path.normcase(d) != os.path.normcase(cur0)]
        if others:
            target = others[0]
            nis_util.activate_document(NIS, target)
            cur1 = nis_util.get_current_document(NIS)
            self.assertTrue(
                os.path.normcase(cur1) == os.path.normcase(target)
                or os.path.normcase(cur1) == os.path.normcase(os.path.basename(target)),
                f'current does not follow ActivateDocument (now: {cur1!r})')
            nis_util.activate_opened_document(NIS, cur0)
            cur2 = nis_util.get_current_document(NIS)
            self.assertEqual(os.path.normcase(cur2), os.path.normcase(cur0),
                             f'not switched back (now: {cur2!r})')
        else:
            print('NOTE  only one document open — skipping the switch test')


if __name__ == '__main__':
    unittest.main()
