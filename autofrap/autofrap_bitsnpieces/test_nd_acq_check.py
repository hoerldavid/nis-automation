"""
Tests for the ND Acquisition template pre-flight check (TODO #14).

run: pytest autofrap/autofrap_bitsnpieces/test_nd_acq_check.py -v
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import unittest
from autofrap.pipeline.autofrap import _check_nd_acq_template, NonRecoverableError

# Shared tab templates
_TABS_BASE = {
    'Time': False, 'XY': False, 'Z': False,
    'Lambda': False, 'Large Image': False}


class TestCheckNDAcqTemplate(unittest.TestCase):
    """Pure logic tests for _check_nd_acq_template."""

    def test_lambda_active_ok(self):
        """Lambda (channel) active alone -> OK."""
        tabs = dict(_TABS_BASE, Lambda=True)
        result = _check_nd_acq_template(tabs)
        self.assertEqual(result, tabs)

    def test_no_tabs_active_ok(self):
        """Nothing active -> OK (single image with current laser/filter)."""
        result = _check_nd_acq_template(dict(_TABS_BASE))
        self.assertEqual(result, _TABS_BASE)

    def test_z_active_ok(self):
        """Z tab active -> OK (future support for Z-stacks)."""
        tabs = dict(_TABS_BASE, Z=True, Lambda=True)
        result = _check_nd_acq_template(tabs)
        self.assertEqual(result, tabs)

    def test_z_and_time_raises(self):
        """Time + Z active -> NonRecoverableError (Time is forbidden)."""
        tabs = dict(_TABS_BASE, Time=True, Z=True)
        with self.assertRaises(NonRecoverableError) as ctx:
            _check_nd_acq_template(tabs)
        self.assertIn('Time', str(ctx.exception))

    def test_time_active_raises(self):
        """Time loop active -> NonRecoverableError."""
        tabs = dict(_TABS_BASE, Time=True)
        with self.assertRaises(NonRecoverableError) as ctx:
            _check_nd_acq_template(tabs)
        self.assertIn('Time', str(ctx.exception))

    def test_xy_active_raises(self):
        """XY multipoint active -> NonRecoverableError."""
        tabs = dict(_TABS_BASE, XY=True)
        with self.assertRaises(NonRecoverableError) as ctx:
            _check_nd_acq_template(tabs)
        self.assertIn('XY', str(ctx.exception))

    def test_large_image_active_raises(self):
        """Large Image scan active -> NonRecoverableError."""
        tabs = dict(_TABS_BASE, **{'Large Image': True})
        with self.assertRaises(NonRecoverableError) as ctx:
            _check_nd_acq_template(tabs)
        self.assertIn('Large Image', str(ctx.exception))

    def test_multiple_forbidden_raises(self):
        """Time + XY active -> NonRecoverableError with both names."""
        tabs = dict(_TABS_BASE, Time=True, XY=True)
        with self.assertRaises(NonRecoverableError) as ctx:
            _check_nd_acq_template(tabs)
        exc_str = str(ctx.exception)
        self.assertIn('Time', exc_str)
        self.assertIn('XY', exc_str)

    def test_all_forbidden_raises(self):
        """All forbidden tabs active -> error lists all three."""
        tabs = dict(_TABS_BASE, Time=True, XY=True, **{'Large Image': True})
        with self.assertRaises(NonRecoverableError) as ctx:
            _check_nd_acq_template(tabs)
        exc_str = str(ctx.exception)
        self.assertIn('Large Image', exc_str)
        self.assertIn('Time', exc_str)
        self.assertIn('XY', exc_str)

    def test_return_value_unchanged(self):
        """Valid config returns the input dict unchanged."""
        tabs = dict(_TABS_BASE, Lambda=True, Z=True)
        result = _check_nd_acq_template(tabs)
        self.assertIs(result, tabs)  # same object


class TestRunNDAcqCheck(unittest.TestCase):
    """Tests for the nis_util wrapper _run_nd_acq_check."""

    def test_calls_get_nd_acq_tabs(self):
        """_run_nd_acq_check delegates to get_nd_acq_tabs + _check_nd_acq_template."""
        from unittest import mock
        import importlib
        import nis_util
        from autofrap.pipeline import autofrap as autofrap_mod
        importlib.reload(autofrap_mod)

        fake = mock.MagicMock()
        fake.get_nd_acq_tabs.return_value = dict(_TABS_BASE, Lambda=True)
        with mock.patch.dict('sys.modules', {'nis_util': fake}):
            importlib.reload(autofrap_mod)
            result = autofrap_mod._run_nd_acq_check('fake_nis')
            fake.get_nd_acq_tabs.assert_called_once_with('fake_nis')
            self.assertEqual(result, dict(_TABS_BASE, Lambda=True))


if __name__ == '__main__':
    unittest.main()
