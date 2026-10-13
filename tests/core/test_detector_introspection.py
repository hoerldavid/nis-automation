"""
Unit tests for detector parameter introspection: the metadata that
build_detector attaches to its detection_fun (parameter_map +
step_functions), get_detector_parameters() for composed and plain
(hand-rolled) detection functions, and load_detector_module.

Synthetic functions only - no detector files, no detection calls.
"""
import functools
import os
import tempfile
import unittest

from autofrap.core.detection import (
    DetectorParameter,
    build_detector,
    get_detector_parameters,
    load_detector_file,
    load_detector_module,
)


def _load(f, load_channel=0):
    return None


def _detect(image, diameter=30.0, sigma=2):
    return None


def _detect_forwarded(image, diameter=30.0, **kwargs):
    """diameter visible, anything else behind **kwargs"""
    return None


def _filter(labels, image, threshold=450, metric="mean"):
    return {1}


def _stim(labels, image):
    return None


def _build(parameter_map, detect_fun=_detect, load_fun=_load):
    return build_detector(load_fun, detect_fun,
                          stim_mask_fun=_stim, filter_function=_filter,
                          visualization_fun=False,
                          parameter_map=parameter_map)


class TestMetadataAttachment(unittest.TestCase):
    """build_detector attaches the introspection metadata."""

    def test_parameter_map_and_step_functions_attached(self):
        """the callable carries the routing table and the sub-functions."""
        pmap = {"detector_fun": {"diameter": "diameter"}}
        det = build_detector(_load, _detect, parameter_map=pmap)
        self.assertIs(det.parameter_map, pmap)
        self.assertIs(det.step_functions["load_fun"], _load)
        self.assertIs(det.step_functions["detector_fun"], _detect)
        self.assertIs(det.step_functions["filter_function"], None)
        self.assertIs(det.step_functions["stim_mask_fun"], None)
        self.assertIs(det.step_functions["visualization_fun"], None)


class TestComposedParameters(unittest.TestCase):
    """get_detector_parameters on build_detector output."""

    def test_names_in_step_routing_order(self):
        """entries follow the load -> detect -> filter -> mask -> viz order."""
        pmap = {"filter_function": {"metric": "metric"},
                "detector_fun": {"diameter": "diameter"},
                "load_fun": {"load_channel": "load_channel"}}
        specs = get_detector_parameters(_build(pmap))
        self.assertEqual([s.name for s in specs],
                         ["load_channel", "diameter", "metric"])

    def test_defaults_and_types_resolved_via_signatures(self):
        """defaults come from the sub-function signature (internal name),
        ptype from the annotation or the default's type."""
        pmap = {"detector_fun": {"diameter": "diameter",
                                 "sigma": "sigma"},
                "filter_function": {"thresh": "threshold",
                                    "metric": "metric"}}
        specs = {s.name: s for s in get_detector_parameters(_build(pmap))}
        # 'thresh' resolves to the internal name 'threshold'
        self.assertEqual(specs["thresh"].default, 450)
        self.assertEqual(specs["thresh"].has_default, True)
        self.assertEqual(specs["thresh"].ptype, "int")
        self.assertEqual(specs["diameter"].default, 30.0)
        self.assertEqual(specs["diameter"].ptype, "float")
        self.assertEqual(specs["metric"].default, "mean")
        self.assertEqual(specs["metric"].ptype, "str")

    def test_key_behind_kwargs_not_introspectable(self):
        """a runtime key whose sub-function hides it behind **kwargs is
        listed, but without a default; a sibling key in the same
        signature is resolved."""
        pmap = {"detector_fun": {"diameter": "diameter",
                                 "min_size": "min_size"}}
        specs = {s.name: s for s in
                 get_detector_parameters(_build(pmap, _detect_forwarded))}
        self.assertEqual(specs["diameter"].has_default, True)
        self.assertEqual(specs["diameter"].default, 30.0)
        self.assertEqual(specs["min_size"].has_default, False)
        self.assertEqual(specs["min_size"].required, False)
        self.assertEqual(specs["min_size"].ptype, "")

    def test_internal_name_missing_from_step(self):
        """a mapping entry pointing at a parameter the step does not
        have is listed without default (not introspectable), not an error."""
        pmap = {"stim_mask_fun": {"halffrac": "halffrac"}}
        specs = get_detector_parameters(_build(pmap))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "halffrac")
        self.assertEqual(specs[0].has_default, False)

    def test_one_key_routed_to_two_steps(self):
        """a runtime key claimed by several steps becomes one entry
        with both steps recorded; the default resolves from whichever
        step's signature exposes it (here: the filter)."""
        pmap = {"detector_fun": {"threshold": "threshold"},
                "filter_function": {"threshold": "threshold"}}
        specs = get_detector_parameters(_build(pmap, _detect_forwarded))
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].steps,
                         ("detector_fun", "filter_function"))
        self.assertEqual(specs[0].default, 450)
        self.assertEqual(specs[0].has_default, True)

    def test_partial_step_function_signature_resolved(self):
        """functools.partial step functions introspect with the bound
        keywords folded into the defaults."""
        detect = functools.partial(_detect, sigma=5)
        pmap = {"detector_fun": {"diameter": "diameter"}}
        specs = get_detector_parameters(_build(pmap, detect_fun=detect))
        self.assertEqual(specs[0].default, 30.0)

    def test_composed_without_parameter_map_has_no_parameters(self):
        """parameter_map=None -> nothing tunable (the plain-signature
        fallback sees only survey_file + **runtime_kwargs)."""
        specs = get_detector_parameters(build_detector(_load, _detect))
        self.assertEqual(specs, [])


class TestPlainParameters(unittest.TestCase):
    """get_detector_parameters on hand-rolled detection functions."""

    def test_first_parameter_skipped(self):
        """the survey-file parameter is not tunable."""
        def detection_fun(survey_file, diameter=30):
            return None
        specs = get_detector_parameters(detection_fun)
        self.assertEqual([s.name for s in specs], ["diameter"])

    def test_defaults_required_and_types(self):
        """defaults, required flag (no default) and inferred types."""
        def detection_fun(survey_file, diameter=30.0, channel=0,
                           server="http://x", fast=False, threshold=None):
            return None
        specs = {s.name: s for s in get_detector_parameters(detection_fun)}
        self.assertEqual(specs["diameter"].default, 30.0)
        self.assertTrue(specs["diameter"].has_default)
        self.assertFalse(specs["diameter"].required)
        self.assertEqual(specs["diameter"].ptype, "float")
        self.assertEqual(specs["channel"].ptype, "int")
        self.assertEqual(specs["fast"].ptype, "bool")
        self.assertEqual(specs["server"].ptype, "str")
        # explicit None default: known default, unknown type
        self.assertEqual(specs["threshold"].default, None)
        self.assertTrue(specs["threshold"].has_default)
        self.assertEqual(specs["threshold"].ptype, "")

    def test_parameter_without_default_is_required(self):
        def detection_fun(survey_file, must_have):
            return None
        specs = get_detector_parameters(detection_fun)
        self.assertEqual(specs[0].required, True)
        self.assertEqual(specs[0].has_default, False)

    def test_var_args_ignored(self):
        """*args / **kwargs entries are skipped, named params kept."""
        def detection_fun(survey_file, diameter=30, *args, **kwargs):
            return None
        specs = get_detector_parameters(detection_fun)
        self.assertEqual([s.name for s in specs], ["diameter"])

    def test_annotation_preferred_over_default_type(self):
        def detection_fun(survey_file, diameter: float = 30):
            return None
        specs = get_detector_parameters(detection_fun)
        self.assertEqual(specs[0].ptype, "float")


class TestDetectorParameterDataclass(unittest.TestCase):
    """the DetectorParameter defaults: nothing claimed unless set."""

    def test_defaults(self):
        p = DetectorParameter(name="x")
        self.assertEqual(p.steps, ())
        self.assertEqual(p.default, None)
        self.assertEqual(p.has_default, False)
        self.assertEqual(p.required, False)
        self.assertEqual(p.ptype, "")


class TestLoadDetectorModule(unittest.TestCase):
    """load_detector_module: module import + docstring access; the
    detection_fun callable check stays in load_detector_file."""

    def _write(self, tmpdir, body):
        path = os.path.join(tmpdir, "my_det.py")
        with open(path, "w") as fh:
            fh.write(body)
        return path

    def test_module_docstring_and_detection_fun(self):
        body = ('"my detector docstring"\n\n'
                'def detection_fun(f):\n    return None\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, body)
            mod = load_detector_module(path)
            self.assertEqual(mod.__doc__, "my detector docstring")
            # a second import (as load_detector_file does) yields a new
            # module object - compare the contract, not identity
            fun = load_detector_file(path)
            self.assertEqual(fun.__name__, "detection_fun")
            self.assertTrue(callable(fun))

    def test_non_callable_still_raises_in_load_detector_file(self):
        body = '"doc"\n\ndetection_fun = 42\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, body)
            mod = load_detector_module(path)
            self.assertEqual(mod.__doc__, "doc")
            with self.assertRaises(ValueError):
                load_detector_file(path)


if __name__ == "__main__":
    unittest.main()
