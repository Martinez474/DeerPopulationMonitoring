"""Offline regression checks for the combined tree, AI, and thermal world."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'world/controllers/drone_tracker'))
from detection import DeerDetection, THERMAL_MODE
from thermal_image_detection import ThermalImageDeerDetector


class IntegrationTests(unittest.TestCase):
    def test_world_and_python(self):
        world = (ROOT / 'world/worlds/CatalinaIsland.wbt').read_text()
        for token in ['Forest {', 'DEF LaunchPad', 'DEF Waypoints',
                      'name "thermal depth"', 'name "thermal display"',
                      'DEF SURVEY_SUN', 'segmentation TRUE']:
            self.assertIn(token, world)
        self.assertEqual(world.count('Mavic2Pro {'), 1)
        self.assertEqual(world.count(' Deer {'), 10)
        self.assertEqual(world.count('{'), world.count('}'))
        for folder in ['drone_startup', 'drone_tracker', 'maveric_patrol']:
            for path in (ROOT / 'world/controllers' / folder).glob('*.py'):
                ast.parse(path.read_text())
        # An accidental leading 's' is syntactically valid, but fails on import.
        patrol = ROOT / 'world/controllers/maveric_patrol/maveric_patrol.py'
        with patch.dict(sys.modules, {'controller': SimpleNamespace(Robot=object)}):
            spec = importlib.util.spec_from_file_location('patrol_import', patrol)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_thermal_pixels_and_shared_result(self):
        detector = ThermalImageDeerDetector()
        width = height = 32
        temperatures = [10.] * (width * height)
        depths = [20.] * (width * height)
        self.assertEqual(detector.detect(temperatures, depths, width, height, 1., 640, 360, 10), [])
        for y in range(14, 17):
            for x in range(14, 17):
                temperatures[y * width + x] = 35.
        found = detector.detect(temperatures, depths, width, height, 1., 640, 360, 10)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].model, 'thermal_candidate')
        self.assertIsNotNone(found[0].thermal_contrast_c)
        rgb = DeerDetection(None, 'deer', 1, 2, 3, 4, None, None, None, .8)
        self.assertEqual(rgb.confidence, .8)
        self.assertIsNone(rgb.thermal_contrast_c)

    def test_toggle_preserves_yolo_and_thermal(self):
        spec = importlib.util.spec_from_file_location('tracker_merge_test',
            ROOT / 'world/controllers/drone_tracker/drone_tracker.py')
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'controller': SimpleNamespace(Keyboard=object, Supervisor=object),
                                      'tracker_merge_test': module}):
            spec.loader.exec_module(module)
        drone = module.WildlifeSurveyDrone.__new__(module.WildlifeSurveyDrone)
        thermal = SimpleNamespace(mode=THERMAL_MODE)
        yolo = SimpleNamespace(mode='yolo')
        drone.detectors = {THERMAL_MODE: thermal}
        drone.detector = thermal
        drone.next_action_time = 0
        drone.time_step = 8
        drone.view = Mock()
        drone._sync_thermal_screen = Mock()
        with patch.dict(sys.modules, {'yolo_detection': SimpleNamespace(YoloDeerDetector=Mock(return_value=yolo))}):
            drone._toggle_detector(1)
        self.assertIs(drone.detector, yolo)
        self.assertFalse(drone.view.thermal_mode)
        self.assertEqual(drone.detections, [])
        drone._toggle_detector(2)
        self.assertIs(drone.detector, thermal)
        self.assertTrue(drone.view.thermal_mode)


if __name__ == '__main__':
    unittest.main()
