"""
Tests for autofrap.core.utils.grid.spiral_positions (grid geometry,
no microscope needed).
"""
import unittest

import numpy as np

from autofrap.core.utils.grid import spiral_positions


class TestSpiralPositions(unittest.TestCase):

    def test_layer_sizes(self):
        """Each layer n (n >= 1) adds 8n positions; layer 0 adds 1."""
        cumulative = 0
        for layer in range(10):
            cumulative += (1 if layer == 0 else 8 * layer)
            with self.subTest(layer=layer):
                pos = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=1.0,
                                       max_positions=cumulative)
                self.assertEqual(len(pos), cumulative)

    def test_layer0_single(self):
        """Layer 0: just the center (1 position)."""
        pos = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=1.0,
                               max_positions=1)
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos[0], (0.0, 0.0))

    def test_layer1_total(self):
        """Layer 0 + Layer 1 = 1 + 8 = 9 positions."""
        pos = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=1.0,
                               max_positions=9)
        self.assertEqual(len(pos), 9)
        self.assertEqual(pos[0], (0.0, 0.0))

    def test_layer2_total(self):
        """Layer 0 + 1 + 2 = 1 + 8 + 16 = 25 positions."""
        pos = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=1.0,
                               max_positions=25)
        self.assertEqual(len(pos), 25)

    def test_center_is_first(self):
        pos = spiral_positions((100.0, 200.0), fov=(10.0, 10.0), spacing=1.0,
                               max_positions=2)
        self.assertTrue(np.allclose(pos[0], (100.0, 200.0)))

    def test_spacing_scaling(self):
        p1 = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=1.0,
                              max_positions=2)
        p2 = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=2.0,
                              max_positions=2)
        self.assertTrue(np.allclose(np.array(p2[1]) - np.array(p2[0]),
                                    2 * (np.array(p1[1]) - np.array(p1[0]))))

    def test_fov_scaling(self):
        p = spiral_positions((0, 0), fov=(20.0, 20.0), spacing=1.0,
                             max_positions=2)
        self.assertTrue(np.allclose(np.array(p[1]) - np.array(p[0]),
                                    (20.0, 0.0)))

    def test_order_is_spiral(self):
        """First 25 positions should form a 5x5 square with center first."""
        pos = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=1.0,
                               max_positions=25)
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        unique_x = sorted(set(xs))
        unique_y = sorted(set(ys))
        self.assertEqual(len(unique_x), 5)
        self.assertEqual(len(unique_y), 5)
        self.assertIn((0.0, 0.0), pos[:1])
        for x in unique_x:
            for y in unique_y:
                self.assertIn((x, y), pos)

    def test_partial_layer(self):
        """max_positions=13 = layer0(1) + layer1(8) + 4 from layer2."""
        pos = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=1.0,
                               max_positions=13)
        self.assertEqual(len(pos), 13)
        # First 9 are center + full layer 1 (all within [-1,1]x[-1,1])
        for i in range(9):
            self.assertLessEqual(abs(pos[i][0]), 1.0)
            self.assertLessEqual(abs(pos[i][1]), 1.0)
        # Next 4 are from layer 2 right edge (x=2)
        for i in range(9, 13):
            self.assertEqual(pos[i][0], 2.0)

    def test_changing_fov(self):
        p1 = spiral_positions((0, 0), fov=(10.0, 10.0), spacing=1.0,
                              max_positions=5)
        p2 = spiral_positions((0, 0), fov=(20.0, 20.0), spacing=1.0,
                              max_positions=5)
        for a, b in zip(p1, p2):
            self.assertTrue(np.allclose(np.array(b), 2 * np.array(a)))

    def test_nonzero_center(self):
        center = (500.0, 300.0)
        pos = spiral_positions(center, fov=(10.0, 10.0), spacing=1.0,
                               max_positions=3)
        self.assertEqual(pos[0], (500.0, 300.0))
        self.assertEqual(pos[1][0], 510.0)
        self.assertEqual(pos[1][1], 300.0)

    def test_no_duplicate_positions(self):
        pos = spiral_positions((0, 0), fov=(1.0, 1.0), spacing=1.0,
                               max_positions=100)
        self.assertEqual(len(set(pos)), 100)


if __name__ == '__main__':
    unittest.main()
