import unittest

import numpy as np

from analyser.mask_feature import MaskFeature


def encode_mask():
    size = np.random.randint(32, 2048)
    mask = np.zeros((size, size), dtype=np.int16)
    number = np.random.randint(1, 100)
    for i in range(0, number):
        radius = np.random.randint(2, int(size/number))
        center_x, center_y = np.random.randint(0, size, 2)
        value = np.random.randint(0, 5)*1000+i
        xx, yy = np.mgrid[:size, :size]
        circle = (xx - center_x) ** 2 + (yy - center_y) ** 2
        donut = circle < radius**2
        mask[donut] = value
    return mask, number


class MaskFeatureTestCase(unittest.TestCase):
    def setUp(self):
        mask, self.number = encode_mask()
        self.widget = MaskFeature(mask)

    def test_get_instance_properties(self):
        data = self.widget.instance_properties
        self.assertEqual(data.shape[0], self.number,
                         'error get instance properties!')

    def tearDown(self):
        del self.widget


if __name__ == '__main__':
    unittest.main()
# python -m unittest -v analyser/test_mask_feature.py
