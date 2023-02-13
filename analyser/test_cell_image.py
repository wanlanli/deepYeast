import unittest

import numpy as np

from analyser.cell_image import CellImage


def encode_mask():
    size = np.random.randint(32, 2048)
    channel = 3
    image = np.zeros((size, size, channel), dtype=np.int16)
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
        for i in range(0, channel):
            image[donut, i] = np.random.randint(0, 255)
    return image, mask


class CellImageTestCase(unittest.TestCase):
    def setUp(self):
        image, mask = encode_mask()
        self.widget = CellImage(image=image, mask=mask)

    def test_instance_fluorescent_intensity(self):
        data = self.widget.instance_fluorescent_intensity()
        labels = np.unique(self.widget.mask.__array__())
        random_index = np.random.randint(1, len(labels))
        label = labels[random_index]
        center_x, center_y = self.widget.mask.get_region_center(label)
        self.assertEqual(data.loc[label, "ch1"], self.widget.image[int(center_x), int(center_y), 1],
                         'error get instance properties!')
        self.assertEqual(data.loc[label, "ch2"], self.widget.image[int(center_x), int(center_y), 2],
                         'error get instance properties!')
    def tearDown(self):
        del self.widget


if __name__ == '__main__':
    unittest.main()
# python -m unittest -v analyser/test_cell_image.py
