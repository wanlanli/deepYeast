import unittest
import numpy as np


class ImageMeasure(unittest.TestCase):
    def setUp(self):
        
        data = np.zeros((100, 100), dtype=np.int16)
        number_obj = np.random.randint(2, 10)
        for i in range(0, number_obj):
            radius = np.random.randint(10, 20)
            cx = np.random.randint(0, 100)
            cy = np.random.randint(0, 100)
            type_id = np.random.randint(1, 4)
            Y, X = np.ogrid[:100, :100]
            dist_from_center = np.sqrt((X - cx)**2 + (Y-cy)**2)
            mask = dist_from_center <= radius
            data = data + (mask*(1000*type_id+1))
        self.data = ImageMeasure(data)
        self.num = number_obj

if __name__ == "__main__":
    unittest.main()
