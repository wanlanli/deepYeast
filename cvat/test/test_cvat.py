# Copyright 2024 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import pathlib

from skimage.io import imread

from ..convert_format import Convert_Format, target_file_list, read_xml
# from ..utils import file_traverse


TEST_PATH = pathlib.Path(__file__).parent.resolve().joinpath("job_test")


class TEST_FORMATE(unittest.TestCase):
    def setUp(self) -> None:
        self.root_path = pathlib.Path(__file__).parent.resolve().joinpath("job_test")
        self.xml_path = self.root_path.joinpath("xml")
        self.mask_path = self.root_path.joinpath("mask")
        self.mask_image_list = target_file_list(str(self.mask_path), format="image")
        self.city_path = self.root_path.joinpath("city")

        self.xml2mask_path = self.root_path.joinpath("test_xml2mask")
        self.xml2training_path = self.root_path.joinpath("test_xml2training")
        self.mask2xml_path = self.root_path.joinpath("test_mask2xml", "test_mask2xml.xml")
        self.city2training_path = self.root_path.joinpath("test_city2mask")
        return super().setUp()

    def test_xml2mask(self):
        print("Test xml to mask")
        Convert_Format().xml2mask(str(self.xml_path),
                                  str(self.xml2mask_path))
        mask_list = target_file_list(str(self.xml2mask_path), format="image")
        if len(mask_list) > 0:
            saved = imread(mask_list[0])
            gt_image = imread(self.mask_image_list[0])
            self.assertLess((saved != gt_image).sum(),
                            gt_image.shape[0]*gt_image.shape[1]*0.1)
        else:
            self.assertFalse()

    def test_xml2training(self):
        # print("Test xml to training")
        # Convert_Format().xml2training(str(self.xml_path),
        #                               str(self.xml2training_path))
        pass

    def test_mask2xml(self):
        print("Test mask to xml")
        Convert_Format().mask2xml(str(self.mask_path),
                                  str(self.mask2xml_path))
        xml = read_xml(self.mask2xml_path)
        count = 0
        for image in xml:
            if image.tag == "image":
                # print(image.tag, image.attrib)
                count += 1
        self.assertEqual(count, len(self.mask_image_list))

    def test_city2training(self):
        print("Test city to training")
        Convert_Format().city2training(str(self.city_path),
                                       str(self.city2training_path))
        count = len(target_file_list(str(self.city2training_path), format="image"))
        self.assertEqual(count, 2*len(self.mask_image_list))


if __name__ == '__main__':
    unittest.main()
