
import glob
import os
import shutil
import unittest

import hipposeg

test_data_path = r'./tests/test_data'
base_working_path = r'./working'
test_working_path = os.path.join(base_working_path, 'hipposeg')

class CTHeadAlignTest(unittest.TestCase):
    if os.path.exists(test_working_path):
        shutil.rmtree(test_working_path)
    os.mkdir(test_working_path)

    def test_integration_hipposeg(self):
        """
        Test segmentation of the hippocampi - test data downloaded from the Human Connectome Project
        """
        input_nii_path = os.path.join(test_data_path, 'HCP_100206_3T_T1w_MPR1.nii.gz')
        output_nii_path = os.path.join(test_working_path, 'hippocampi-labels.nii.gz')
        hipposeg.segment(input_nii_path, output_nii_path)
