# Hippocampal Segmentation
This code will segment the hippocampi on a 3D noncontrast T1 MRI. The segmentation will closely mirror that of FreeSurfer/FastSurfer, but exclude more enhancing structures such as choroid plexus. The models were trained specifically to exclude enhancing structures by procuring MRIs with both 3D T1 and 3D T1 postcontrast sequences, segmenting the T1 sequences with FastSurfer, using the postcontrast information to subtract enhancing structures from the FastSurfer segmentation, and retraining a neural network on the subtracted segmentations. This should work on CPU or GPU.

## Installation

1. Clone the repository:
```
git clone https://github.com/radiplab/hipposeg.git
cd hipposeg
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run hippocampal segmentation on the test MRI via either of these options:
* Run the tests/hipposeg_test.py/test_integration_hipposeg method as a unittest
* If you run into path issues with unittest, run hipposeg.py, which has a test() method set up to run automatically as well.

4. Segment your own studies:
```
import hipposeg
hipposeg.segment(input_nii_path, output_nii_path)
```


