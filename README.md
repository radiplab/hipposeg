# Hippocampal Segmentation
This code will segment the hippocampi on a 3D noncontrast T1 MRI. The models were trained specifically to exclude adjacent enhancing structures such as the choroid plexus and basal vein of Rosenthal. This was done by procuring MRIs with both 3D T1 and 3D T1 postcontrast sequences, segmenting the T1 sequences with FastSurfer, using the postcontrast information to subtract enhancing structures from the FastSurfer segmentation, and retraining a neural network on the subtracted segmentations.

View the tests/hipposeg_test.py file for an example of how to run this segmentation, but you essentially run:
```
hipposeg.segment(input_nii_path, output_nii_path)
```
