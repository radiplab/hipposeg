import configparser
import os
import shutil
import time
from itertools import product

import ants
import monai
import numpy as np
import scipy as sp
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from skimage.measure import label


def segment(input_nii_path, output_nii_path):
    """ Segments the hippocampi on a 3D T1-weighted MRI

    Parameters:
        input_nii_path (str): Full path to the .nii file to segment
        output_nii_path (str): Full path to the .nii file to write to - any existing file will be overwritten. Label 1 = right hippocampus, label 2 = left hippocampus
    """

    hippocampus_config = configparser.ConfigParser()
    hippocampus_config.read('./paths.ini')
    hippocampus_paths_config = hippocampus_config['paths']
    rh_model_path = hippocampus_paths_config['rh_model_path']
    lh_model_path = hippocampus_paths_config['lh_model_path']
    rh_cropped_model_path = hippocampus_paths_config['rh_cropped_model_path']
    lh_cropped_model_path = hippocampus_paths_config['lh_cropped_model_path']
    t1_mni_path = hippocampus_paths_config['t1_mni_path']
    working_path = r'./working/hipposeg'
    if os.path.exists(working_path):
        shutil.rmtree(working_path)
    os.mkdir(working_path)

    # Convert to axial, bias correct, register, convert to uint16
    input_sitk = nii2sitk(input_nii_path)
    input_sitk = to_axial(input_sitk)

    # Resample MNI template to 1 mm isovoxel (it's 0.5, and already in axial plane)
    mni_tmp_path = os.path.join(working_path, 't1_mni_1mmiso.nii.gz')
    if not os.path.exists(mni_tmp_path):
        t1_mni_image = nii2sitk(t1_mni_path)
        t1_mni_resampled_image = resample_spacing(t1_mni_image, new_spacing=[1,1,1])
        sitk.WriteImage(t1_mni_resampled_image, mni_tmp_path)

    # Bias Correct
    input_axial_path = os.path.join(working_path, 'input_a.nii.gz')
    sitk.WriteImage(input_sitk, input_axial_path)
    input_axial_bc_path = os.path.join(working_path, 'input_a_bc.nii.gz')
    image = ants.image_read(input_axial_path)
    image_n4 = ants.n4_bias_field_correction(image).astype('uint32')
    ants.image_write(image_n4, input_axial_bc_path, ri=False)

    # Register to MNI
    input_axial_bc_registered_path = os.path.join(working_path, 'input_a_bc_r.nii.gz')
    fi = ants.image_read(mni_tmp_path, pixeltype='float', reorient=False)
    mi = ants.image_read(input_axial_bc_path, pixeltype='float', reorient=False)
    mi = ants.resample_image(mi, (fi.shape[0], fi.shape[1], mi.shape[2]), 1, 0)
    # “Similarity”: Similarity transformation: scaling, rotation and translation.
    mytx = ants.registration(fixed=fi, moving=mi, type_of_transform = 'Similarity' )
    transform_tmp_path = os.path.join(working_path, 'ants_transform.mat')
    shutil.copy(mytx['fwdtransforms'][0], transform_tmp_path)
    transform = sitk.ReadTransform(transform_tmp_path) 
    fixed_image_sitk = nii2sitk(mni_tmp_path)
    moving_image_sitk = nii2sitk(input_axial_bc_path)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image_sitk)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkBSpline)
    registered_orig_t1_sitk = resampler.Execute(moving_image_sitk)
    sitk.WriteImage(registered_orig_t1_sitk, input_axial_bc_registered_path)
    os.remove(transform_tmp_path)

    # Convert to uint16
    input_sitk = nii2sitk(input_axial_bc_registered_path)
    input_sitk = sitk.Cast(input_sitk, sitk.sitkInt16)

    # Predict
    r_prediction = predict_high_res(input_sitk, rh_model_path, rh_cropped_model_path)
    l_prediction = predict_high_res(input_sitk, lh_model_path, lh_cropped_model_path)

    # Merge the predctions
    r_array = sitk.GetArrayFromImage(r_prediction)
    l_array = sitk.GetArrayFromImage(l_prediction)
    hippocampi_array = np.zeros_like(r_array)
    hippocampi_array[r_array == 1] = 1
    hippocampi_array[l_array == 1] = 2
    hippocampi_predictions = sitk.GetImageFromArray(hippocampi_array)
    hippocampi_predictions.CopyInformation(r_prediction)

    sitk.WriteImage(hippocampi_predictions, output_nii_path) 

    #shutil.rmtree(working_path)

def add_columns(input_image, num_columns, add_even=True, add_right=False, add_left=False, fill_data=None):
    """ Add columns to an sitk image

    Parameters:
        input_image (sitk.Image): SITK Image to add columns to (will not be modified)
        num_columns (int): Number of columns to add
        add_even (bool): Distribute the columns evenly between right and left
        add_right (bool): Add columns only to the right
        add_left (bool): Add columns only to the left
        fill_data (float): Voxel value for the extra data. Defaults to the min voxel value. Can be np.nan
    Returns:
        output_image (sitk.Image): SITK Image with columns added
    """    
    input_image_data = sitk.GetArrayFromImage(input_image)
    input_image_data_added = None
    if fill_data is None:
        fill_data = np.amin(input_image_data)

    # sitk data is z,y,x
    right_num_columns = None
    left_num_columns = None
    right_columns = None
    left_columns = None
    if add_even:
        right_num_columns = int(np.floor(num_columns/2))
        left_num_columns = int(np.ceil(num_columns/2))
        right_columns = np.full((input_image_data.shape[0], input_image_data.shape[1], right_num_columns), fill_data)
        left_columns = np.full((input_image_data.shape[0], input_image_data.shape[1], left_num_columns), fill_data)
        input_image_data_added = np.concatenate((right_columns, input_image_data, left_columns), axis=2)
    elif add_right:
        right_num_columns = num_columns
        right_columns = np.full((input_image_data.shape[0], input_image_data.shape[1], right_num_columns), fill_data)
        input_image_data_added = np.concatenate((right_columns, input_image_data), axis=2)
    elif add_left:
        left_num_columns = num_columns
        left_columns = np.full((input_image_data.shape[0], input_image_data.shape[1], left_num_columns), fill_data)
        input_image_data_added = np.concatenate((input_image_data, left_columns), axis=2)

    output_image = sitk.GetImageFromArray(input_image_data_added)
    new_origin = None
    if right_num_columns is not None: 
        new_origin = input_image.TransformIndexToPhysicalPoint((int(-right_num_columns),0,0))
    else:
        new_origin = input_image.GetOrigin()
    
    output_image.SetOrigin(new_origin)
    output_image.SetSpacing(input_image.GetSpacing())
    output_image.SetDirection(input_image.GetDirection())

    return output_image 


def add_padding_for_transform(input_image, transform, xy=True, z=True):
    """ Pad an image so as not to lose data with the provided transform

    Parameters:
        input_image (sitk.Image): SITK Image to pad (will not be modified)
        transform (sitk.Transform) STIK transform that will be applied
        xy (bool): Add padding to rows and columns
        z (bool): Add padding to slices
    Returns:
        output_image (sitk.Image): Padded SITK Image
    """    
    input_image_data = sitk.GetArrayFromImage(input_image)
    output_image = sitk.GetImageFromArray(input_image_data)
    output_image.CopyInformation(input_image)     
    
    #Compute the bounding box of the rotated volume
    start_index = [0,0,0] # x,y,z
    end_index = [sz-1 for sz in input_image.GetSize()] # x,y,z
    orig_indices = list(product(*zip(start_index, end_index)))
    orig_min_indices = np.min(orig_indices,0) # x,y,z
    orig_max_indices = np.max(orig_indices,0) # x,y,z    

    physical_corners = [input_image.TransformIndexToPhysicalPoint(corner) for corner in list(product(*zip(start_index, end_index)))]
    transformed_corners = [transform.TransformPoint(corner) for corner in physical_corners]
    transformed_indices = [input_image.TransformPhysicalPointToIndex(corner) for corner in transformed_corners]
    transformed_min_indices = np.min(transformed_indices,0) # x,y,z
    transformed_max_indices = np.max(transformed_indices,0) # x,y,z

    if xy:
        # Right
        right_diff = orig_min_indices[0] - transformed_min_indices[0]
        if right_diff > 0:
            output_image = add_columns(output_image, num_columns=right_diff, add_even=False, add_right=True, add_left=False)
        # Left
        left_diff = transformed_max_indices[0] - orig_max_indices[0]
        if left_diff > 0:
            output_image = add_columns(output_image, num_columns=left_diff, add_even=False, add_right=False, add_left=True)
        
        # Anterior
        anterior_diff = orig_min_indices[1] - transformed_min_indices[1]
        if anterior_diff > 0:
            output_image = add_rows(output_image, num_rows=anterior_diff, add_even=False, add_anterior=True, add_posterior=False)
        # Posterior
        posterior_diff = transformed_max_indices[1] - orig_max_indices[1]
        if posterior_diff > 0:
            output_image = add_rows(output_image, num_rows=posterior_diff, add_even=False, add_anterior=True, add_posterior=False)

    if z:
        # Inferior
        inferior_diff = orig_min_indices[2] - transformed_min_indices[2]
        if inferior_diff > 0:
            output_image = add_slices(output_image, num_slices=inferior_diff, add_even=False, add_inferior=True, add_superior=False)
        # Superior
        superior_diff = transformed_max_indices[2] - orig_max_indices[2]
        if superior_diff > 0:
            output_image = add_slices(output_image, num_slices=superior_diff, add_even=False, add_inferior=False, add_superior=True)

    return output_image


def add_rows(input_image, num_rows, add_even=True, add_anterior=False, add_posterior=False, fill_data=None):
    """ Add rows to an sitk image

    Parameters:
        input_image (sitk.Image): SITK Image to add rows to (will not be modified)
        num_rows (int): Number of rows to add
        add_even (bool): Distribute the rows evenly between anterior and posterior
        add_anterior (bool): Add rows only anterior
        add_posterior (bool): Add columns only posterior
        fill_data (float): Voxel value for the extra data. Defaults to the min voxel value. Can be np.nan
    Returns:
        output_image (sitk.Image): SITK Image with columns added
    """     
    input_image_data = sitk.GetArrayFromImage(input_image)
    input_image_data_added = None
    if fill_data is None:
        fill_data = np.amin(input_image_data)

    # sitk data is z,y,x
    anterior_num_rows = None
    posterior_num_rows = None
    anterior_rows = None
    posterior_rows = None
    if add_even:
        anterior_num_rows = int(np.floor(num_rows/2))
        posterior_num_rows = int(np.ceil(num_rows/2))
        anterior_rows = np.full((input_image_data.shape[0], anterior_num_rows, input_image_data.shape[2]), fill_data)
        posterior_rows = np.full((input_image_data.shape[0], posterior_num_rows, input_image_data.shape[2]), fill_data)
        input_image_data_added = np.concatenate((anterior_rows, input_image_data, posterior_rows), axis=1)
    elif add_anterior:
        anterior_num_rows = num_rows
        anterior_rows = np.full((input_image_data.shape[0], anterior_num_rows, input_image_data.shape[2]), fill_data)
        input_image_data_added = np.concatenate((anterior_rows, input_image_data), axis=1)
    elif add_posterior:
        posterior_num_rows = num_rows
        posterior_rows = np.full((input_image_data.shape[0], posterior_num_rows, input_image_data.shape[2]), fill_data)
        input_image_data_added = np.concatenate((input_image_data, posterior_rows), axis=1)

    output_image = sitk.GetImageFromArray(input_image_data_added)
    new_origin = None
    if anterior_num_rows is not None: 
        new_origin = input_image.TransformIndexToPhysicalPoint((0,int(-anterior_num_rows),0))
    else:
        new_origin = input_image.GetOrigin()
    
    output_image.SetOrigin(new_origin)
    output_image.SetSpacing(input_image.GetSpacing())
    output_image.SetDirection(input_image.GetDirection())

    return output_image

def add_slices(input_image, num_slices, add_even=True, add_inferior=False, add_superior=False, fill_data=None):
    """ Add slices to an sitk image

    Parameters:
        input_image (sitk.Image): SITK Image to add slices to (will not be modified)
        num_slices (int): Number of slices to add
        add_even (bool): Distribute the slices evenly between inferior and superior
        add_inferior (bool): Add rows only to the inferior
        add_superior (bool): Add columns only to the superior
        fill_data (float): Voxel value for the extra data. Defaults to the min voxel value. Can be np.nan
    Returns:
        output_image (sitk.Image): SITK Image with columns added
    """    
    input_image_data = sitk.GetArrayFromImage(input_image)
    input_image_data_added = None
    if fill_data is None:
        fill_data = np.amin(input_image_data)

    # sitk data is z,y,x
    inferior_num_slices = None
    superior_num_slices = None
    inferior_slices = None
    superior_slices = None
    if add_even:
        inferior_num_slices = int(np.floor(num_slices/2))
        superior_num_slices = int(np.ceil(num_slices/2))
        inferior_slices = np.full((inferior_num_slices, input_image_data.shape[1], input_image_data.shape[2]), fill_data)
        superior_slices = np.full((superior_num_slices, input_image_data.shape[1], input_image_data.shape[2]), fill_data)
        input_image_data_added = np.concatenate((inferior_slices, input_image_data, superior_slices), axis=0)
    elif add_inferior:
        inferior_num_slices = num_slices
        inferior_slices = np.full((inferior_num_slices, input_image_data.shape[1], input_image_data.shape[2]), fill_data)
        input_image_data_added = np.concatenate((inferior_slices, input_image_data), axis=0)
    elif add_superior:
        superior_num_slices = num_slices
        superior_slices = np.full((superior_num_slices, input_image_data.shape[1], input_image_data.shape[2]), fill_data)
        input_image_data_added = np.concatenate((input_image_data, superior_slices), axis=0)

    output_image = sitk.GetImageFromArray(input_image_data_added)
    new_origin = None
    if inferior_num_slices is not None: 
        new_origin = input_image.TransformIndexToPhysicalPoint((0,0,int(-inferior_num_slices)))
    else:
        new_origin = input_image.GetOrigin()
    
    output_image.SetOrigin(new_origin)
    output_image.SetSpacing(input_image.GetSpacing())
    output_image.SetDirection(input_image.GetDirection())

    return output_image

def composite(input_image, transforms, maintain_xy_data=True, maintain_z_data=True, maintain_physical_location=True, interpolation=sitk.sitkBSpline, padding_empty_voxel_value=None, segmentation=False, verbose=False):
    """ Applies a list of transforms as a composite transform
    Parameters:
        input_image (sitk.Image): SimpleITK image, will not be modified
        transforms (list): List of transforms to apply
        maintain_xy_data (bool): If true, rows and columns will not be truncated
        maintain_z_data (bool): If true, slices will not be truncated
        maintain_physical_location (bool): If true, the physical location of the rotated image is maintained
        interpolation (optional): SimpleITK interpolation method - default is sitk.sitkLinear
        verbose (bool): Verbose flag
    Returns:
        transformed SimpleITK image    
    """
    start_time = None
    if verbose:
        start_time = time.time()
        print("Applying transforms as composite..", end = '')

    composite_transform = sitk.CompositeTransform(transforms)

    input_image_data = sitk.GetArrayFromImage(input_image)
    #padding_empty_voxel_value = np.amin(input_image_data)
    padding_empty_voxel_value = determine_empty_voxel_value(input_image)
    working_image = sitk.GetImageFromArray(input_image_data)
    working_image.CopyInformation(input_image) 

    if maintain_xy_data or maintain_z_data:
        working_image = add_padding_for_transform(working_image, transform=composite_transform, xy=maintain_xy_data, z=maintain_z_data)

    working_image_data = sitk.GetArrayFromImage(working_image)
    min_value = np.amin(working_image_data)

    # Composite transforms were applied stack-based - first in, last applied
    # With sitk 2.0 there is now a CompositeTransform class. Not sure what order they're applied.
    working_image = sitk.Resample(working_image, composite_transform, interpolation, float(min_value)) # Float required for Resample

    if maintain_physical_location:
        # Adjust the image direction
        new_direction = transform_direction(working_image, composite_transform)
        working_image.SetDirection(new_direction)
        
        # Adjust the image origin
        new_origin = transform_origin(working_image, composite_transform)
        working_image.SetOrigin(new_origin)

    if maintain_xy_data or maintain_z_data:
        working_image = remove_padding(working_image, padding_empty_voxel_value, xy=maintain_xy_data, z=maintain_z_data, segmentation=segmentation)

    if verbose:
        print('.done - time = ' + str(np.around((time.time() - start_time), decimals=1)) + 'sec') 

    return working_image 

def crop_around_prediction(input_sitk, input_label_sitk, crop_size=None, verbose=False):
    """ Crops a simple ITK image and label around the label so that prediction can be done at full resolution. Crops at 1/4 the size in x and y

    Parameters:
        input_sitk (sitk.Image): Loaded sitk image
        input_label_sitk (sitk.Image): sitk label
        crop_size (list): Optional provided x,y,z crop size (e.g. (128,128,64)). Otherwise it crops to 1/4 the original size.
        verbose (bool): Verbosity flag

    Returns:
        cropped_image (sitkImage): Cropped image
        cropped_label (sitkImage): Cropped label
        crop_indices (array): Indices used to crop the image: [z_start:z_end, y_start:y_end, x_start:x_end]
    """
    if verbose:
        print('Cropping for iterative prediction..', end='')

    # Put the 2 images in the same orientation - I'm commenting this out for now because I'm concerned
    # it shouldn't be necessary, and may mask problems in the data
    #resample = sitk.ResampleImageFilter() # Trim the subtracted image to match the segmentation dimensions
    #resample.SetReferenceImage(input_sitk)
    #resample.SetInterpolator(sitk.sitkNearestNeighbor)
    #input_label_sitk_r = resample.Execute(input_label_sitk)

    # Calculate the center of the identified structure
    # input_label_sitk_data = z,y,x
    input_label_sitk_data = sitk.GetArrayFromImage(input_label_sitk)
    predict_center = sp.ndimage.measurements.center_of_mass(input_label_sitk_data)
    predict_center = [np.around(predict_center[0]), np.around(predict_center[1]), np.around(predict_center[2])]

    
    # Crop the input_sitk data around the center at 1/4 the original size
    orig_spacing = input_sitk.GetSpacing()
    #new_spacing = tuple(ti/4 for ti in orig_spacing)

    orig_size = input_sitk.GetSize() # x,y,z
    orig_x_size = orig_size[0]
    orig_y_size = orig_size[1]
    orig_z_size = orig_size[2]

    new_x_size = None
    new_y_size = None
    new_z_size = None

    if crop_size is None:
        new_x_size = int(orig_x_size / 4)
        new_y_size = int(orig_y_size / 4)
        new_x_size = min(new_x_size, new_y_size)
        new_y_size = min(new_x_size, new_y_size)
        new_z_size = int(orig_z_size / 4)
    else: # Override the above crop dimensions
        new_x_size = crop_size[0]
        new_y_size = crop_size[1]
        if len(crop_size) == 2:
            new_z_size = int(orig_z_size / 4)
        if len(crop_size) == 3:
            new_z_size = crop_size[2]

    # cropped_data = z,y,x
    cropped_data = sitk.GetArrayFromImage(input_sitk)
    z_start = int(predict_center[0] - int(new_z_size/2))
    if z_start < 0:
        z_start = 0
    z_end = z_start + new_z_size
    if z_end >= orig_z_size:
        z_end = int(orig_z_size - 1)
        z_start = z_end - new_z_size
        # Number of slices doesn't matter
        if z_start < 0:
            z_start = 0
    
    y_start = int(predict_center[1] - int(new_y_size/2))
    if y_start < 0:
        y_start = 0
    y_end = y_start + new_y_size
    if y_end >= orig_y_size:
        y_end = int(orig_y_size - 1)
        y_start = y_end - new_y_size
        if y_start < 0:
            raise Exception('Cropping in Y dimension failed')

    x_start = int(predict_center[2] - int(new_x_size/2))
    if x_start < 0:
        x_start = 0
    x_end = x_start + new_x_size
    if x_end >= orig_x_size:
        x_end = int(orig_x_size - 1)
        x_start = x_end - new_x_size
        if x_start < 0:
            raise Exception('Cropping in X dimension failed')

    crop_indices = [z_start, z_end, y_start, y_end, x_start, x_end]
    cropped_data = cropped_data[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3], crop_indices[4]:crop_indices[5]]
    cropped_sitk = sitk.GetImageFromArray(cropped_data)
    cropped_sitk.SetSpacing(input_sitk.GetSpacing())
    cropped_sitk.SetDirection(input_sitk.GetDirection())

    cropped_input_label_sitk_data = sitk.GetArrayFromImage(input_label_sitk)
    cropped_input_label_sitk_data = cropped_input_label_sitk_data[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3], crop_indices[4]:crop_indices[5]]
    cropped_input_label_sitk = sitk.GetImageFromArray(cropped_input_label_sitk_data)
    cropped_input_label_sitk.SetSpacing(input_label_sitk.GetSpacing())
    cropped_input_label_sitk.SetDirection(input_label_sitk.GetDirection())

    if verbose:
        print('.done')

    return cropped_sitk, cropped_input_label_sitk, crop_indices

def determine_empty_voxel_value(input_image):
    input_image_data = sitk.GetArrayFromImage(input_image)
    min_value = np.amin(input_image_data[int(input_image.GetSize()[2]/2),:,:])
    sd = np.nanstd(input_image_data)
    empty_voxel_value = int(min_value + sd/10)
    return empty_voxel_value

def determine_plane(image_sitk=None, sitk_direction=None):
    """
    Determines the orientation of the study (axial, sagittal, coronal). Must provide either a path to DICOM or a SimpleITK Image

    Parameters:
        input_sitk (SimpleITK.Image, optional): SimpleITK image

    Returns:
        orientation (str): Possible values: axial, sagittal, coronal. None if no input is provided
    """
    orientation = None
    d = None
    if image_sitk is not None:
        d = image_sitk.GetDirection()
    else:
        d = sitk_direction

    # Want absolute values of the vectors - maximum movement in any direction
    vector_x = list(map(abs, (d[0], d[3], d[6])))
    vector_y = list(map(abs, (d[1], d[4], d[7])))

    max_x = np.argmax(vector_x)
    max_y = np.argmax(vector_y)

    if max_x == 0 and max_y == 1:
        # Axial: Moving in L-R direction changes x the most, moving in AP direction changes y the most
        orientation = 'axial'
    elif max_x == 1 and max_y == 2:
        # Sagittal: Moving in L-R direction changes y the most, moving in AP direction changes z the most
        orientation = 'sagittal'
    elif max_x == 0 and max_y == 2:
        # Sagittal: Moving in L-R direction changes x the most, moving in AP direction changes z the most
        orientation = 'coronal'

    return orientation

def get_rotate_transform(input_image, angle, pitch=False, roll=False, yaw=False, verbose=False):
    """ Generates a transform to rotate an image along 1 axis
    Based on: https://stackoverflow.com/questions/56171643/simpleitk-rotation-of-mri-image

    Parameters:
        input_image: SimpleITK image that has been loaded by the load_sitk method
        angle: Angle in degrees to rotate the image counterclockwise
        pitch (bool): rotate the image in a pitch direction (around x axis)
        roll (bool): rotate the image in a roll direction (around y axis)
        yaw (bool): rotate the image in a yaw direction (around z axis)
        verbose (bool): Verbose flag
    Returns:
        rotated SimpleITK image    
    """
    start_time = None
    if verbose:
        start_time = time.time()
        print("Generating rotate transform..", end = '')
            
    angle_rads = np.deg2rad(angle)
    euler_transform = sitk.Euler3DTransform()
    x, y, z = input_image.GetSize()
    image_center = input_image.TransformIndexToPhysicalPoint((int(np.ceil(x/2)), int(np.ceil(y/2)), int(np.ceil(z/2))))
    euler_transform.SetCenter(image_center)

    direction = input_image.GetDirection()
    axis_angle = None
    
    if pitch:
        axis_angle = (direction[0], direction[3], direction[6], angle_rads)
    elif roll:
        axis_angle = (direction[1], direction[4], direction[7], angle_rads)
    elif yaw:
        axis_angle = (direction[2], direction[5], direction[8], angle_rads)

    if axis_angle is None:
        raise ValueError("transform.get_rotate_transform: pitch, roll, or yaw must be set to True")
    np_rot_mat = matrix_from_axis_angle(axis_angle)
    euler_transform.SetMatrix(np_rot_mat.flatten().tolist())

    if verbose:
        print('.done - time = ' + str(np.around((time.time() - start_time), decimals=1)) + 'sec')
    
    return euler_transform

def keep_only_largest_connected_components(prediction_data):
    """ Keeps the largest connected component of each label class (e.g. largest '1', largest '2', etc)

    Parameters:
        prediction_data (numpy Array): Data generated from prediction. Background = 0, labels are integers starting with 1

    Returns:
        prediction_data_new (numpy Array): Data with the largest connected components isolated
    """     
    prediction_data_new = np.zeros_like(prediction_data)
    num_labels = prediction_data.max()
    for label_num in range(1, num_labels + 1): # Iterate through each label number
        prediction_data_tmp = np.zeros_like(prediction_data)
        label_indices = np.where(prediction_data == label_num) # Create a tmp array with only the label number
        prediction_data_tmp[label_indices] = 1
        labels = label(prediction_data_tmp) # Identifies connected components, assigns each component a different number
        if labels.max() != 0:
            largest_cc_indices = labels == np.argmax(np.bincount(labels.flat)[1:])+1 # np.bincount counts the "size" of each component, np.argmax picks the largest one
            prediction_data_new[largest_cc_indices] = label_num
    return prediction_data_new

# This function is from https://github.com/rock-learning/pytransform3d/blob/7589e083a50597a75b12d745ebacaa7cc056cfbd/pytransform3d/rotations.py#L302
def matrix_from_axis_angle(a):
    """ Compute rotation matrix from axis-angle.
    This is called exponential map or Rodrigues' formula.
    https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R

def nii2sitk(nii_path, transform_pixels_to_standard_orientation=True, verbose=False):
    """ Loads an nii file as an SITK image

    Parameters:
        nii_path (str): Full path to an .nii or .nii.gz file
        transform_pixels_to_standard_orientation (bool): If true, align pixels to standard LPS orientation

    Returns:
        sitk.Image
    """
    if verbose:
        print('Loading nii as sitk..', end='')
    sitk_image = sitk.ReadImage(nii_path)
    sitk_image_data = sitk.GetArrayFromImage(sitk_image)
    if int(np.max(sitk_image_data)) > 1: # Don't cast to int if values are between 0 and 1.99 (sometimes values intended to be 0-1 creep over 1)
        sitk_image = toInt16(sitk_image) # Convert to 16 bit int

    if transform_pixels_to_standard_orientation:
        sitk_image = pixels_to_standard_orientation(sitk_image)
    
    if verbose:
        print('.done')
    return sitk_image

def pixels_to_standard_orientation(input_image):
    """ Rotates the pixels of an image to the expected orientation. Important to get expected behavior with rotations.
    In SimpleITK, every image is LPS, meaning the left, posterior, superior-most pixel has the greatest physical location value in mm.
    In standard orientation, the left, posterior, superior-most pixel should have the highest pixel index (x-size-1, y-size-1, z-size-1).
    The right, anterior, inferior-most pixel should be the origin (0,0,0).

    Parameters:
        input_sitk (SimpleITK.Image): Simple ITK image to manipulate

    Returns:
        standard_sitk (SimpleITK.Image): Simple ITK image with pixels in expected orientation
    """
    input_image_data = sitk.GetArrayFromImage(input_image) # z,y,x
    converted_image_data = input_image_data
    
    d = input_image.GetDirection()
    new_x_vector = (d[0], d[3], d[6])
    new_y_vector = (d[1], d[4], d[7])
    new_z_vector = (d[2], d[5], d[8])

    new_origin = None
    new_origin_indices = (0,0,0)

    first_index_location = input_image.TransformIndexToPhysicalPoint((0,0,0))
    last_index_location = input_image.TransformIndexToPhysicalPoint(input_image.GetSize())

    plane = determine_plane(image_sitk=input_image)
    image_modified = False
    if plane == 'axial':
        # Is the greatest x pixel index physically left or right?
        # Greatest physical position (biggest number) is left
        if last_index_location[0] < first_index_location[0]: # The highest x pixel value is right - should be left
            # Flip about the x axis
            converted_image_data = np.flip(converted_image_data, 2)

            # Flip the orientation x component
            new_x_vector = tuple(-np.array(new_x_vector)+0)

            # Flip the origin x component
            new_origin_indices = (input_image.GetSize()[0]-1, new_origin_indices[1], new_origin_indices[2])
            image_modified = True

        # Is the greatest y pixel index physically anterior or posterior?
        # Greatest physical position is posterior
        if last_index_location[1] < first_index_location[1]: # The highest y pixel value is anterior - should be posterior
            # Flip about the y axis
            converted_image_data = np.flip(converted_image_data, 1)

            # Flip the orientation y component
            new_y_vector = tuple(-np.array(new_y_vector)+0)

            # Flip the origin y component
            new_origin_indices = (new_origin_indices[0], input_image.GetSize()[1]-1, new_origin_indices[2])
            image_modified = True

        # Is the greatest z pixel index physically superior or inferior?
        # Greatest physical position is superior
        if last_index_location[2] < first_index_location[2]: # The highest z pixel value is inferior - should be superior
            # Flip about the z axis
            converted_image_data = np.flip(converted_image_data, 0)

            # Flip the orientation z component
            new_z_vector = tuple(-np.array(new_z_vector)+0)  

            # Flip the origin z component
            new_origin_indices = (new_origin_indices[0], new_origin_indices[1], input_image.GetSize()[2]-1)

            image_modified = True
        
    elif plane == 'coronal':
        # Is the greatest x pixel index physically left or right?
        # Greatest physical position (biggest number) is left
        if last_index_location[0] < first_index_location[0]: # The highest x pixel value is right - should be left
            # Flip about the x axis
            converted_image_data = np.flip(converted_image_data, 2)

            # Flip the orientation x component
            new_x_vector = tuple(-np.array(new_x_vector)+0)

            # Flip the origin x component
            new_origin_indices = (input_image.GetSize()[0]-1, new_origin_indices[1], new_origin_indices[2])

            image_modified = True

        '''
        # Is the greatest y pixel index physically superior or inferior?
        # Greatest physical position is superior
        if last_index_location[2] > first_index_location[2]: # The highest z pixel value is superior - should be inferior
            # Flip about the y axis
            converted_image_data = np.flip(converted_image_data, 1)

            # Flip the orientation y component
            new_y_vector = tuple(-np.array(new_y_vector)+0)

            # Flip the origin y component
            new_origin_indices = (new_origin_indices[0], input_image.GetSize()[1]-1, new_origin_indices[2])

            image_modified = True
        '''
        # Is the greatest y pixel index physically superior or inferior?
        # Greatest physical position is superior
        if last_index_location[2] < first_index_location[2]: # The highest z pixel value is inferior - should be superior
            # Flip about the y axis
            converted_image_data = np.flip(converted_image_data, 1)

            # Flip the orientation y component
            new_y_vector = tuple(-np.array(new_y_vector)+0)

            # Flip the origin y component
            new_origin_indices = (new_origin_indices[0], input_image.GetSize()[1]-1, new_origin_indices[2])

            image_modified = True        

        # Is the greatest z pixel index physically posterior or anterior?
        # Greatest physical position is posterior
        if last_index_location[1] < first_index_location[1]: # The highest z pixel value is anterior - should be posterior
            # Flip about the z axis
            converted_image_data = np.flip(converted_image_data, 0)

            # Flip the orientation z component
            new_z_vector = tuple(-np.array(new_z_vector)+0)

            # Flip the origin z component
            new_origin_indices = (new_origin_indices[0], new_origin_indices[1], input_image.GetSize()[2]-1)

            image_modified = True
    elif plane == 'sagittal':
        # Is the greatest x pixel index physically posterior or anterior?
        # Greatest physical position (biggest number) is posterior
        if last_index_location[1] < first_index_location[1]: # The highest x pixel value is anterior - should be posterior
            # Flip about the x axis
            converted_image_data = np.flip(converted_image_data, 2)

            # Flip the orientation x component
            new_x_vector = tuple(-np.array(new_x_vector)+0)

            # Flip the origin x component
            new_origin_indices = (input_image.GetSize()[0]-1, new_origin_indices[1], new_origin_indices[2])

            image_modified = True
        
        '''
        # Is the greatest y pixel index physically superior or inferior?
        # Greatest physical position is superior
        if last_index_location[2] > first_index_location[2]: # The highest z pixel value is superior - should be inferior
            # Flip about the y axis
            converted_image_data = np.flip(converted_image_data, 1)

            # Flip the orientation y component
            new_y_vector = tuple(-np.array(new_y_vector)+0)

            # Flip the origin y component
            new_origin_indices = (new_origin_indices[0], input_image.GetSize()[1]-1, new_origin_indices[2])

            image_modified = True
        '''

        # Is the greatest y pixel index physically superior or inferior?
        # Greatest physical position is superior
        if last_index_location[2] < first_index_location[2]: # The highest z pixel value is inferior - should be superior
            # Flip about the y axis
            converted_image_data = np.flip(converted_image_data, 1)

            # Flip the orientation y component
            new_y_vector = tuple(-np.array(new_y_vector)+0)

            # Flip the origin y component
            new_origin_indices = (new_origin_indices[0], input_image.GetSize()[1]-1, new_origin_indices[2])

            image_modified = True

        # Is the greatest z pixel index physically left or right?
        # Greatest physical position is left
        if last_index_location[0] < first_index_location[0]: # The highest z pixel value is right - should be left
            # Flip about the z axis
            converted_image_data = np.flip(converted_image_data, 0)

            # Flip the orientation z component
            new_z_vector = tuple(-np.array(new_z_vector)+0)

            # Flip the origin z component
            new_origin_indices = (new_origin_indices[0], new_origin_indices[1], input_image.GetSize()[2]-1)

            image_modified = True

    if image_modified:
        converted_image = sitk.GetImageFromArray(converted_image_data)
        new_origin = input_image.TransformIndexToPhysicalPoint(new_origin_indices)
        converted_image.SetOrigin(new_origin)

        new_direction = (new_x_vector[0], new_y_vector[0], new_z_vector[0],
                        new_x_vector[1], new_y_vector[1], new_z_vector[1],
                        new_x_vector[2], new_y_vector[2], new_z_vector[2])
        converted_image.SetDirection(new_direction)
        
        converted_image.SetSpacing(input_image.GetSpacing())
    else:
        converted_image = input_image
    
    return converted_image


def predict_high_res(input_sitk, initial_model_path, cropped_model_path, spacing=(1.5,1.5,1.5), roi=(96,96,96), crop_size=None, crop_roi=None, min_voxel_value=None, max_voxel_value=None, keep_only_largest=True, debug=False, verbose=False):
    """ Performs a prediction at the original spatial resolution of the image. Does this by doing an initial low-res prediction,
    then cropping the image around the prediciton and predicting again at the original resolution. Necessary due to memory 
    constraints. This is the most accurate, slowest prediction. Requires 2 models.

    Parameters:
        input_sitk (sitk.Image): Loaded sitk image
        initial_model_path (str): Path to model to be used for the initial low-resolution segmentation prediction
        cropped_model_path (str): Path to model to be used for the second cropped high-resolution segmentation prediction
        spacing (list): Optional spacing for initial low-res prediction. Defaults to (1.5,1.5,1.5)
        roi (list): Optional roi for initial low-res prediction. Defaults to (96,96,96)
        crop_size (list): Optional provided x,y or x,y,z crop size (e.g. (128,128) or (128,128,64)). Otherwise it crops to 1/4 the original size. Provide just x and y if z is to be calculated to 1/4 original size. Recommend x and y size be the size the model was trained at.
        crop_roi (list): Optional provided different ROI for the high-res cropped prediction (e.g. (96,96,48)). Otherwise it uses the roi argument. Recommend the crop_roi be the roi the model was trained at.
        min_voxel_value (int): smallest possible value of the provided image (need to specify to avoid outliers)
        max_voxel_value (int): largest possible value of the provided image (need to specify to avoid outliers)
        verbose (bool): Verbosity flag

    Returns:
        prediction_sitk (sitkImage): Prediction label for the provided image and model
    """ 
    start_time = None
    if verbose:
        start_time = time.time()
        print('Predicting iteratively with ' + initial_model_path + '..', end='')
    
    # Variables
    x_spacing = spacing[0]
    y_spacing = spacing[1]
    z_spacing = spacing[2]

    orig_spacing = input_sitk.GetSpacing()
    orig_size = input_sitk.GetSize() # x,y,z
    orig_x_size = orig_size[0]
    orig_y_size = orig_size[1]
    orig_z_size = orig_size[2]

    if crop_roi is None:
        crop_roi = roi

    if min_voxel_value is None or max_voxel_value is None:
        input_sitk_data = sitk.GetArrayFromImage(input_sitk)
        min_voxel_value = np.min(input_sitk_data)
        max_voxel_value = np.max(input_sitk_data)

    # Make initial prediction to find the structure
    predict_sitk = predict_low_res_interpolate(input_sitk, 
                    initial_model_path, 
                    spacing=(x_spacing, y_spacing, z_spacing), 
                    roi=roi, 
                    min_voxel_value=min_voxel_value, 
                    max_voxel_value=max_voxel_value, 
                    num_labels=1, 
                    keep_only_largest=True, 
                    verbose=False)
    predict_sitk_data = sitk.GetArrayFromImage(predict_sitk)
    final_predict_sitk = predict_sitk
    debug_path = '/home/neurorad/Downloads/h-debug'
    timestamp = str(int(time.time()))
    if debug:
        if not os.path.exists(debug_path):
            os.mkdir(debug_path)
        sitk.WriteImage(predict_sitk, os.path.join(debug_path, 'low_res_rh_predict-' + timestamp + '-label.nii.gz'))
    if np.max(predict_sitk_data) > 0:
        # Crop the image - if crop size isn't provided then calculate it
        if crop_size is None:
            crop_ratio = np.divide(orig_spacing, spacing)
            crop_size = np.multiply(crop_ratio, input_sitk.GetSize()).astype(int).tolist()
        cropped_sitk, cropped_predict_sitk, crop_indices = crop_around_prediction(input_sitk, predict_sitk, crop_size=crop_size)
        if debug:
            sitk.WriteImage(cropped_sitk, os.path.join(debug_path, 'cropped-' + timestamp + '.nii.gz'))

        # Predict on the cropped image at full resolution
        cropped_predict_sitk = predict_low_res_interpolate(cropped_sitk, 
                                cropped_model_path, 
                                spacing=orig_spacing, 
                                roi=crop_roi, 
                                min_voxel_value=min_voxel_value, 
                                max_voxel_value=max_voxel_value,                                 
                                num_labels=1, 
                                keep_only_largest=True, 
                                verbose=False)
        if debug:
            sitk.WriteImage(cropped_predict_sitk, os.path.join(debug_path, 'cropped-' + timestamp + '-label.nii.gz'))

        # Insert the cropped prediction back into the full resolution image
        final_predict_data = np.zeros((orig_z_size, orig_y_size, orig_x_size), dtype=np.int16)
        cropped_predict_data = sitk.GetArrayFromImage(cropped_predict_sitk)
        final_predict_data[crop_indices[0]:crop_indices[1], crop_indices[2]:crop_indices[3], crop_indices[4]:crop_indices[5]] = cropped_predict_data

        final_predict_sitk = sitk.GetImageFromArray(final_predict_data)
        final_predict_sitk.CopyInformation(input_sitk)

    if verbose:
        print('.done - time = ' + str(np.around((time.time() - start_time), decimals=1)) + 'sec') 

    return final_predict_sitk

def predict_low_res_interpolate(input_sitk, model_path, spacing=(1.5,1.5,1.5), roi=(96,96,96), min_voxel_value=None, max_voxel_value=None, num_labels=1, keep_only_largest=True, verbose=False):
    """ Same as predict_low_res but the prediction results are interpolated when resampled back to original image resolution.
    Results in a smoother and likely more accurate segmentation.

    Parameters:
        input_sitk (sitk.Image): Loaded sitk image
        model_path (str): Path to model to be used for segmentation prediction
        spacing (list): x,y,z spacing e.g. (1.5,1.5,1.5)
        roi (list): x,y,z matrix size e.g. roi=(96,96,96) - MUST be a multiple of 16
        min_voxel_value (int): smallest possible value of the provided image (need to specify to avoid outliers)
        max_voxel_value (int): largest possible value of the provided image (need to specify to avoid outliers)        
        num_labels (int): Number of labels to predict
        keep_only_largest (bool): Whether to only keep the largest connected component of each label
        verbose (bool): Verbosity flag

    Returns:
        prediction_sitk (sitkImage): Prediction label for the provided image and model
    """ 
    start_time = None
    if verbose:
        start_time = time.time()
        print('Predicting low res with ' + model_path + '..', end='')
    
    # Variables
    x_spacing = spacing[0]
    y_spacing = spacing[1]
    z_spacing = spacing[2]

    if min_voxel_value is None or max_voxel_value is None:
        input_sitk_data = sitk.GetArrayFromImage(input_sitk)
        min_voxel_value = np.min(input_sitk_data)
        max_voxel_value = np.max(input_sitk_data)

    # Prepare the image for prediction
    resampled_sitk = resample_spacing(input_sitk, new_spacing=[x_spacing,y_spacing,z_spacing])
    resampled_data = sitk.GetArrayFromImage(resampled_sitk)
    resampled_rescaled_data = np.interp(resampled_data, (min_voxel_value, max_voxel_value), (0, 1))  
    resampled_rescaled_data = np.transpose(resampled_rescaled_data, (2, 1, 0)) # Transpose from z,x,y to z,y,x (convert from sitk to RAS)
    
    resampled_rescaled_data = np.array(resampled_rescaled_data)[np.newaxis, np.newaxis, :, :, :]
    resampled_rescaled_data = torch.from_numpy(resampled_rescaled_data)
    resampled_rescaled_data = resampled_rescaled_data.type(torch.FloatTensor)

    # Make the prediction
    device = torch.device('cuda:0')
    model = monai.networks.nets.UNet(spatial_dims=3, in_channels=1, out_channels=num_labels+1, channels=(16, 32, 64, 128, 256),
                                    strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)                                  
    model.load_state_dict(torch.load(model_path))
    sw_batch_size = 4

    predict_data = sliding_window_inference(resampled_rescaled_data.to(device), roi, sw_batch_size, model)

    # Prepare the image for output (convert to numpy, undo the initial transforms of transpose and resample)
    predict_data = torch.argmax(predict_data, dim=1).detach().cpu().numpy()
    predict_data = predict_data[0,:,:,:]
    predict_data = np.transpose(predict_data, (2, 1, 0)) # Reverse transpose - basically convert back from RAS to sitk
    predict_data = predict_data.astype(np.float32)
    
    predict_sitk = sitk.GetImageFromArray(predict_data)
    predict_sitk.CopyInformation(resampled_sitk)
    orig_spacing = input_sitk.GetSpacing()
    predict_sitk = resample_spacing(predict_sitk, new_spacing=orig_spacing, interpolation=sitk.sitkLinear)
    predict_sitk_data = sitk.GetArrayFromImage(predict_sitk)

    # Odd correction from resampling - sometimes there is a "shell" of num_labels+1 at the periphery - remove that if present
    extra_labels_indices = np.where(predict_sitk_data > num_labels)
    predict_sitk_data[extra_labels_indices] = 0

    # Correct for linear interpolation
    zero_indices = np.where(predict_sitk_data < 0.5)
    one_indices = np.where(predict_sitk_data >= 0.5)
    predict_sitk_data[zero_indices] = 0.0
    predict_sitk_data[one_indices] = 1.0
    predict_sitk_data = predict_sitk_data.astype(np.int16)

    if num_labels == 1 and keep_only_largest:
        # From: https://stackoverflow.com/questions/47520487/how-to-use-python-opencv-to-find-largest-connected-component-in-a-single-channel?rq=1
        predict_sitk_data = keep_only_largest_connected_components(predict_sitk_data)

    # predict_sitk_data = z,y,x
    # input_sitk.GetSize() = x,y,z

    # Odd correction necessary - set the max x, y, z values = 0 (there's a "shell" label at these locations for some reason)
    # Compare x dim
    x_diff = predict_sitk_data.shape[2] - input_sitk.GetSize()[0]
    if x_diff > 0:
        for i in range(x_diff):
            predict_sitk_data = np.delete(predict_sitk_data, predict_sitk_data.shape[2]-1, axis=2)
    else:
        predict_sitk_data[:,:,predict_sitk_data.shape[2]-1] = 0
    if x_diff < 0:
        append_data = np.zeros((predict_sitk_data.shape[0], predict_sitk_data.shape[1], 1), dtype=np.int16)
        for i in range(abs(x_diff)):
            predict_sitk_data = np.append(predict_sitk_data, append_data, axis=2)
    
    # Compare y dim
    y_diff = predict_sitk_data.shape[1] - input_sitk.GetSize()[1]
    if y_diff > 0:
        for i in range(y_diff):
            predict_sitk_data = np.delete(predict_sitk_data, predict_sitk_data.shape[1]-1, axis=1)
    else:
        predict_sitk_data[:,predict_sitk_data.shape[1]-1,:] = 0
    if y_diff < 0:
        append_data = np.zeros((predict_sitk_data.shape[0], 1, predict_sitk_data.shape[2]), dtype=np.int16)
        for i in range(abs(y_diff)):
            predict_sitk_data = np.append(predict_sitk_data, append_data, axis=1)

    # Compare z dim
    z_diff = predict_sitk_data.shape[0] - input_sitk.GetSize()[2]
    if z_diff > 0:
        for i in range(z_diff):
            predict_sitk_data = np.delete(predict_sitk_data, predict_sitk_data.shape[0]-1, axis=0)
    else:
        predict_sitk_data[predict_sitk_data.shape[0]-1,:,:] = 0    
    if z_diff < 0:
        append_data = np.zeros((1, predict_sitk_data.shape[1], predict_sitk_data.shape[2]), dtype=np.int16)
        for i in range(abs(z_diff)):
            predict_sitk_data = np.append(predict_sitk_data, append_data, axis=0)

    corrected_predict_sitk = sitk.GetImageFromArray(predict_sitk_data)
    corrected_predict_sitk.CopyInformation(input_sitk)

    if verbose:
        print('.done - time = ' + str(np.around((time.time() - start_time), decimals=1)) + 'sec') 

    return corrected_predict_sitk

def remove_empty_columns(input_image, empty_voxel_value=None, segmentation=False):
    """ Removes all empty columns from an image. Any empty column has every value equal to the min value for the image.

    Parameters:
        input_image (sitk.Image): SITK Image to remove empty columns from
    Returns:
        output_image (sitk.Image): SITK Image with empty columns removed
    """
    # sitk data is z,y,x
    input_image_data = sitk.GetArrayFromImage(input_image)
    input_image_data_removed = None
    if empty_voxel_value is None:
        empty_voxel_value = determine_empty_voxel_value(input_image)
    else:
        if not segmentation:
            empty_voxel_value = empty_voxel_value+1 # Correct for noise introduced by bspline interpolation

    # The next 2 operations create a (1 x num_columns) array that is true if the column is empty, false otherwise
    empty_columns = np.all(input_image_data <= empty_voxel_value, axis=1)
    empty_columns = np.all(empty_columns, axis=0)

    # Then find indices of the non-empty columns
    non_empty_indices = np.where(empty_columns == False)[0]

    # Then find the number of right and left empty columns
    num_right_empty_columns = int(non_empty_indices[0])
    num_left_empty_columns = (input_image.GetSize()[0]-1) - non_empty_indices[-1]

    # Then delete the columns
    input_image_data_removed = input_image_data[:,:,num_right_empty_columns:input_image.GetSize()[0]-num_left_empty_columns]
    
    # Recaulculate the origin
    output_image = sitk.GetImageFromArray(input_image_data_removed)
    new_origin = None
    if num_right_empty_columns != 0: 
        new_origin = input_image.TransformIndexToPhysicalPoint((num_right_empty_columns,0,0))
    else:
        new_origin = input_image.GetOrigin()
    
    output_image.SetOrigin(new_origin)
    output_image.SetSpacing(input_image.GetSpacing())
    output_image.SetDirection(input_image.GetDirection())

    return output_image

def remove_empty_rows(input_image, empty_voxel_value=None, segmentation=False):
    """ Removes all empty rows from an image. Any empty row has every value equal to the min value for the image.

    Parameters:
        input_image (sitk.Image): SITK Image to remove empty rows from
    Returns:
        output_image (sitk.Image): SITK Image with empty rows removed
    """
    # sitk data is z,y,x
    input_image_data = sitk.GetArrayFromImage(input_image)
    input_image_data_removed = None
    if empty_voxel_value is None:
        empty_voxel_value = determine_empty_voxel_value(input_image)
    else:
        if not segmentation:
            empty_voxel_value = empty_voxel_value+1 # Correct for noise introduced by bspline interpolation

    # The next 2 operations create a (1 x num_rows) array that is true if the row is empty, false otherwise
    empty_rows = np.all(input_image_data <= empty_voxel_value, axis=2)
    empty_rows = np.all(empty_rows, axis=0)

    # Then find indices of the non-empty rows
    non_empty_indices = np.where(empty_rows == False)[0]

    # Then find the number of anterior and posterior empty rows
    num_anterior_empty_rows = int(non_empty_indices[0])
    num_posterior_empty_rows = (input_image.GetSize()[1]-1) - non_empty_indices[-1]

    # Then delete the rows
    input_image_data_removed = input_image_data[:,num_anterior_empty_rows:input_image.GetSize()[1]-num_posterior_empty_rows,:]
    
    # Recaulculate the origin
    output_image = sitk.GetImageFromArray(input_image_data_removed)
    new_origin = None
    if num_anterior_empty_rows != 0: 
        new_origin = input_image.TransformIndexToPhysicalPoint((0,num_anterior_empty_rows,0))
    else:
        new_origin = input_image.GetOrigin()
    
    output_image.SetOrigin(new_origin)
    output_image.SetSpacing(input_image.GetSpacing())
    output_image.SetDirection(input_image.GetDirection())

    return output_image

def remove_empty_slices(input_image, empty_voxel_value=None, segmentation=False):
    """ Removes all empty slices from an image. Any empty row has every value equal to the min value for the image.

    Parameters:
        input_image (sitk.Image): SITK Image to remove empty slices from
    Returns:
        output_image (sitk.Image): SITK Image with empty slices removed
    """
    # sitk data is z,y,x
    input_image_data = sitk.GetArrayFromImage(input_image)
    input_image_data_removed = None
    if empty_voxel_value is None:
        empty_voxel_value = determine_empty_voxel_value(input_image)
    else:
        if not segmentation:
            empty_voxel_value = empty_voxel_value+1 # Correct for noise introduced by bspline interpolation

    # The next 2 operations create a (1 x num_slices) array that is true if the slice is empty, false otherwise
    empty_slices = np.all(input_image_data <= empty_voxel_value, axis=1)
    empty_slices = np.all(empty_slices, axis=1)

    # Then find indices of the non-empty slices
    non_empty_indices = np.where(empty_slices == False)[0]

    # Then find the number of inferior and superior empty slices
    num_inferior_empty_slices = int(non_empty_indices[0])
    num_superior_empty_slices = (input_image.GetSize()[2]-1) - non_empty_indices[-1]

    # Then delete the slices
    input_image_data_removed = input_image_data[num_inferior_empty_slices:input_image.GetSize()[2]-num_superior_empty_slices,:,:]
    
    # Recaulculate the origin
    output_image = sitk.GetImageFromArray(input_image_data_removed)
    new_origin = None
    if num_inferior_empty_slices != 0: 
        new_origin = input_image.TransformIndexToPhysicalPoint((0,0,num_inferior_empty_slices))
    else:
        new_origin = input_image.GetOrigin()
    
    output_image.SetOrigin(new_origin)
    output_image.SetSpacing(input_image.GetSpacing())
    output_image.SetDirection(input_image.GetDirection())

    return output_image

def remove_padding(input_image, empty_voxel_value=None, xy=True, z=True, segmentation=False):
    """ Remove padding added to an image via add_padding

    Parameters:
        input_image (sitk.Image): SITK Image to remove padding from (will not be modified)
    Returns:
        output_image (sitk.Image): De-padded SITK Image
    """
    # Find the min voxel value at the first slice - assume that is padding
    # Otherwise, bspline interpolation introduces some lower values at the edge of the actual image
    if empty_voxel_value is None:
        empty_voxel_value = determine_empty_voxel_value(input_image)

    input_image_data = sitk.GetArrayFromImage(input_image)
    output_image = sitk.GetImageFromArray(input_image_data)
    output_image.CopyInformation(input_image)

    if xy:
        output_image = remove_empty_columns(output_image, empty_voxel_value=empty_voxel_value, segmentation=segmentation)
        output_image = remove_empty_rows(output_image, empty_voxel_value=empty_voxel_value, segmentation=segmentation)
    if z:
        output_image = remove_empty_slices(output_image, empty_voxel_value=empty_voxel_value, segmentation=segmentation)

    return output_image

def resample_spacing(input_image, new_spacing=[1,1,1], interpolation=sitk.sitkBSpline, verbose=False):
    """ Resamples to the specified pixel spacing, in mm

    Parameters:
        input_image (SimpleITK.Image): Simple ITK image to be resampled
        new_spacing (list, default = [1,1,1]): New spacing in mm, [x,y,z]
        is_label (Boolean - default = False): Whether to resample the image as anatomic or label
        verbose (bool): Verbose flag

    Returns:
        output_image (SimpleITK.Image): Resampled Simple ITK image
    """
    # This page really helped:
    # https://gist.github.com/mrajchl/ccbd5ed12eb68e0c1afc5da116af614a
    
    start_time = None
    if verbose:
        start_time = time.time()
        print("Beginning resample_spacing..", end = '')

    original_spacing = input_image.GetSpacing()
    original_size = input_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_image.GetPixelIDValue())
    resample.SetInterpolator(interpolation)

    output_image = resample.Execute(input_image)
    #output_image = remove_erroneous_high_voxels(input_image, output_image)

    if verbose:
        print('.done - time = ' + str(np.around((time.time() - start_time), decimals=1)) + 'sec')

    return output_image

def to_axial(input_sitk, interpolation=sitk.sitkBSpline, rotate_without_interpolation=False, maintain_physical_location=True, maintain_data=True, segmentation=False, verbose=False):
    """ Rotates an image into the axial plane. Does nothing if already axial.

    Parameters:
        input_sitk: SimpleITK image that has been loaded by the load_sitk method
        interpolation (sitk interpolation method): SimpleITK interpolation method - sitk.sitkLinear is faster, sitk.sitkBSpline is more accurate and slightly slower
        rotate_without_interpolation (bool): If true, only rotates pixels in 90 degree increment with no interpolation
        maintain_physical_location (bool): If true, the physical location of the image is maintained
        verbose (bool): Verbose flag
    Returns:
        axial_sitk (sitk.Image): Axial version of image. None returned if apply_transforms=False
        transforms (list): Lisimage_sitkt of transforms (None returned if only rotating without interpolation)
    """
    standard_input_sitk = pixels_to_standard_orientation(input_sitk)
    axial_sitk = None
    transforms = None
    
    plane = determine_plane(image_sitk=standard_input_sitk)
    if plane == 'axial':
        axial_sitk = standard_input_sitk
    elif plane == 'coronal':         
        transforms = []
        pitch_transform = get_rotate_transform(standard_input_sitk, -90, pitch=True, verbose=False) # Clockwise
        transforms.append(pitch_transform)
        axial_sitk = composite(standard_input_sitk, transforms, maintain_xy_data=maintain_data, maintain_z_data=maintain_data, maintain_physical_location=maintain_physical_location, interpolation=interpolation, segmentation=segmentation, verbose=verbose)
    elif plane == 'sagittal':
        transforms = []
        roll_transform = get_rotate_transform(standard_input_sitk, 90, roll=True, verbose=False) # Counter-clockwise
        transforms.append(roll_transform)
        pitch_transform = get_rotate_transform(standard_input_sitk, 90, pitch=True, verbose=False) # Counter-clockwise
        transforms.append(pitch_transform)
        axial_sitk = composite(standard_input_sitk, transforms, maintain_xy_data=maintain_data, maintain_z_data=maintain_data, maintain_physical_location=maintain_physical_location, interpolation=interpolation, segmentation=segmentation, verbose=verbose)

    axial_sitk = pixels_to_standard_orientation(axial_sitk)
    return axial_sitk

def toInt16(input_image, force_signed=False, force_unsigned=False):
    """ Casts an SITK image to signed or unsigned int16 unless it is 4D (e.g. DWI source) or RGB (8 bit). 
    Parameters:
        input_image (SimpleITK.Image): SimpleITK image, not modified

    Returns:
        image_16 (SimpleITK.Image): The input_image as a 16 bit image
    """
    sitk_image_data = sitk.GetArrayFromImage(input_image)
    image_16 = sitk.GetImageFromArray(sitk_image_data)
    #image_16.CopyInformation(input_image) # Weird bug on 1 study - said dimensions didn't match. Just manually copy 3 relevant components instead.
    image_16.SetOrigin(input_image.GetOrigin())
    image_16.SetSpacing(input_image.GetSpacing())
    image_16.SetDirection(input_image.GetDirection())    
    if image_16.GetDimension() < 4: # Casting doesn't work for a 4D dataset like DWI source
        if image_16.GetNumberOfComponentsPerPixel() == 1: # Don't cast to 16 bit if RGB, which has 3 components per pixel (R,G,B)
            if np.min(sitk_image_data) > -1 or force_unsigned: # Unsigned int
                if np.max(sitk_image_data) > 2**16-1 or image_16.GetPixelID() < 2: # Scale if 8 bit or if values greater than max 16 bit
                    rescaled_sitk_image_data = np.interp(sitk_image_data, (np.min(sitk_image_data), np.max(sitk_image_data)), (0, 2**12-1)) # Confusing, but 16 bit medical images only use up to 12 bits (4096 max value) - for CR at least - test for CT and MR
                    rescaled_sitk = sitk.GetImageFromArray(rescaled_sitk_image_data)
                    rescaled_sitk.CopyInformation(image_16)    
                    image_16 = rescaled_sitk
                image_16 = sitk.Cast(image_16, sitk.sitkUInt16)
            elif np.min(sitk_image_data) < 0 or force_signed: # Signed int
                if np.max(sitk_image_data) > int(2**16/2)-1 or image_16.GetPixelID() < 2: # Scale if 8 bit or if values greater than max 16 bit
                    rescaled_sitk_image_data = np.interp(sitk_image_data, (np.min(sitk_image_data), np.max(sitk_image_data)), (-int(2**12/2), int(2**12/2)-1))
                    rescaled_sitk = sitk.GetImageFromArray(rescaled_sitk_image_data)
                    rescaled_sitk.CopyInformation(image_16)    
                    image_16 = rescaled_sitk      
                image_16 = sitk.Cast(image_16, sitk.sitkInt16)
    return image_16

def transform_direction(input_sitk, transform):
    """ Applies a transform to the image direction vectors

    Parameters:
        input_sitk: SimpleITK image that has been loaded by the load_sitk method
        transform: SimpleITK transform to apply to the direction vectors
    Returns:
        new_direction (list): List of vectors compatible with SimpleITK.SetDirection()
    """    
    x, y, z = input_sitk.GetSize()
    image_center = input_sitk.TransformIndexToPhysicalPoint((int(np.ceil(x/2)), int(np.ceil(y/2)), int(np.ceil(z/2))))
    
    # Adjust the image direction
    d = input_sitk.GetDirection()
    vector_x = (d[0], d[3], d[6])
    vector_y = (d[1], d[4], d[7])
    vector_z = (d[2], d[5], d[8])                                                

    new_vector_x = transform.TransformVector(vector_x, image_center)
    new_vector_y = transform.TransformVector(vector_y, image_center)
    new_vector_z = transform.TransformVector(vector_z, image_center)

    new_direction = (new_vector_x[0], new_vector_y[0], new_vector_z[0],
                    new_vector_x[1], new_vector_y[1], new_vector_z[1],
                    new_vector_x[2], new_vector_y[2], new_vector_z[2])

    return new_direction 

def transform_origin(input_sitk, transform):
    """ Transforms the origin of an SimpleITK image

    Parameters:
        input_sitk: SimpleITK image that has been loaded by the load_sitk method
        transform: SimpleITK transform to apply to the origin
    Returns:
        new_direction (list): List of vectors compatible with SimpleITK.SetDirection()
    """ 
    new_origin = transform.TransformPoint(input_sitk.GetOrigin())
    return new_origin