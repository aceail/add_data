import pydicom 
import os
import pathlib
import glob
import math
import numpy as np
import SimpleITK as sitk
from skimage import morphology
from scipy import ndimage, stats
from einops import rearrange
from collections.abc import Iterable
from skimage import morphology
from scipy import ndimage
import cv2
import sys
import nibabel as nib
import warnings
# Change the SITK_SHOW_EXTENSION
os.environ["SITK_SHOW_EXTENSION"] = 'null'
# Ignore all warnings
warnings.filterwarnings("ignore")
sitk.ProcessObject_SetGlobalWarningDisplay(False)
sys.path.append(r'./module/SkullStripping')
# from SkullStrippingModel import SkullStripping




def get_padding_size(height, width):
    # Assuming image is a NumPy array of shape (height, width, channels)
    # Determine the difference and the required padding
    diff = abs(height - width)
    pad_height = diff // 2 if height < width else 0
    pad_width = diff // 2 if width < height else 0

    # Apply padding
    # If the difference is odd, we need one more row/column of padding
    if diff % 2:
        if width < height:
            pad_h_list = (pad_height, pad_height)
            pad_w_list = (pad_width+1, pad_width)
        else:
            pad_h_list = (pad_height+1, pad_height)
            pad_w_list = (pad_width, pad_width)
    else:
        pad_h_list = (pad_height, pad_height)
        pad_w_list = (pad_width, pad_width)
    return pad_h_list, pad_w_list

def pad_in_3d(sitk_img):
    pad_h_list, pad_w_list = get_padding_size(sitk_img.GetSize()[0], sitk_img.GetSize()[1])
    pad_filter = sitk.ConstantPadImageFilter()
    pad_filter.SetPadLowerBound([pad_h_list[0], pad_w_list[0], 0])
    pad_filter.SetPadUpperBound([pad_h_list[1], pad_w_list[1], 0])
    pad_filter.SetConstant(-1000)
    padded_image = pad_filter.Execute(sitk_img)
    return padded_image


def resize_image(input_image):
    """
    Resize the given SimpleITK image to the specified output size.
    
    Parameters:
    - input_image: SimpleITK image object to be resized
    - output_size: Tuple specifying the new size (width, height, depth)
    
    Returns:
    - SimpleITK image object with the new size
    """
    # Get the original size and spacing
    input_size = input_image.GetSize()
    input_spacing = input_image.GetSpacing()
    if input_size[0] == 512 and input_size[1] == 512:
        return input_image
    output_size = [512, 512,input_image.GetSize()[-1]]
    # Calculate the new spacing
    output_spacing = [
        insp * sz / osz
        for insp, sz, osz in zip(input_spacing, input_size, output_size)
    ]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetOutputDirection(input_image.GetDirection())
    resample.SetOutputOrigin(input_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(input_image.GetPixelIDValue())

    resample.SetInterpolator(sitk.sitkBSpline)
    
    return resample.Execute(input_image) 


def file_plane(loc):
    r""" get axis of the patients
    args:
        loc (list) : image orientation patient
    return:
        axis_info (str) : 
    """
 
    row_x = round(loc[0])
    row_y = round(loc[1])
    row_z = round(loc[2])
    col_x = round(loc[3])
    col_y = round(loc[4])
    col_z = round(loc[5])

    if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 0:
        return "Coronal"

    if row_x == 0 and row_y == 1 and col_x == 0 and col_y == 0:
        return "Sagittal"

    if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 1:
        return "Axial"

    return "Unknown"


def read_nii_aix(file_path):
    masks = nib.load(file_path)
    masks = masks.get_fdata().transpose(2,0,1)
    return masks


class ImageWrapper:
    def __init__(self, sitk_image):
        self.sitk_image = sitk_image

    def to_numpy(self):
        return sitk.GetArrayFromImage(self.sitk_image)

class PreprocessingCTImage:
    def __init__(self):
        pass
    
    ## before
    @staticmethod
    def read_dicom(file_dir, sid=None):
        r""" read from dicom directory
        ars:
            file_dir (str) : directory where dicom files are located
                            (** the directory should contains one tyes of
                                SeriesInstanceUID, otherwise, the method
                                assert error.)
        """
        #sitk.ProcessObject_SetGlobalWarningDisplay(False)
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(file_dir)
        if sid == None:
            assert len(series_IDs) == 1, 'There are more than one type of images found!'
            series_uid = series_IDs[0]
        else:
            assert sid in series_IDs, f'{sid} not in given folder'
            series_uid = sid

                
        dicom_names = reader.GetGDCMSeriesFileNames(file_dir,series_IDs[0])
        #print(dicom_names)
        # Create a dictionary to track unique instances based on Instance Number and Series Instance UID
        unique_instances = {}

        # Iterate through DICOM files
        for dicom_name in dicom_names:
            dicom_image = sitk.ReadImage(dicom_name)

            # Get Instance Number and Series Instance UID
            instance_number = dicom_image.GetMetaData('0020|0013')
            series_instance_uid = dicom_image.GetMetaData('0020|000e')

            # Check if Instance Number and Series Instance UID combination is already present
            key = (instance_number, series_instance_uid)
            if key in unique_instances:
                continue  # Skip duplicate DICOM files

            # Add unique DICOM files to the dictionary
            unique_instances[key] = dicom_name
        
        

        # Select unique DICOM file names
        dicom_names_filtered = list(unique_instances.values())

        reader.SetFileNames(dicom_names_filtered)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        img3d = reader.Execute()

        ds = pydicom.dcmread(dicom_names_filtered[0], force=True, stop_before_pixels=True)
        if hasattr(ds, 'ImageOrientationPatient'):
            image_plane = file_plane(ds.ImageOrientationPatient)
            if image_plane != 'Axial' or image_plane == 'Unknown':
                img3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        else:
            img3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            
        img3d = sitk.DICOMOrient(img3d, "LPS")

        return img3d
    
    @staticmethod
    # def read_dicom(file_dir, sid=None):

    #     reader = sitk.ImageSeriesReader()
    #     series_IDs = reader.GetGDCMSeriesIDs(file_dir)
    #     series_counts = {
    #         sid: len(reader.GetGDCMSeriesFileNames(file_dir, sid))
    #         for sid in series_IDs
    #         }
    #     series_uid = max(series_counts.items(), key=lambda x: x[1])[0]
    #     if sid is None:
    #         assert len(series_IDs) <= 2, f'There are more than two type of images found!, {series_IDs}'
    #         # series_uid = series_IDs[0]
            

    #     else:
    #         assert sid in series_IDs, f'{sid} not in given folder'
    #         series_uid = sid

    #     # dicom_names = reader.GetGDCMSeriesFileNames(file_dir, series_uid)
    #     dicom_names = []
    #     for filename in os.listdir(file_dir):
    #         filepath = os.path.join(file_dir, filename)
    #         try:
    #             ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
    #             if 'SOPInstanceUID' in ds:
    #                 dicom_names.append(filepath)
    #         except Exception as e:
    #             print(f"Skipping non-DICOM file: {filename} - {e}")
        

    #     unique_instances = {}
    #     normal_ref_shape = None
    #     modified_dicom_paths = []

    #     for dicom_name in dicom_names:
    #         dicom_image = sitk.ReadImage(dicom_name)
    #         instance_number = dicom_image.GetMetaData('0020|0013')
    #         series_instance_uid = dicom_image.GetMetaData('0020|000e')
    #         key = (instance_number, series_instance_uid)

    #         if key in unique_instances:
    #             continue

    #         ds = pydicom.dcmread(dicom_name, force=True)
    #         image_type = getattr(ds, 'ImageType', [])

    #         if 'SCOUT' in image_type or 'LOCALIZER' in image_type:
    #             # ref shape 확보 (처음 나오는 일반 이미지 기준)
    #             if normal_ref_shape is None:
    #                 for ref_path in dicom_names:
    #                     ref_ds = pydicom.dcmread(ref_path, force=True)
    #                     if 'SCOUT' not in getattr(ref_ds, 'ImageType', []):
    #                         normal_ref_shape = ref_ds.pixel_array.shape
    #                         break

    #             if normal_ref_shape is not None:
    #                 pixel_array = ds.pixel_array
    #                 resized_array = cv2.resize(pixel_array, (normal_ref_shape[1], normal_ref_shape[0]), interpolation=cv2.INTER_LINEAR)
    #                 ds.Rows, ds.Columns = normal_ref_shape
    #                 ds.PixelData = resized_array.astype(pixel_array.dtype).tobytes()
    #                 ds.ImageType = [t for t in image_type if t.upper() != 'SCOUT']

    #                 # 수정된 파일을 임시로 저장
    #                 temp_path = dicom_name.replace('.dcm', '_mod.dcm')
    #                 ds.save_as(temp_path)
    #                 dicom_name = temp_path  # 이 파일을 사용하도록 교체
    #                 modified_dicom_paths.append(temp_path)

    #         unique_instances[key] = dicom_name

    #     dicom_names_filtered = list(unique_instances.values())

    #     reader.SetFileNames(dicom_names_filtered)
    #     reader.MetaDataDictionaryArrayUpdateOn()
    #     reader.LoadPrivateTagsOn()

    #     img3d = reader.Execute()

    #     ds = pydicom.dcmread(dicom_names_filtered[0], force=True, stop_before_pixels=True)
    #     if hasattr(ds, 'ImageOrientationPatient'):
    #         image_plane = file_plane(ds.ImageOrientationPatient)
    #         if image_plane != 'Axial' or image_plane == 'Unknown':
    #             img3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    #     else:
    #         img3d.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    #     img3d = sitk.DICOMOrient(img3d, "LPS")

    #     # 원하면 임시 파일 삭제 가능
    #     # for path in modified_dicom_paths:
    #     #     os.remove(path)

    #     return img3d

    
    @staticmethod
    def read_nii(file_path):
        r""" read from nifti file format
        args:
            file_path (str) : absolute or relative path of nifti files (*nii, *nii.gz)
        """
        image= sitk.ReadImage(file_path, imageIO='NiftiImageIO')
        image = sitk.DICOMOrient(image, "LPS")
        return image
    @staticmethod
    def save_as_nifti(image, save_path):
        r"""Save sitk image object to nii.gz extention file
        args:
            image : SimpleITK image object
        """
        writer = sitk.ImageFileWriter()
        writer.SetFileName(save_path)
        image = sitk.DICOMOrient(image, "LPS")
        writer.Execute(image)
    @staticmethod
    def image_window(image, level=40, width=80):
        r""" apply windowing to an original CT array in HU
             (if you apply this method it would return normalized array 
              in 0-255 range. If you want to apply just window please use
              window_image function in this file)
        args:
            image (sitk.image)   : SimpleITK image object in HU 
            level (int)          : window center 
            width (int)          : window width
        return:
            img_win (sitk.image) : SimpleITK image object in 0-255 scale
        """
        windowing = sitk.IntensityWindowingImageFilter()
        windowing.SetWindowMinimum(level - width//2)
        windowing.SetWindowMaximum(level + width//2)
        img_win = windowing.Execute(image)

        return img_win
    @staticmethod
    def to_sitk(numpy_arr, image):
        r""" convert numpy array to SimpleITK image object
        args: 
            numpy_arr (np.array) : numpy array of CT image (z, **)
            image (sitk.image)   : SimpleITK image object from which an input array copy information
        return:
            trans_image (sitk.image) : SimpleITK image object that has array information of input numpy_arr 
        """
        trans_image = sitk.GetImageFromArray(numpy_arr)
        trans_image.CopyInformation(image)
        return trans_image
    @staticmethod
    def to_numpy(image):
        r""" convert SimpleITK image object to numpy array
        args: 
            image (sitk.image) : SimpleITK image object
        return:
            numpy array
        """
        return sitk.GetArrayFromImage(image)
    @staticmethod
    def gantryRemoval(image, np_object=False, return_np=False):
        r"""Intensity based tight crop
        args:
           ct_image (sitk.image) : SimpleITK image object
                                   Raw CT scan (Hounsfield scale)
        return:
           tmp_img (sitk.image)  : SimpleITK image object with granty removed
        """
        if not np_object:
            image_np = sitk.GetArrayFromImage(image)#self.to_numpy(image)
        else:
            image_np = image
        brain_image = window_image(image_np, 40, 80) # brain-tissue
        seg_template = morphology.dilation(brain_image, np.ones((1, 1, 1)))
        
        labels, label_nb = ndimage.label(seg_template)
        label_count = np.bincount(labels.ravel().astype(np.int16))
        label_count[0] = 0

        mask = labels == label_count.argmax()
        mask = morphology.dilation(mask, np.ones((1, 1, 1)))
        mask = ndimage.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((1, 3, 3)))
        masked_img = np.where(mask, image_np, -1000)
        masked_img[masked_img<-1000] = -1000
        trans_image = sitk.GetImageFromArray(masked_img)
        trans_image.CopyInformation(image) 
        if not return_np:
            return trans_image#self.to_sitk(masked_img, image)
        else:
            return masked_img
    #@staticmethod
    def read_as_numpy(file_path, window=None, remove_noise=True):
        r""" read nii or dicom file and convert it to numpy array with some preprocessing
        args:
            file_path (str) : path to file 
            window (iter) : window center and width. if None is given no windowing is applied
            remove_noise (bool) : whether or not to appy gantry removal to image
        return:
            image (np) : numpy array of given image
        """
        extension = pathlib.Path(file_path).suffix
        if extension == '.gz' or extension == '.nii':
            image = PreprocessingCTImage.read_nii(file_path)
        elif extension == '':
            image = PreprocessingCTImage.read_dicom(file_path)
        else:
            raise TypeError("Only dicom or nii extensions are allowed")

        if remove_noise:
            image = PreprocessingCTImage.gantryRemoval(image)

        if window != None:
            assert isinstance(window, Iterable), f'Window should be either tuple or list, but I got {type(window)}'
            image = PreprocessingCTImage.image_window(image, window[0], window[1])
        return PreprocessingCTImage.to_numpy(image)
    
    @staticmethod
    def resample_volume_v2(image, target_thickness = None):
        r"""Resample funtion resample thickness of the slices
        args:
            image : SimpleITK image object
                    Non-windowed CT scan in LPS orientation
            target_thickness : Thickness into which we would like to convert 
        returns:
            image : SimpleITK image object with Downsampled size
        """
        # just make sure that image is in LPS form
        image = sitk.DICOMOrient(image, "LPS")
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        new_spacing = list(original_spacing)
        if target_thickness:
            new_spacing[-1] = target_thickness
        new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, new_spacing)]
        return sitk.Resample(image, new_size, sitk.Transform(), sitk.sitkLinear, 
                            image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                            image.GetPixelID())



def process_to_sqaure(raw_hu):
    """ some brain ct scans are in circular shape; thus, images outside the circles should be handled correctly
    args:
        raw_hu (np.array) : numpy array of ct scan in HU 
    return:
        raw_hu (np.array) : numpy array of ct scan in squared form

    """
    min_hu = np.min(raw_hu)
    if min_hu < -1100 and np.sum(raw_hu == min_hu) > (512**2 - (math.pi*(256)**2))*0.9:
        min_hu_new = np.min(raw_hu*(raw_hu != min_hu))
        raw_hu[raw_hu<min_hu_new] = min_hu_new
    return raw_hu


def window_image(image, window_center, window_width):
    """ apply windowing to pixel array
    args ::
        image (np.array) : pixel array with standardized HU 
        window_center (int|float) : value of window center
        window_width (int|float) : value of window width
    return :
        window_image (np.array) : pixel array with appied window 
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image




def normalize_array(img, norm_max=255, norm_min=0, arr_max=None, arr_min=None):
    """ min max scaler
    args:
        img (np.array) : image array with certain range of scale
        norm_max (int) : maximum of output scale
        norm_min (int) : minimum of output scale
        arr_max (int) : predefined maximum of array
        arr_min (int) : predefined miminum of array
    return:
        img_scaled (np.array) : scaled image array
    """
    arr_min = arr_min if arr_min is not None else np.min(img)
    arr_max = arr_max if arr_max is not None else np.max(img)
    if arr_min == arr_max:
        img_scaled = np.zeros(img.shape)
    else:
        img_std = (img - arr_min)/(arr_max -  arr_min)
        img_scaled = img_std*(norm_max-norm_min) + norm_min
    img_scaled = np.uint8(img_scaled)
    return img_scaled





