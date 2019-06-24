import numpy as np
import json, os, pydicom, copy
from os import listdir
import nibabel as nib

def normalize_numpy_array(x,minx=-1,maxx=1, minx0=None, maxx0=None):
    """
    x=np.random.normal(0,0.1,size=(3,4))
    print(x,"\n")
    for xx in x:
        y=normalize_numpy_array(xx,minx=-1,maxx=1, minx0=np.min(x), maxx0=np.max(x))
        print(y)
    print("\n",normalize_numpy_array(x,minx=-1,maxx=1, minx0=np.min(x), maxx0=np.max(x)))

    output:
    [[-0.08430609  0.10034664  0.00050507  0.0413716 ]
     [ 0.01563976 -0.10492962  0.21774622 -0.05562354]
     [ 0.10780014  0.00253532  0.14603062 -0.02202592]] 

    [-0.87217184  0.27233735 -0.34649778 -0.0932    ]
    [-0.25269034 -1.          1.         -0.69439246]
    [ 0.31853544 -0.33391394  0.55549448 -0.48614868]

     [[-0.87217184  0.27233735 -0.34649778 -0.0932    ]
     [-0.25269034 -1.          1.         -0.69439246]
     [ 0.31853544 -0.33391394  0.55549448 -0.48614868]]
    """
    if minx0 is None:
        minx0 = np.min(x)
    if maxx0 is None:
        maxx0=np.max(x)
    if minx0==maxx0:
        print("normalize_numpy_array: constant array, return unmodified input")
        return x
    # print("              normalize_numpy_array().minx0,miaxx0=%.5f,%.5f"%(minx0,maxx0))
    midx0=0.5*(minx0+maxx0)
    midx=0.5*(minx+maxx)
    y=x-midx0
    y=y*(maxx-minx)/(maxx0-minx0)
    return y+midx

def json_to_dict(json_dir):
    f = open(json_dir)
    fr = f.read()
    data = json.loads(fr)
    f.close()
    return data


DESCRIPTION = '''Processing PANCREAS CT-82.
Each data point is a dcm file. Each dcm file consists of tags and values.
Tags are like (0008, 0030) or (0020, 0032).
Data source: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
The default downloaded data is saved in directory format like:
  PANCREAS_0001/11-24-2015-PANCREAS0001-Pancreas-18957/Pancreas-99667/000000.dcm
and likewise for other slices 000001.dcm, 000002.dcm etc

--mode:
    stack_all
      If the downloaded data PANCREAS_0001, PANCREAS_0002 etc are stored in config: [directory][multiple_patients],
      then this function will combine the slices in each PANCREAS_XXXX/somefolder/somefolder/YYYYYY.dcm into 3D volume
      in .nii format with dummy affine value. 
    Requires
    --config_dir: full path to the configuration file, typically called config.json
    Relevant configurations: 
    1. [normalize]. 
    2. [directory]. Subconfigs: [mutliple_patients], [save_folder], [suffix_check]
    For example:
    python ctpan.py --mode stack_all --config_dir D:/Desktop@D/dicom_conv/config.json



Some other modes to observe PANCREAS CT-82 data
--mode:
    (1) verify_info
        For each slice in the patient's PANCREAS CT, print the dicom tag and values.
        Requires --dir: directory to the file of a patient. 
        For example:
        python ctpan.py --mode verify_info --dir path/to/PANCREAS_0001
        python ctpan.py --mode verify_info --dir D:/Desktop@D/dicom_conv/PANCREAS_0001
    
    (2) slice_info
        For each slice in the patient's PANCREAS CT, print the dicom tag and value indexed by tag_number.
        Tag numbers are not explicitly defined. As an example, 
          tag_number=0 is (0008, 0005) Specific Character Set
        Requires
        --dir: directory to the file of a patient. 
        --tag_number: integer pointing to a tag in the dicom file, starting with 0 and max 46
        For example:
        python ctpan.py --mode slice_info --dir D:/Desktop@D/dicom_conv/PANCREAS_0001 --tag_number 31

    (3) another_view
        Each slice of a patient's CT is stored in different dcm file 000000.dcm, 000001.dcm etc.
        They are likely to contain the same data. For example, the date during which the slice is taken
        should be the same for the same patient. This mode prints out the tags which show DIFFERENT values.  
        Requires
        --dir: directory to the file of a patient. 
        For example:
        python ctpan.py --mode another_view --dir D:/Desktop@D/dicom_conv/PANCREAS_0001

    (4) z_position
        The z axis values of [(0020, 0032) Image Position (Patient)] in each slice are listed.
        Once the list is sorted, we check if the list is equivalent to range(z_min, z_max) for 
        some z_min, z_max.
        Requires
        --dir: directory to the file of a patient. 
        For example:
        python ctpan.py --mode z_position --dir D:/Desktop@D/dicom_conv/PANCREAS_0001

    (5) stack_one
        For one patient, stack the 2D slices into a 3D volume in .nii format with dummy affine value.
        return cube, this_min, this_max 
            where this_min and this_max are respectively the min and max values of the cube
        Requires
        --dir: directory to the file of a patient. 
        For example:
        python ctpan.py --mode stack_one --dir D:/Desktop@D/dicom_conv/PANCREAS_0001

    (6) stack_one_and_save
        For one patient, stack the 2D slices into a 3D volume and save.
        Requires
        --dir: directory to the file of a patient. 
        --out_name: name to save the stacked volume with.
        --target_folder: directory to save the file in.
        Optional
        --config_dir: directory of the config file. The format can be found by running this program 
          in 'generate_config' mode.
        For example:
        python ctpan.py --mode stack_one_and_save --dir D:/Desktop@D/dicom_conv/PANCREAS_0001 --out_name PANCREAS_0001 --target_folder D:/Desktop@D/dicom_conv/output_dir
        python ctpan.py --mode stack_one_and_save --dir D:/Desktop@D/dicom_conv/PANCREAS_0001 --out_name PANCREAS_0001 --target_folder D:/Desktop@D/dicom_conv/output_dir --config_dir D:/Desktop@D/dicom_conv/config.json

    (7) mass_observe
        In a folder which stores patients' file, find the maximum and minimum values
          of the stored pixel arrays. This is useful for observation
        Requires 
        --dir: directory to the file of all patients. If the directory is data, then we have
            the patients' files stored as data/PANCREAS0001, data/PANCREAS0002 etc 
        For example:
        python ctpan.py --mode mass_observe --dir D:/Desktop@D/Attention-Gated-Networks/data/TCIA_pancreas_in_DICOM_format/PANCREAS-CT-82

    (8) generate_config
        Generate a default configuration file.
        Requires
        --dir: directory to put the config file.
        For example:
        python ctpan.py --mode generate_config --dir D:/Desktop@D/dicom_conv

    (9) mass_observe_labels
        Observe labels of segmentation files.
        Requires
         --dir: directory to the file of all patients. If the directory is data, then we have
            the patients' files stored as data/PANCREAS0001, data/PANCREAS0002 etc 
        For example:
        python ctpan.py --mode mass_observe_labels --dir D:\Desktop@D\Attention-Gated-Networks\data\TCIA_pancreas_labels-02-05-2017
'''

CONFIG_ARG_EXPLANATIONS = ["\n# [normalize][0]\nif set to True, then the 2D slices are stacked into a volume normalized using the following arguments\n# [normalize][1]\n",
"global_max and global_min are user-specified max and min values threshold of the 3D volume across all patients. target_max and target_min are the final resized max ",
"and min values\n# [directory][multiple_patients]\nDirectory containing all patients folders PANCREAS0001, PANCREAS0002 etc",
"\n# [directory][save_folder]\n",
"The directory to save the output of the process.",
"\n# [directory][suffix_check]\n",
"Make sure the only patient file whose folder starts with this suffix is included as the valid patient file."]