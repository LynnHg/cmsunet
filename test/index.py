import os
import numpy as np
import nibabel as nb
from utils.misc import check_mkdir
import SimpleITK as sitk


def npy_to_nii(data_path, save_path):
    check_mkdir(save_path)
    name_list = [str(i) for i in range(201, 221)]
    data_list = os.listdir(data_path)

    for name in name_list:
        segs = []
        for item in data_list:
            if name in item:
                seg = np.load(os.path.join(data_path, item))
                segs.append(np.expand_dims(seg, axis=2))
        segs = np.concatenate(segs, axis=-1)
        segs = np.array(segs, dtype=np.int16)
        print(segs.shape)
        segs = segs.transpose([2, 1, 0])
        label_nii = sitk.GetImageFromArray(segs)
        sitk.WriteImage(label_nii, os.path.join(save_path, 'myops_test_{}_seg.nii.gz'.format(name)))
        # segs = nb.Nifti1Image(segs, np.eye(4))
        # print(os.path.join(save_path, 'myops_test_{}_seg.nii.gz'.format(name)))
        # nb.save(segs, os.path.join(save_path, 'myops_test_{}_seg.nii.gz'.format(name)))


def show_raw_data_shape():
    raw_data_path = r'D:\Learning\Datasets\MyoPS2020\test20'
    raw_data_list = [item for item in os.listdir(raw_data_path) if 'C0' in item]
    for path in raw_data_list:
        raw_data = nb.load(os.path.join(raw_data_path, path)).dataobj
        print(raw_data.shape)


if __name__ == '__main__':
    npy_to_nii('./results/test/npy', './results/test/nii')
    print('raw data------')
    # show_raw_data_shape()
