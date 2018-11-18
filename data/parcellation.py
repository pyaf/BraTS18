import nibabel as nib
import os
import numpy as np

def nib_load(file_name):
    proxy = nib.load(file_name)
    data = proxy.get_data()
    #print('thuyen', data.dtype)
    #data = data.astype('float32')
    proxy.uncache()
    return data


def parcel(root, mask_dir, flist):
    flist = open(os.path.join(root, flist)).read().splitlines()
    names = [x.split('/')[-1] for x in flist]

    for k, name in enumerate(names):
        oname = os.path.join(root, flist[k], name + '_' + suffix + '.npy')
        iname = os.path.join(mask_dir, name + '_' + suffix + '.nii.gz')

        img = np.array(nib_load(iname), dtype='uint8', order='C')

        np.save(oname, img)


suffix = 'HarvardOxford-sub'

mask_dir = '/home/eee/ug/15084015/GE/BraTS2018-tumor-segmentation/BrainParcellation/HarvardOxford-sub/training'
root = '/home/eee/ug/15084015/GE/BraTS2018/MICCAI_BraTS_2018_Data_Training/'
flist = 'all.txt'

parcel(root, mask_dir, flist)

mask_dir = '/usr/data/pkao/brats2018/BrainParcellation/HarvardOxford-sub/validation/'
root = '/home/eee/ug/15084015/GE/BraTS2018/MICCAI_BraTS_2018_Data_Validation/'
flist = 'test.txt'

parcel(root, mask_dir, flist)

# mask_dir = '/home/eee/ug/15084015/GE/BraTS2018-tumor-segmentation/BrainParcellation/HarvardOxford-sub/testing/'
# root = '/usr/data/pkao/brats2018/testing'
# flist = 'test.txt'
