import os
import os
import zipfile
import pdb
from glob import glob
import torch.nn.functional as F
import torch.optim
import traceback
from tqdm import tqdm
import multicrop
import numpy as np
import nibabel as nib
import models
from data.data_utils import get_all_coords, _shape
from utils import Parser


class Model:
    def __init__(self, log=print):
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # use_cuda = torch.cuda.is_available()
        # device = torch.device('cuda' if use_cuda else 'cpu')
        self.log = log
        self.log("Initializing Model...")
        cfg = "deepmedic_ce_50_50_fold0"
        ckpt = "model_last.tar"
        args = Parser(cfg, log="test")
        ckpts_folder = os.path.join("ckpts", cfg)
        args.saving = True
        args.scoring = True
        self.sample_size = args.sample_size
        self.sub_sample_size = args.sub_sample_size
        self.batch_size = args.batch_size
        dtype = torch.float32
        target_size = args.target_size
        self.modalities = ("flair", "t1ce", "t1", "t2")
        print("getting all coords")
        coords = get_all_coords(target_size)
        # np.save('files/coords.npy', coords.numpy())
        self.coords = coords.cuda(non_blocking=True)
        print("done")
        try:  # load a saved model, faster
            self.log("loading model..")
            model = torch.load(os.path.join(ckpts_folder, "model.pth"))
            self.log("loaded saved model..")
        except Exception as e:
            self.log(e)
            Network = getattr(models, args.net)
            model = Network(**args.net_params)
            self.log("loading ckpt %s" % cfg)
            model_file = os.path.join(ckpts_folder, ckpt)
            checkpoint = torch.load(
                model_file, map_location=lambda storage, loc: storage
            )
            state_dict = {
                k: v
                for k, v in checkpoint["state_dict"].items()
                if k in model.state_dict()
            }
            model.load_state_dict(state_dict)
            torch.save(model, os.path.join(ckpts_folder, "model.pth"))
        self.model = model.cuda()
        self.model.eval()
        self.HWT = [240, 240, 155]
        h, w, t = np.ceil(np.array(_shape, dtype="float32") / target_size).astype("int")
        self.hwt = [int(h), int(w), int(t)]
        self.outputs = torch.zeros(
            (5, h * w * t, target_size, target_size, target_size), dtype=dtype
        )
        self.log("Done")

    def nib_load(self, file_name):
        proxy = nib.load(file_name)
        data = proxy.get_data()
        proxy.uncache()
        return data

    def extract_zip(self, zipscanpath):
        # zipscanpath = 'files/scans/24/1.zip'
        scan_id = zipscanpath.split("/")[-1].split(".")[0]
        patient_id = zipscanpath.split("/")[-2]
        extraction_path = os.path.join("files/scans", patient_id, scan_id)
        # the scans will be extracted to a folder named <scan_id> in files/scans/patient_id/
        with zipfile.ZipFile(zipscanpath, "r") as zip_ref:
            zip_ref.extractall(extraction_path)  # footnote 1
        files = []
        for modal in self.modalities:
            files.append(glob(extraction_path + "/*%s.nii.gz" % modal)[0])
        # seg_file = files/segmentations/patient_id/scan_id.nii.gz
        seg_file_path = "files/segmentations/%s/%s.nii.gz" % (patient_id, scan_id)
        print(files)
        return files, seg_file_path

    def segment(self, zipscanpath):
        try:
            self.log("Starting segmentation")
            H, W, T = self.HWT
            h, w, t = self.hwt
            sample_size = self.sample_size
            sub_sample_size = self.sub_sample_size
            batch_size = self.batch_size
            coords = self.coords
            outputs = self.outputs
            self.log("Extracting zip")
            files, seg_file_path = self.extract_zip(zipscanpath)
            self.log("Preprocessing..")
            images = np.stack(
                [
                    np.array(self.nib_load(file), dtype="float32", order="C")
                    for file in files
                ],
                -1,
            )
            mask = images.sum(-1) > 0
            for k in range(4):
                x = images[..., k]
                y = x[mask]
                lower = np.percentile(y, 0.2)
                upper = np.percentile(y, 99.8)
                x[mask & (x < lower)] = lower
                x[mask & (x > upper)] = upper
                y = x[mask]
                x -= y.mean()
                x /= y.std()
                images[..., k] = x

            images = torch.from_numpy(images)
            images = images.permute(3, 0, 1, 2).contiguous()
            images = images.cuda(non_blocking=True)
            for b, coord in tqdm(enumerate(coords.split(batch_size))):
                x1 = multicrop.crop3d_gpu(
                    images, coord, sample_size, sample_size, sample_size, 1, True
                )
                x2 = multicrop.crop3d_gpu(
                    images,
                    coord,
                    sub_sample_size,
                    sub_sample_size,
                    sub_sample_size,
                    3,
                    True,
                )
                # pdb.set_trace()
                # 1x4x25x25x25 1x4x19x19x19
                logit = self.model((x1, x2))  # nx5x9x9x9, target nx9x9x9
                output = F.softmax(logit, dim=1)
                start = b * batch_size
                end = start + output.shape[0]
                outputs[:, start:end] = output.permute(1, 0, 2, 3, 4).data.cpu()
            self.log("Segmentation done, saving output..")
            outputs = outputs.view(5, h, w, t, 9, 9, 9).permute(0, 1, 4, 2, 5, 3, 6)
            outputs = outputs.reshape(5, h * 9, w * 9, t * 9)
            outputs = outputs[:, :H, :W, :T].numpy()

            preds = outputs.argmax(0).astype("uint8")
            img = nib.Nifti1Image(preds, None)
            nib.save(img, os.path.join(seg_file_path))
            self.log("Segmentation saved at %s" % seg_file_path)
        except Exception as e:
            traceback.print_exc()
            pdb.set_trace()


if __name__ == "__main__":
    model = Model()
    model.segment("test/scans/Brats18_CBICA_AWG_1_t2.nii.gz")


# footnote 1: extractall extracts the zip files in a folder named same as the zipfile iteself, in the given extraction_path location.
# in this case, zipfile name is same as extraction_path folder (last one, in the path) so, it extracts there (scan_id folder) only
