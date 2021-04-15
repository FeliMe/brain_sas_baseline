import argparse
from glob import glob
import os
import shutil
import zipfile

import nibabel as nib
import numpy as np
from tqdm import tqdm

from utils.data_utils import DATAROOT
from utils.registrator import MRIRegistrator
from utils.robex import strip_skull_ROBEX


class BraTSHandler():
    def __init__(self, args):
        """ipp.cbica.upenn.edu 2020 version"""
        if args.register:
            self.registerBraTS(args)
        else:
            self.prepare_BraTS(args)

    def prepare_BraTS(self, args):
        if args.dataset_path is None:
            args.dataset_path = os.path.join(DATAROOT, 'BraTS')

        if not os.path.exists(os.path.join(args.dataset_path, 'MICCAI_BraTS2020_TrainingData.zip')):
            raise RuntimeError(f"Apply for the BraTS2020 data at ipp.cbica.upenn.edu"
                               f" and download it to {args.dataset_path}")

        self.unzip_BraTS(
            dataset_path=args.dataset_path,
            force=False
        )
        self.rename_lesions(args)

    @staticmethod
    def unzip_BraTS(dataset_path, force=False):
        train_zip = os.path.join(
            dataset_path, 'MICCAI_BraTS2020_TrainingData.zip')
        val_zip = os.path.join(
            dataset_path, 'MICCAI_BraTS2020_ValidationData.zip')

        train_dir = os.path.join(dataset_path, 'MICCAI_BraTS2020_TrainingData')
        val_dir = os.path.join(dataset_path, 'MICCAI_BraTS2020_ValidationData')

        # Remove target directories if force
        if force:
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(val_dir, ignore_errors=True)

        # Extract zip
        print(f"Extracting {train_zip}")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print(f"Extracting {val_zip}")
        with zipfile.ZipFile(val_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)

    @staticmethod
    def rename_lesions(args):
        print("Renaming segmentation files in BraTS to "
              "'anomaly_segmentation_unregistered.nii.gz'")
        lesion_files = glob(f"{args.dataset_path}/*/*/*_seg.nii.gz")
        target_files = [
            '/'.join(f.split('/')[:-1] + ['anomaly_segmentation_unregistered.nii.gz']) for f in lesion_files]
        for lesion, target in zip(lesion_files, target_files):
            data = nib.load(lesion, keep_file_open=False)
            volume = data.get_fdata(caching='unchanged',
                                    dtype=np.float32).astype(np.dtype("short"))
            nib.save(nib.Nifti1Image(volume, data.affine), target)
            # shutil.copy(lesion, target)

    @staticmethod
    def registerBraTS(args):
        if args.dataset_path is None:
            args.dataset_path = os.path.join(DATAROOT, 'BraTS')

        print("Registering BraTS")

        # Get all files
        files = glob(
            f"{args.dataset_path}/MICCAI_BraTS2020_TrainingData/*/*_t1.nii.gz")
        print(f"Found {len(files)} files.")

        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator
        # template_path = os.path.join(
        #     DATAROOT, f'BrainAtlases/mni_icbm152_nlin_sym_09a/{w.lower()}_stripped.nii')
        template_path = os.path.join(DATAROOT, 'BrainAtlases/sri24_spm8/templates/T1_brain.nii')
        # registrator = SitkRegistrator(template_path)
        registrator = MRIRegistrator(template_path=template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path [:path.rfind("t1")]
            folder = '/'.join(path.split('/')[:-1])
            # Transform T2 image
            path = base + "t2.nii.gz"
            save_path = base + "t2_registered.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )
            # Transform FLAIR image
            path = base + "flair.nii.gz"
            save_path = base + "flair_registered.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )
            # Transform segmentation
            path = os.path.join(
                folder, "anomaly_segmentation_unregistered.nii.gz")
            save_path = os.path.join(
                folder, "anomaly_segmentation.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype='short'
            )


class MSLUB_Handler():
    def __init__(self, args):
        """
        3D MR image database of Multiple Sclerosis patients with white matter lesion segmentations
        http://lit.fe.uni-lj.si/tools.php?lang=eng
        """
        if args.dataset_path is None:
            args.dataset_path = os.path.join(DATAROOT, 'MSLUB')

        if args.register:
            self.register_MSLUB(args)
        elif args.skull_strip:
            print(f"Skull stripping for MSLUB {args.weighting} scans")
            self.skull_strip_MSLUB(args)
        else:
            print("Download the dataset from https://github.com/muschellij2/open_ms_data"
                  " and put the content of the folder cross_sectional/coregistered_resampled/"
                  " in your $DATAROOT/MSLUB/normal/")

    @staticmethod
    def skull_strip_MSLUB(args):
        w = args.weighting
        if not isinstance(w, str):
            raise RuntimeError(f"Invalid value for --weighting {w}")
        # Get list of all files
        if w != 'FLAIR':
            w += 'W'
        paths = glob(
            f"{os.path.join(args.dataset_path, 'normal')}/*/*{w.upper()}.nii.gz")

        for p in tqdm(paths):
            mask_path = os.path.join(
                '/'.join(p.split('/')[:-1]), "brainmask.nii.gz")

            stripped_path = p.split('.')[0] + "_stripped.nii.gz"

            # Load scan
            data = nib.load(p, keep_file_open=False)
            volume = data.get_fdata(caching='unchanged', dtype=np.float32)
            affine = data.affine
            # Load brain_mask
            brain_mask = nib.load(mask_path, keep_file_open=False).get_fdata(
                caching='unchanged', dtype=np.float32)

            volume = volume * brain_mask

            # Save volume
            nib.save(nib.Nifti1Image(volume, affine), stripped_path)

    @staticmethod
    def register_MSLUB(args):

        print("Registering MSLUB")

        # Get all files
        files = glob(f"{args.dataset_path}/lesion/*/T1W_stripped.nii.gz")
        print(f"Found {len(files)} files.")

        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize the registrator
        # template_path = os.path.join(
        #     DATAROOT, f'BrainAtlases/mni_icbm152_nlin_sym_09a/t1_stripped.nii')
        template_path = os.path.join(DATAROOT, 'BrainAtlases/sri24_spm8/templates/T1_brain.nii')

        # registrator = SitkRegistrator(template_path)
        registrator = MRIRegistrator(template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            folder = '/'.join(path.split('/')[:-1])
            # Transform T2 image
            path = os.path.join(folder, "T2W_stripped.nii.gz")
            save_path = os.path.join(
                folder, "T2W_stripped_registered.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short"
            )
            # Transform FLAIR image
            path = os.path.join(folder, "FLAIR_stripped.nii.gz")
            save_path = os.path.join(
                folder, "FLAIR_stripped_registered.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short"
            )
            # Transform segmentation
            path = os.path.join(
                folder, "anomaly_segmentation_unregistered.nii.gz")
            save_path = os.path.join(
                folder, "anomaly_segmentation.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short"
            )


class WMHHandler():
    def __init__(self, args):
        """ Download data from https://wmh.isi.uu.nl/data/
        Your args.dataset_path should contain the following elements now:
        Amsterdam_GE3T.zip
        Singapore.zip
        Utrecht.zip
        """
        if args.dataset_path is None:
            args.dataset_path = os.path.join(DATAROOT, 'WMH')

        if args.skull_strip:
            self.skull_stripWMH(args)
        elif args.register:
            self.register_WMH(args)
        else:
            self.prepare_WMH(args)

    @staticmethod
    def prepare_WMH(args):
        """Puts all the files in the correct folder and renames them correctly"""

        # Unzip files
        zip_files = ['Amsterdam_GE3T.zip', 'Singapore.zip', 'Utrecht.zip']
        zip_files = [os.path.join(args.dataset_path, z) for z in zip_files]

        for zip_file in zip_files:
            # If file is missing, throw an error
            if not os.path.exists(zip_file):
                raise RuntimeError(f"No file found at {zip_file}")

            # Extract zip
            print(f"Extracting {zip_file}")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(args.dataset_path)

        # Rename anomaly segmentation files
        seg_files = glob(f"{args.dataset_path}/*/*/wmh.nii.gz")
        for seg_file in seg_files:
            target = seg_file[:seg_file.rfind("wmh.nii.gz")] + 'orig/anomaly_segmentation_unregistered.nii.gz'
            shutil.copy(seg_file, target)

    @staticmethod
    def skull_stripWMH(args):
        # Get list of all files
        files = glob(f"{args.dataset_path}/*/*/orig/T1.nii.gz")
        print(f"Found {len(files)} files.")

        # Run ROBEX
        strip_skull_ROBEX(files)

        files_stripped = glob(f"{args.dataset_path}/*/*/orig/T1_stripped.nii.gz")
        print(f"Found {len(files_stripped)} stripped files.")
        for fi in files_stripped:
            # Strip FLAIR based on T1
            folder = '/'.join(fi.split('/')[:-1])
            f_flair = os.path.join(folder, "FLAIR.nii.gz")
            f_flair_stripped = os.path.join(folder, "FLAIR_stripped.nii.gz")

            # Load files
            data = nib.load(f_flair, keep_file_open=False)
            flair = data.get_fdata(caching='unchanged', dtype=np.float32).astype(np.short)
            t1 = nib.load(fi, keep_file_open=False).get_fdata(
                          caching='unchanged', dtype=np.float32).astype(np.short)

            # Strip FLAIR
            flair_stripped = flair * np.where(t1 > 0, 1, 0)

            # Save stripped FLAIR image
            nib.save(nib.Nifti1Image(flair_stripped.astype(np.short), data.affine), f_flair_stripped)


    @staticmethod
    def register_WMH(args):
        print("Registering WMH")

        # Get all files
        files = glob(f"{args.dataset_path}/*/*/orig/T1_stripped.nii.gz")
        print(f"Found {len(files)} files.")

        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator
        # template_path = os.path.join(DATAROOT, 'BrainAtlases/mni_icbm152_nlin_sym_09a/t1_stripped.nii')
        template_path = os.path.join(DATAROOT, 'BrainAtlases/sri24_spm8/templates/T1_brain.nii')
        # registrator = SitkRegistrator(template_path)
        registrator = MRIRegistrator(template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path [:path.rfind("T1")]
            folder = '/'.join(path.split('/')[:-1])
            # Transform FLAIR image
            path = base + "FLAIR_stripped.nii.gz"
            save_path = base + "FLAIR_stripped_registered.nii.gz"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short"
            )
            # Transform segmentation
            path = os.path.join(
                folder, "anomaly_segmentation_unregistered.nii.gz")
            save_path = os.path.join(
                folder, "anomaly_segmentation.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short"
            )


class MSSEGHandler():
    def __init__(self, args):
        """ Download data from https://wmh.isi.uu.nl/data/
        Your args.dataset_path should contain the following elements now:
        Amsterdam_GE3T.zip
        Singapore.zip
        Utrecht.zip
        """
        if args.dataset_path is None:
            args.dataset_path = os.path.join(DATAROOT, 'MSSEG2015')

        if args.register:
            self.register_MSSEG(args)
        else:
            self.prepare_MSSEG(args)

    @staticmethod
    def prepare_MSSEG(args):
        """Puts all the files in the correct folder and renames them correctly"""

        data_path = os.path.join(args.dataset_path, 'training')
        if not os.path.exists(data_path):
            print("Download MSSEG2015 data from https://smart-stats-tools.org/lesion-challenge-2015"
                  f" and put it into {args.dataset_path}")

        # Put every scan of a patient intoa separate folder
        patients = glob(f"{data_path}/training*")
        for patient in patients:
            patient_name = patient.split('/')[-1]
            scans = glob(f"{patient}/preprocessed/{patient_name}*")
            scans = set([s.split('/')[-1][:len(patient_name) + 3] for s in scans])
            for scan in scans:
                folder = os.path.join(patient, scan)
                os.makedirs(folder, exist_ok=True)
                # Move all scans to folder
                imgs = glob(os.path.join(patient, "preprocessed", f"{scan}*"))
                for img in imgs:
                    name = img[img.rfind(scan) + len(scan) + 1:]
                    target = os.path.join(folder, name)
                    shutil.move(img, target)
                    # print(f"mv {img}, {target}")

                masks = glob(os.path.join(patient, "masks", f"{scan}*"))
                for mask in masks:
                    name = mask[mask.rfind(scan) + len(scan) + 1:]
                    target = os.path.join(folder, name)
                    shutil.move(mask, target)
                    # print(f"mv {mask}, {target}")

            shutil.rmtree(os.path.join(patient, "preprocessed"))
            shutil.rmtree(os.path.join(patient, "masks"))
            # print(f"rm -r {os.path.join(patient, 'preprocessed')}")
            # print(f"rm -r {os.path.join(patient, 'masks')}")


    @staticmethod
    def register_MSSEG(args):
        print("Registering MSSEG 2015")

        # Get all files
        files = glob(f"{args.dataset_path}/training/*/*/mprage_pp.nii")
        print(f"Found {len(files)} files.")

        if len(files) == 0:
            raise RuntimeError("0 files to be registered")

        # Initialize registrator
        # template_path = os.path.join(DATAROOT, 'BrainAtlases/mni_icbm152_nlin_sym_09a/t1_stripped.nii')
        template_path = os.path.join(DATAROOT, 'BrainAtlases/sri24_spm8/templates/T1_brain.nii')
        # registrator = SitkRegistrator(template_path)
        registrator = MRIRegistrator(template_path)

        # Register files
        transformations = registrator.register_batch(files)

        for path, t in tqdm(transformations.items()):
            base = path [:path.rfind("mprage")]
            folder = '/'.join(path.split('/')[:-1])
            # Transform FLAIR image
            path = base + "flair_pp.nii"
            save_path = base + "flair_pp_registered.nii"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="float32"
            )
            # Transform T2 image
            path = base + "t2_pp.nii"
            save_path = base + "t2_pp_registered.nii"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="float32"
            )
            # Transform PD image
            path = base + "pd_pp.nii"
            save_path = base + "pd_pp_registered.nii"
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="float32"
            )
            # Transform segmentation
            path = os.path.join(
                folder, "mask1.nii")
            save_path = os.path.join(
                folder, "anomaly_segmentation.nii.gz")
            registrator.transform(
                img=path,
                save_path=save_path,
                transformation=t,
                affine=registrator.template_affine,
                dtype="short"
            )


def download_data(args):
    if args.dataset == 'BraTS':
        BraTSHandler(args)
    elif args.dataset == 'MSLUB':
        MSLUB_Handler(args)
    elif args.dataset == 'WMH':
        WMHHandler(args)
    elif args.dataset == 'MSSEG2015':
        MSSEGHandler(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['BraTS', 'MSLUB', 'WMH', 'MSSEG2015'])
    parser.add_argument('--dataset_path', default=None)
    # Preprocessing arguments
    parser.add_argument('--weighting', type=str,
                        choices=['t1', 't2', 'T1', 'T2', 'FLAIR'])
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--skull_strip', action='store_true')
    args = parser.parse_args()

    # Add this to handle ~ in path variables
    if args.dataset_path:
        args.dataset_path = os.path.expanduser(args.dataset_path)

    download_data(args)
