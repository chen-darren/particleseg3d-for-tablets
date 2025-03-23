from nnunet.training.network_training.nnUNetTrainerV2_ParticleSeg3D import nnUNetTrainerV2_ParticleSeg3D
from batchgenerators.utilities.file_and_folder_operations import *

class nnUNetTrainerV2_ParticleSeg3D_DarrenSGD(nnUNetTrainerV2_ParticleSeg3D):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        task_name = dataset_directory.split("/")[-1]
        nnunet_datasets_dir = dataset_directory.split("/")[:-2]
        
        # Darren's new and revised variables to overwrite orignals
        self.raw_data_dir = join("/", *nnunet_datasets_dir, "nnUNet_raw_data_base", "nnUNet_raw_data", task_name)
        self.save_every = 1
        self.initial_lr = 1e-3 # Original SGD LR is 1e-2