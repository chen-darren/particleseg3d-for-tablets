from nnunet.training.network_training.nnUNetTrainerV2_ParticleSeg3D_DarrenAdam_CUDAErrorSkip import nnUNetTrainerV2_ParticleSeg3D_DarrenAdam_CUDAErrorSkip

class nnUNetTrainerV2_ParticleSeg3D_DarrenAdam_CUDAErrorSkip_FineTune(nnUNetTrainerV2_ParticleSeg3D_DarrenAdam_CUDAErrorSkip):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        # Revised max epochs and initial learning rate for fine tuning
        self.max_num_epochs = 500 # 1/2 the max number of epochs of nnUNetTrainerV2_ParticleSeg3D_DarrenAdam_CUDAErrorSkip
        self.inital_lr = 3e-6 # 1/10 the initial learnging rate of nnUNetTrainerV2_ParticleSeg3D_DarrenAdam_CUDAErrorSkip