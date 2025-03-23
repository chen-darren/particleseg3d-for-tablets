from nnunet.training.network_training.nnUNetTrainerV2_ParticleSeg3D import nnUNetTrainerV2_ParticleSeg3D
from batchgenerators.utilities.file_and_folder_operations import *
from torch.optim import lr_scheduler
import torch

class nnUNetTrainerV2_ParticleSeg3D_DarrenAdam(nnUNetTrainerV2_ParticleSeg3D):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
        task_name = dataset_directory.split("/")[-1]
        nnunet_datasets_dir = dataset_directory.split("/")[:-2]
        
        # Darren's new and revised variables to overwrite orignals
        self.raw_data_dir = join("/", *nnunet_datasets_dir, "nnUNet_raw_data_base", "nnUNet_raw_data", task_name)
        self.save_every = 1
        self.initial_lr = 3e-5 # Original Adam LR is 3e-4
    
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode="abs")
        
    def maybe_update_lr(self, epochs=None):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))