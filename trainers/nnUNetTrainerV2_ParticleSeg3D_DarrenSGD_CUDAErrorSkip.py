from nnunet.training.network_training.nnUNetTrainerV2_ParticleSeg3D_DarrenSGD import nnUNetTrainerV2_ParticleSeg3D_DarrenSGD
import torch
from torch.cuda.amp import autocast
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda

class nnUNetTrainerV2_ParticleSeg3D_DarrenSGD_CUDAErrorSkip(nnUNetTrainerV2_ParticleSeg3D_DarrenSGD):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        
    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                
                try:
                    self.amp_grad_scaler.step(self.optimizer)  # Attempt step
                except RuntimeError as e:
                    if "CUDA error" in str(e):
                        print("CUDA error in optimizer step. Skipping this step but keeping gradients.")
                        # Don't zero the gradients, just skip this step
                    else:
                        raise  # Raise other errors
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()