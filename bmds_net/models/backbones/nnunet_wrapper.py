"""
nnU-Net Wrapper Module.

nnU-Net operates as a standalone framework with its own preprocessing and inference pipeline.
This wrapper provides a Python interface to interact with nnU-Net v2 commands, 
facilitating comparison experiments within the BMDS-Net project structure.
"""

import os
import subprocess
import json
from typing import List, Optional

class NNUNetWrapper:
    """
    Wrapper for nnU-Net v2 commands.
    Requires 'nnUNetv2_predict' and other commands to be available in the system path.
    """
    
    def __init__(self, 
                 dataset_id: int = 137, 
                 config: str = '3d_fullres', 
                 trainer: str = 'nnUNetTrainer',
                 plans: str = 'nnUNetPlans'):
        """
        Args:
            dataset_id (int): BraTS Dataset ID (e.g., 137).
            config (str): nnU-Net configuration (usually '3d_fullres').
            trainer (str): Trainer class name.
            plans (str): Plans identifier.
        """
        self.dataset_id = dataset_id
        self.config = config
        self.trainer = trainer
        self.plans = plans
        
        self._check_env()

    def _check_env(self):
        """Check if necessary environment variables are set."""
        required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
        missing = [v for v in required_vars if v not in os.environ]
        if missing:
            print(f"[Warning] nnU-Net environment variables missing: {missing}. "
                  "Inference commands might fail if not set globally.")

    def predict(self, 
                input_folder: str, 
                output_folder: str, 
                checkpoint_name: str = 'checkpoint_best.pth',
                folds: Optional[List[int]] = None,
                save_probabilities: bool = False):
        """
        Run 'nnUNetv2_predict'.
        
        Args:
            input_folder (str): Path to input NIfTI files.
            output_folder (str): Path to save predictions.
            checkpoint_name (str): Checkpoint file name.
            folds (List[int]): List of folds to use (default: [0]).
            save_probabilities (bool): Whether to save softmax maps.
        """
        if folds is None:
            folds = [0]
            
        cmd = [
            'nnUNetv2_predict',
            '-i', input_folder,
            '-o', output_folder,
            '-d', str(self.dataset_id),
            '-c', self.config,
            '-f', *[str(f) for f in folds],
            '-tr', self.trainer,
            '-p', self.plans,
            '-chk', checkpoint_name
        ]
        
        if save_probabilities:
            cmd.append('--save_probabilities')
            
        print(f"Running nnU-Net Inference: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print("[Success] nnU-Net inference completed.")
        except subprocess.CalledProcessError as e:
            print(f"[Error] nnU-Net inference failed: {e}")
            raise

    def convert_dataset_to_nnunet(self, src_dir: str, dst_raw_dir: str):
        """
        Helper to convert standard BraTS structure to nnU-Net raw format.
        (Implementation depends on specific file naming conventions).
        """
        raise NotImplementedError("Data conversion logic needs to be customized for specific BraTS versions.")
