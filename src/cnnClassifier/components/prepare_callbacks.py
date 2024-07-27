import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig
from pathlib import Path


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config
    
    
    
    @property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = Path(self.config.tensorboard_root_log_dir) / f"tb_logs_at_{timestamp}"
        return tf.keras.callbacks.TensorBoard(log_dir=str(tb_running_log_dir))
    
    
    
    @property
    def _create_ckpt_callbacks(self):
        # Ensure the file path ends with .weights.h5
        checkpoint_filepath = self.config.checkpoint_model_filepath
        checkpoint_filepath = str(checkpoint_filepath)  # Convert Path object to string
        if not checkpoint_filepath.endswith('.weights.h5'):
            checkpoint_filepath = checkpoint_filepath.rsplit('.', 1)[0] + '.weights.h5'
        
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_best_only=True,
            save_weights_only=True  # Save weights only to ensure the format is .weights.h5
        )
    
    
    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks
        ]
