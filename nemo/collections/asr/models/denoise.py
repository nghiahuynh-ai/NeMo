# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import ceil
import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from einops.layers.torch import Rearrange
from nemo.collections.asr.parts.submodules.noise_mixing import NoiseMixer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.parts.mixins import ASRModuleMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.mixins import AccessMixin, set_access_cfg
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    AudioSignal,
    LabelsType,
    LengthsType,
    NeuralType,
    SpectrogramType,
)
from nemo.utils import logging

__all__ = ['Denoising']


class Denoising(ModelPT, ASRModuleMixin, AccessMixin):
    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        return []

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Get global rank and total number of GPU workers for IterableDataset partitioning, if applicable
        # Global_rank and local_rank is set by LightningModule in Lightning 1.2.0
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.world_size

        super().__init__(cfg=cfg, trainer=trainer)
        
        self.preprocessor = Denoising.from_config_dict(self._cfg.preprocessor)
        self.encoder = Denoising.from_config_dict(self._cfg.encoder)
        
        patch_size = self._cfg.patch_size
        self.patch_size = patch_size
        n_feats = self._cfg.preprocessor.features
        
        self.patchifier = Rearrange('b (h p1) (w p2) -> b (h w) (p1 p2)', h=n_feats, p1=patch_size, p2=patch_size)
        
        self.unpatchifier = Rearrange('b (h w) (p1 p2) -> b (h p1) (w p2)', h=n_feats, p1=patch_size, p2=patch_size)
        
        self.noise_mixer = NoiseMixer(
            real_noise_filepath=self._cfg.real_noise.filepath,
            real_noise_snr=self._cfg.real_noise.snr,
            white_noise_mean=self._cfg.white_noise.mean,
            white_noise_std=self._cfg.white_noise.std,
        )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        # Automatically inject args from model config to dataloader config
        audio_to_text_dataset.inject_dataloader_value_from_model_config(self.cfg, config, key='sample_rate')

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_char_dataset(
                config=config,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_char_dataset(config=config, augmentor=augmentor)

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def setup_training_data(self, train_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the training data loader via a Dict-like object.

        Args:
            train_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in train_data_config:
            train_data_config['shuffle'] = True

        # preserve config
        self._update_dataset_config(dataset_name='train', config=train_data_config)

        self._train_dl = self._setup_dataloader_from_config(config=train_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in train_data_config and train_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_train_batches, float):
                self._trainer.limit_train_batches = int(
                    self._trainer.limit_train_batches
                    * ceil((len(self._train_dl.dataset) / self.world_size) / train_data_config['batch_size'])
                )

    def setup_validation_data(self, val_data_config: Optional[Union[DictConfig, Dict]]):
        """
        Sets up the validation data loader via a Dict-like object.

        Args:
            val_data_config: A config that contains the information regarding construction
                of an ASR Training dataset.

        Supported Datasets:
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.AudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToCharDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text.TarredAudioToBPEDataset`
            -   :class:`~nemo.collections.asr.data.audio_to_text_dali.AudioToCharDALIDataset`
        """
        if 'shuffle' not in val_data_config:
            val_data_config['shuffle'] = False

        # preserve config
        self._update_dataset_config(dataset_name='validation', config=val_data_config)

        self._validation_dl = self._setup_dataloader_from_config(config=val_data_config)

        # Need to set this because if using an IterableDataset, the length of the dataloader is the total number
        # of samples rather than the number of batches, and this messes up the tqdm progress bar.
        # So we set the number of steps manually (to the correct number) to fix this.
        if 'is_tarred' in val_data_config and val_data_config['is_tarred']:
            # We also need to check if limit_train_batches is already set.
            # If it's an int, we assume that the user has set it to something sane, i.e. <= # training batches,
            # and don't change it. Otherwise, adjust batches accordingly if it's a float (including 1.0).
            if isinstance(self._trainer.limit_val_batches, float):
                self._trainer.limit_val_batches = int(
                    self._trainer.limit_val_batches
                    * ceil((len(self._validation_dl.dataset) / self.world_size) / val_data_config['batch_size'])
                )

    def forward(self, input_patch, input_patch_length):
        return self.encoder(audio_signal=input_patch, length=input_patch_length)

    def training_step(self, batch, batch_nb):
        signal, signal_len, _, _ = batch
        
        clean_spec, clean_spec_len = self.preprocessor(input_signal=signal, length=signal_len)
        noisy_signal = self.noise_mixer(signal)
        noisy_spec, _ = self.preprocessor(input_signal=noisy_signal, length=signal_len)
        del signal
        
        max_spec_len = max(clean_spec_len).item()
        max_spec_len = math.ceil(max_spec_len / self.patch_size) * self.patch_size
        padding_clean_spec, padding_noisy_spec = [], []
        for ith in range(len(clean_spec)):
            pad = (0, max_spec_len - clean_spec[ith].size(1))
            clean_spec_i = torch.nn.functional.pad(clean_spec[ith], pad, value=0.0)
            padding_clean_spec.append(clean_spec_i)
            noisy_spec_i = torch.nn.functional.pad(noisy_spec[ith], pad, value=0.0)
            padding_noisy_spec.append(noisy_spec_i)
        clean_spec = torch.stack(padding_clean_spec)
        noisy_spec = torch.stack(padding_noisy_spec)
        del padding_clean_spec, padding_noisy_spec
        
        patch = self.patchifier(noisy_spec)
        
        patch, _ = self.forward(
            input_patch=patch, input_patch_length=patch.size(2),
        )
        
        denoised_spec = self.unpatchifier(patch)
        
        for ith in range(len(denoised_spec)):
            denoised_spec[ith, :,clean_spec_len[ith]:] = 0.0
            
        loss_value = torch.nn.functional.mse_loss(clean_spec, denoised_spec)

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
            'train_loss': loss_value,
        }

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, _, _ = batch
        
        clean_spec, clean_spec_len = self.preprocessor(input_signal=signal, length=signal_len)
        noisy_signal = self.noise_mixer(signal)
        noisy_spec, _ = self.preprocessor(input_signal=noisy_signal, length=signal_len)
        del signal
        
        max_spec_len = max(clean_spec_len).item()
        max_spec_len = math.ceil(max_spec_len / self.patch_size) * self.patch_size
        padding_clean_spec, padding_noisy_spec = [], []
        for ith in range(len(clean_spec)):
            pad = (0, max_spec_len - clean_spec[ith].size(1))
            clean_spec_i = torch.nn.functional.pad(clean_spec[ith], pad, value=0.0)
            padding_clean_spec.append(clean_spec_i)
            noisy_spec_i = torch.nn.functional.pad(noisy_spec[ith], pad, value=0.0)
            padding_noisy_spec.append(noisy_spec_i)
        clean_spec = torch.stack(padding_clean_spec)
        noisy_spec = torch.stack(padding_noisy_spec)
        del padding_clean_spec, padding_noisy_spec
        
        patch = self.patchifier(noisy_spec)
        
        patch, _ = self.forward(
            input_patch=patch, input_patch_length=patch.size(2),
        )
        
        denoised_spec = self.unpatchifier(patch)
        
        for ith in range(len(denoised_spec)):
            denoised_spec[ith, :,clean_spec_len[ith]:] = 0.0
            
        loss_value = torch.nn.functional.mse_loss(clean_spec, denoised_spec)

        return {
            'val_loss': loss_value,
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_loss_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}
