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
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

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
        
        n_feats = self._cfg.preprocessor.features
        self.conv_channels = self._cfg.conv_channels
        d_model = self._cfg.encoder.d_model
        self.scaling_factor = self._cfg.scaling_factor
        n_layers = int(math.log(self.scaling_factor, 2))
        
        subencoder = []
        in_channels = 1
        for _ in range(n_layers):
            subencoder.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.conv_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            in_channels = self.conv_channels
        subencoder.append(nn.Linear(in_features=n_feats//self.scaling_factor * self.conv_channels, out_features=d_model))
        self.subencoder = nn.ModuleList(subencoder)
        
        subdecoder = []
        subdecoder.append(nn.Linear(in_features=d_model, out_features=n_feats//self.scaling_factor * self.conv_channels))
        for ith in range(n_layers):
            out_channels = 1 if ith == n_layers - 1 else self.conv_channels
            subdecoder.append(
                nn.ConvTranspose2d(
                    in_channels=self.conv_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
        self.subdecoder = nn.ModuleList(subdecoder)
        
        self.skip_connections = [] 
        
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
    
    def process(self, signal, signal_len):
        clean_spec, clean_spec_len = self.preprocessor(input_signal=signal, length=signal_len)
        noisy_signal = self.noise_mixer(signal)
        noisy_spec, _ = self.preprocessor(input_signal=noisy_signal, length=signal_len)
        del signal, noisy_signal
        
        max_spec_len = max(clean_spec_len).item()
        max_spec_len = math.ceil(max_spec_len / self.scaling_factor) * self.scaling_factor
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
        
        return clean_spec, noisy_spec, clean_spec_len
               
    def calc_length(self, lengths):
        kernel_size=4
        stride=2
        padding=1
        for i in range(int(math.log(self.scaling_factor, 2))):
            lengths = torch.div(lengths.to(dtype=torch.float) + 2 * padding - kernel_size, stride) + 1
            lengths = torch.floor(lengths)
        return lengths.to(dtype=torch.int)
    
    def forward_subenc(self, noisy_spec):
        self.skip_connections.clear()
        for ith, layer in enumerate(self.subencoder):
            if ith == 0:
                noisy_spec = noisy_spec.unsqueeze(1)
            if ith == len(self.subencoder) - 1:
                b, c, f, t = noisy_spec.shape
                noisy_spec = noisy_spec.reshape(b, t, c * f)
            noisy_spec = nn.functional.relu(layer(noisy_spec))
            self.skip_connections.append(noisy_spec)
        noisy_spec = noisy_spec.transpose(1, 2)
        return noisy_spec
    
    def forward(self, noisy_spec, length):
        noisy_spec, _ = self.encoder(audio_signal=noisy_spec, length=length)
        noisy_spec = noisy_spec.transpose(1, 2)
        return noisy_spec
    
    def forward_subdec(self, noisy_spec):
        for ith, layer in enumerate(self.subencoder):
            if ith == 1:
                b, t, d = noisy_spec.shape
                noisy_spec = noisy_spec.reshape(b, self.conv_channels, d//self.conv_channels, t)
            noisy_spec = noisy_spec + self.skip_connections.pop(-1)
            if ith < len(self.subdecoder) - 1:
                noisy_spec = nn.functional.relu(layer(noisy_spec))
            else:
                noisy_spec = layer(noisy_spec)
        noisy_spec = noisy_spec.squeeze(0)
        return noisy_spec
    
    def forward_loss(self, denoised_spec, clean_spec, spec_len):
        for ith in range(len(denoised_spec)):
            denoised_spec[ith, :, spec_len[ith]:] = 0.0
        return torch.nn.functional.mse_loss(denoised_spec, clean_spec)
        

    def training_step(self, batch, batch_nb):
        signal, signal_len, _, _ = batch
        
        clean_spec, noisy_spec, spec_length = self.process(signal, signal_len)
        noisy_spec_ref = noisy_spec.clone()
        
        noisy_spec = self.forward_subenc(noisy_spec)
        noisy_spec = self.forward(noisy_spec=noisy_spec, length=self.calc_length(spec_length))
        noisy_spec = self.forward_subdec(noisy_spec)
        
        denoised_spec = noisy_spec + noisy_spec_ref
        
        loss_value = self.forward_loss(denoised_spec, clean_spec, spec_length)

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': self.trainer.global_step,
            'train_loss': loss_value,
        }
        self.log_dict(tensorboard_logs)

        return {'loss': loss_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        
        self.eval()
        
        signal, signal_len, _, _ = batch
        
        clean_spec, noisy_spec, spec_length = self.process(signal, signal_len)
        noisy_spec_ref = noisy_spec.clone()
        
        noisy_spec = self.forward_subenc(noisy_spec)
        noisy_spec = self.forward(noisy_spec=noisy_spec, length=self.calc_length(spec_length))
        noisy_spec = self.forward_subdec(noisy_spec)
        
        denoised_spec = noisy_spec + noisy_spec_ref
        
        loss_value = self.forward_loss(denoised_spec, clean_spec, spec_length)
        
        ssim_score = ssim(denoised_spec, clean_spec)
        psnr_score = psnr(denoised_spec, clean_spec)

        tensorboard_logs = {'val_loss': loss_value, 'ssim_score': ssim_score, 'psnr_score': psnr_score}
        self.log_dict(tensorboard_logs)
        
        self.train()
        
        return {
            'val_loss': loss_value, 
            'ssim_score': ssim_score, 
            'psnr_score': psnr_score, 
            'log': tensorboard_logs
        }

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        ssim_mean = torch.stack([x['ssim_score'] for x in outputs]).mean()
        psnr_mean = torch.stack([x['psnr_score'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_loss_mean, 'ssim_score': ssim_mean, 'psnr_score': psnr_mean}
        return {
            'val_loss': val_loss_mean, 
            'ssim_score': ssim_mean, 
            'psnr_score': psnr_mean, 
            'log': tensorboard_logs
        }
