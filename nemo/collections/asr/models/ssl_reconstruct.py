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

__all__ = ['ReconstructSSL']


class ReconstructSSL(ModelPT, ASRModuleMixin, AccessMixin):
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
        
        self.preprocessor = ReconstructSSL.from_config_dict(self._cfg.preprocessor)
        self.spec_augmentation = ReconstructSSL.from_config_dict(self.cfg.spec_augment)
        self.encoder = ReconstructSSL.from_config_dict(self._cfg.encoder)
        
        self.mask_ratio = self._cfg.mask_ratio
        self.scaling_factor = self._cfg.scaling_factor
        conv_channels = self._cfg.conv_channels
        n_resblocks = self._cfg.n_resblocks
        dim_in = self._cfg.dim_in
        dim_out = self._cfg.dim_out
        
        self.subencoder = SubEncoder(self.scaling_factor, conv_channels, n_resblocks, dim_in, dim_out)
        self.subdecoder = SubDecoder(self.scaling_factor, conv_channels, n_resblocks, dim_in, dim_out)

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

    def forward(self, input_spec, input_spec_length):
        return self.encoder(audio_signal=input_spec, length=input_spec_length)

    def training_step(self, batch, batch_nb):
        signal, signal_len, _, _ = batch
        
        spec_orig, spec_orig_len = self.preprocessor(input_signal=signal, length=signal_len)
        spec_masked = spec_orig.clone().detach()
        del signal
        
        max_spec_len = max(spec_orig_len).item()
        max_spec_len = math.ceil(max_spec_len / self.scaling_factor) * self.scaling_factor
        padding_spec_orig, padding_spec_masked = [], []
        for ith in range(len(spec_orig)):
            pad = (0, max_spec_len - spec_orig[ith].size(1))
            spec_orig_i = torch.nn.functional.pad(spec_orig[ith], pad, value=0.0)
            padding_spec_orig.append(spec_orig_i)
            spec_masked_i = torch.nn.functional.pad(spec_masked[ith], pad, value=0.0)
            padding_spec_masked.append(spec_masked_i)
        spec_orig = torch.stack(padding_spec_orig)
        spec_masked = torch.stack(padding_spec_masked)
        del padding_spec_orig, padding_spec_masked
        
        spec_masked = self.spec_augmentation(input_spec=spec_masked, length=spec_orig_len)
        encoded, _ = self.forward(input_spec=spec_masked, input_spec_length=spec_orig_len)
        encoded = encoded.transpose(1, 2)
        
        spec_reconstruct = self.subdecoder(encoded, self.subencoder.enc_out)
        spec_reconstruct = spec_reconstruct.transpose(1, 2)
        
        for ith in range(len(spec_reconstruct)):
            spec_reconstruct[ith, :,spec_orig_len[ith]:] = 0.0
            
        loss_value = torch.nn.functional.mse_loss(spec_reconstruct, spec_orig)

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
        
        spec_orig, spec_orig_len = self.preprocessor(input_signal=signal, length=signal_len)
        spec_masked = spec_orig.clone().detach()
        del signal
        
        max_spec_len = max(spec_orig_len).item()
        max_spec_len = math.ceil(max_spec_len / self.scaling_factor) * self.scaling_factor
        padding_spec_orig, padding_spec_masked = [], []
        for ith in range(len(spec_orig)):
            pad = (0, max_spec_len - spec_orig[ith].size(1))
            spec_orig_i = torch.nn.functional.pad(spec_orig[ith], pad, value=0.0)
            padding_spec_orig.append(spec_orig_i)
            spec_masked_i = torch.nn.functional.pad(spec_masked[ith], pad, value=0.0)
            padding_spec_masked.append(spec_masked_i)
        spec_orig = torch.stack(padding_spec_orig)
        spec_masked = torch.stack(padding_spec_masked)
        del padding_spec_orig, padding_spec_masked
        
        spec_masked = self.spec_augmentation(input_spec=spec_masked, length=spec_orig_len)
        spec_masked = self.subdecoder(spec_masked, self.subencoder.enc_out)
        
        encoded, _ = self.forward(input_spec=spec_masked, input_spec_length=spec_orig_len)
        encoded = encoded.transpose(1, 2)
        
        spec_reconstruct = self.subdecoder(encoded)
        spec_reconstruct = spec_reconstruct.transpose(1, 2)
        
        for ith in range(len(spec_reconstruct)):
            spec_reconstruct[ith, :,spec_orig_len[ith]:] = 0.0
            
        loss_value = torch.nn.functional.mse_loss(spec_reconstruct, spec_orig)
            
        loss_value = torch.nn.functional.mse_loss(spec_reconstruct, spec_orig)
        ssim_score = ssim(spec_reconstruct.unsqueeze(1), spec_orig.unsqueeze(1))
        psnr_score = psnr(spec_reconstruct, spec_orig)

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


class SubEncoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, n_resblocks, dim_in, dim_out):
        super().__init__()
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        in_channels = 1
        for _ in range(n_layers):
            self.layers.append(
                SubEncoderLayer(in_channels=in_channels, out_channels=conv_channels, n_resblocks=n_resblocks)
            )
            in_channels = conv_channels
        self.enc_out = []
        
        self.proj_out = nn.Linear(int(dim_in / scaling_factor) * conv_channels, dim_out)
            
    def forward(self, x):
        # x: (b, t, d)
        
        self.enc_out.clear()
        x = x.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x)
            self.enc_out = [x] + self.enc_out
        
        b, c, t, d = x.shape
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.proj_out(x)
        
        return x
        
    
class SubDecoder(nn.Module):
    def __init__(self, scaling_factor, conv_channels, n_resblocks, dim_in, dim_out):
        super().__init__()
        
        self.conv_channels = conv_channels
        self.proj_in = nn.Linear(dim_in, conv_channels * dim_out // scaling_factor)
        
        self.layers = nn.ModuleList()
        n_layers = int(math.log(scaling_factor, 2))
        for ith in range(n_layers):
            out_channels = 1 if ith == n_layers - 1 else conv_channels
            self.layers.append(
                SubDecoderLayer(in_channels=conv_channels, out_channels=out_channels, n_resblocks=n_resblocks)
            )
            
    def forward(self, x, enc_out):
        # x: (b, t, d)

        x = self.proj_in(x)
        b, t, d = x.shape
        x = x.reshape(b, self.conv_channels, t, d // self.conv_channels)
        
        for ith, layer in enumerate(self.layers):
            x = x + enc_out[ith]
            x = layer(x)

        x = x.squeeze(1)
        x = x.transpose(1, 2)
        
        return x
    
    
class SubEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n_resblocks):
        super().__init__()
        
        self.strided = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )
        
        if n_resblocks > 0:
            self.resblocks = nn.ModuleList()
            for _ in range(n_resblocks):
                self.resblocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.BatchNorm2d(num_features=out_channels)
                    )
                )
        else:
            self.resblocks = None
    
    def forward(self, x):
        # x: (b, c, t, d)
        
        x = self.strided(x)
        
        if self.resblocks is not None:
            for layer in self.resblocks:
                x = x + layer(x)
                x = nn.functional.relu(x)

        return x
    
    
class SubDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n_resblocks):
        super().__init__()
        
        if n_resblocks > 0:
            self.resblocks = nn.ModuleList()
            for _ in range(n_resblocks):
                self.resblocks.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=in_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                        nn.BatchNorm2d(num_features=in_channels)
                    )
                )
        else:
            self.resblocks = None
            
        self.transposed = nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
    
    def forward(self, x):
        # x: (b, c, t, d)
        
        if self.resblocks is not None:
            for layer in self.resblocks:
                x = x + layer(x)
                x = nn.functional.relu(x)
                
        x = self.transposed(x)
        
        return x
        

class SETransModule(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.att_norm = nn.LayerNorm(d_model)
        self.att = MultiHeadAttention(n_head=n_heads, n_feat=d_model, dropout_rate=dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        x = self.att_norm(residual)
        x = self.att(query=x, key=x, value=x, mask=None)
        residual = residual + self.dropout(x)
        
        x = self.ff_norm(residual)
        x = self.ff(x)
        residual = residual + self.dropout(x)
        
        return self.activation(residual)


def calc_length(lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)