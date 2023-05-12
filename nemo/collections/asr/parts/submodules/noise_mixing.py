import numpy as np
import torch

class NoiseMixer:
    def __init__(
        self,
        real_noise_filepath=None,
        real_noise_snr=[0, 0],
        white_noise_mean=0.0,
        white_noise_std=[0.0, 0.0],
        ):
        super().__init__()
        
        self.add_noise_methods = []
        if real_noise_filepath is not None and real_noise_snr[1] > real_noise_snr[0]:
            self.add_noise_methods.append(self._add_real_noise)
            self.real_noise_corpus = np.load(real_noise_filepath, allow_pickle=True)
            self.real_noise_snr = real_noise_snr
        if white_noise_std[0] >= 0.0 and white_noise_std[1] > white_noise_std[0]:
            self.add_noise_methods.append(self._add_white_noise)
            self.white_noise_mean = white_noise_mean
            self.white_noise_std = white_noise_std
    
    def __call__(self, signal):
        if len(self.add_noise_methods) < 1:
            return signal
        method = np.random.choice(self.add_noise_methods)
        return method(signal)
    
    def _add_real_noise(self, signal):
        def get_noise(noise_corpus, length):
            noise = np.random.choice(noise_corpus)
            start = np.random.randint(0, len(noise) - length - 1)
            return noise[start:start + length]
        
        signal_length = signal.size(1)
        noise_signal = signal.clone().detach()
        
        for idx in range(len(signal)):
            noise = get_noise(self.real_noise_corpus, signal_length)
            noise = torch.from_numpy(noise).to(signal.device)

            # calculate power of audio and noise
            snr = torch.randint(low=self.real_noise_snr[0], high=self.real_noise_snr[1], size=(1,)).to(signal.device)
            signal_energy = torch.mean(noise_signal[idx]**2)
            noise_energy = torch.mean(noise**2)
            coef = torch.sqrt(10.0 ** (-snr/10) * signal_energy / noise_energy)
            signal_coef = torch.sqrt(1 / (1 + coef**2))
            noise_coef = torch.sqrt(coef**2 / (1 + coef**2))
            noise_signal[idx] = signal_coef * noise_signal[idx] + noise_coef * noise
            
        del noise, snr, signal_energy, noise_energy, coef, signal_coef, noise_coef
        return noise_signal
    
    def _add_white_noise(self, signal):
        noise_signal = signal.clone().detach()
        for idx in range(len(signal)):
            std = np.random.uniform(self.white_noise_std[0], self.white_noise_std[1])
            noise = np.random.normal(self.white_noise_mean, std, size=signal[idx].shape)
            noise = torch.from_numpy(noise).type(torch.FloatTensor)
            noise_signal[idx] = noise_signal[idx] + noise.to(signal.device)
        del noise
        return noise_signal