import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
import json
from speechbrain.lobes.features import Fbank
import speechbrain as sb

# [{idx : {encoded_text : Tensor, wav_path : text} }]

def load_json(path):
    """
    Load a json file and return the content as a dictionary.
    """
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

class Vocab:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        self.itos = {v: k for k, v in self.vocab.items()}
        self.stoi = self.vocab

    def get_sos_token(self):
        return self.stoi["<s>"]
    def get_eos_token(self):
        return self.stoi["</s>"]
    def get_pad_token(self):
        return self.stoi["<pad>"]
    def get_unk_token(self):
        return self.stoi["<unk>"]
    def get_blank_token(self):
        return self.stoi["<blank>"]
    def __len__(self):
        return len(self.vocab)

class Speech2Text(Dataset):
    def __init__(self, json_path, vocab_path, train = True):
        super().__init__()
        self.data = load_json(json_path)
        self.vocab = Vocab(vocab_path)
        self.sos_token = self.vocab.get_sos_token()
        self.eos_token = self.vocab.get_eos_token()
        self.pad_token = self.vocab.get_pad_token()
        self.unk_token = self.vocab.get_unk_token()

        self.train = train    

        self.fbank = Fbank(
            sample_rate=16000,
            n_mels=160,
            n_fft=512,
            win_length=25,
        )

    def __len__(self):
        return len(self.data)
    
    def _extract_feature(self, waveform, sample_rate=16000):
        # mel_extractor = T.MelSpectrogram(
        #     sample_rate=sample_rate,
        #     n_fft=512,
        #     win_length=int(0.032 * 16000),
        #     hop_length=int(0.010 * 16000),
        #     n_mels=80,
        #     power=2.0
        # )

        # log_mel = mel_extractor(waveform.unsqueeze(0))
        # log_mel = torchaudio.functional.amplitude_to_DB(log_mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
        # log_mel = log_mel.squeeze(0)
    
        # mean = log_mel.mean(dim=1, keepdim=True)
        # std = log_mel.std(dim=1, keepdim=True)
        # normalized_log_mel_spec = (log_mel - mean) / (std + 1e-5)

        fbank = self.fbank(waveform).squeeze(0)  # [T, 80]

        return fbank
       
    def extract_from_path(self, wave_path):
        sig  = sb.dataio.dataio.read_audio(wave_path)

        return self._extract_feature(sig.unsqueeze(0))

    def __getitem__(self, idx):
        current_item = self.data[idx]
        wav_path = current_item["wav_path"]
        encoded_text = torch.tensor(current_item["encoded_text"] + [self.eos_token], dtype=torch.long)
        decoder_input = torch.tensor([self.sos_token] + current_item["encoded_text"], dtype=torch.long)
        tokens = torch.tensor(current_item["encoded_text"], dtype=torch.long)
        fbank = self.extract_from_path(wav_path).float()  # [T, 40]
        
        return {
            "text": encoded_text,
            "fbank": fbank,
            "text_len": len(encoded_text),
            "fbank_len": fbank.shape[0],
            "decoder_input": decoder_input,
            "tokens": tokens,
        }
    
from torch.nn.utils.rnn import pad_sequence

def calculate_mask(lengths, max_len):
    """Tạo mask cho các tensor có chiều dài khác nhau"""
    mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
    return mask

def speech_collate_fn(batch):
    batch = sorted(batch, key=lambda x: x['fbank_len'], reverse=True)
    # decoder_outputs = [torch.tensor(item["decoder_input"]) for item in batch]
    decoder_outputs = [item["decoder_input"].detach().clone() for item in batch]
    texts = [item["text"] for item in batch]
    fbanks = [item["fbank"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
    fbank_lens = torch.tensor([item["fbank_len"] for item in batch], dtype=torch.long)
    tokens_lens = torch.tensor([len(item["tokens"]) for item in batch], dtype=torch.long)

    padded_decoder_inputs = pad_sequence(decoder_outputs, batch_first=True, padding_value=0)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)       # [B, T_text]
    padded_fbanks = pad_sequence(fbanks, batch_first=True, padding_value=0.0)   # [B, T_audio, 40]
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)      # [B, T_text]

    speech_mask=calculate_mask(fbank_lens, padded_fbanks.size(1))      # [B, T]
    text_mask=calculate_mask(text_lens, padded_texts.size(1))

    return {
        "decoder_input": padded_decoder_inputs,
        "text": padded_texts,
        "text_mask": text_mask,
        "text_len" : text_lens,
        "fbank_len" : fbank_lens,
        "fbank": padded_fbanks,
        "fbank_mask": speech_mask,
        "tokens" : padded_tokens,
        "tokens_lens": tokens_lens
    }

class SpecAugment(nn.Module):

    """Spectrogram Augmentation

    Args:
        spec_augment: whether to apply spec augment
        mF: number of frequency masks
        F: maximum frequency mask size
        mT: number of time masks
        pS: adaptive maximum time mask size in %

    References:
        SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition, Park et al.
        https://arxiv.org/abs/1904.08779

        SpecAugment on Large Scale Datasets, Park et al.
        https://arxiv.org/abs/1912.05533

    """

    def __init__(self, spec_augment, mF, F, mT, pS):
        super(SpecAugment, self).__init__()
        self.spec_augment = spec_augment
        self.mF = mF
        self.F = F
        self.mT = mT
        self.pS = pS

    def forward(self, x, x_len):

        # Spec Augment
        if self.spec_augment:
        
            # Frequency Masking
            for _ in range(self.mF):
                x = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.F, iid_masks=False).forward(x)

            # Time Masking
            for b in range(x.size(0)):
                T = int(self.pS * x_len[b])
                for _ in range(self.mT):
                    x[b:b+1, :, :x_len[b]] = torchaudio.transforms.TimeMasking(time_mask_param=T).forward(x[b:b+1, :, :x_len[b]])

        return x