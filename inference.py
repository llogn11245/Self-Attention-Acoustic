import os
import csv
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from models.model import AcousticModel
from utils.dataset import Speech2Text, speech_collate_fn
from jiwer import wer, cer
from contextlib import contextmanager

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ids_to_text(ids, itos, eos_id=None):
    tokens = []
    for idx in ids:
        if eos_id is not None and idx == eos_id:
            break
        token = itos.get(idx, '')
        if token in ['<pad>','<s>','</s>','<unk>','<blank>']:
            continue
        tokens.append(token)
    return ' '.join(tokens)

@contextmanager
def optional_file(filename):
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            yield f
    else:
        yield None

def main():
    parser = argparse.ArgumentParser(description="Inference script for RNN-T speech-to-text model")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--epoch', type=int, default=1, help='Epoch number of the checkpoint to load')
    parser.add_argument('--output', default=None, help='File to save predictions (optional, uses config if not specified)')
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    model_cfg = full_cfg.get('model', full_cfg)

    # Xác định output file path
    if args.output is None:
        # Lấy từ config nếu không truyền --output
        output_file = full_cfg['training'].get('infer_path', None)
    else:
        output_file = args.output

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #===Load Checkpoint===
    checkpoint_path = os.path.join(full_cfg['training']['save_path'], f"SAA_epoch_{args.epoch}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    #===Load Model===
    model = AcousticModel(model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #===Load Data===
    dataset = Speech2Text(full_cfg['training']['test_path'], 
                          full_cfg['training']['vocab_path']
                          )
    itos    = dataset.vocab.itos
    eos_id  = dataset.vocab.get_eos_token()

    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        collate_fn=speech_collate_fn)

    pred_texts = []
    true_texts = []

    with optional_file(output_file) as fout:
        for batch in loader:
            speech = batch["fbank"].to(device)
            target_text = batch["text"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            fbank_len = batch["fbank_len"].to(device)
            text_len = batch["text_len"].to(device)
            decoder_input = batch["decoder_input"].to(device)

            with torch.no_grad():
                batch_preds = model.recognize(enc_inputs=speech, 
                                              speech_length=fbank_len, 
                                              target_length=text_len, 
                                              enc_mask=speech_mask)

            for i in range(len(batch_preds)):
                pred_ids = batch_preds[i]
                true_ids = batch['text'][i].tolist()

                pred_text = ids_to_text(pred_ids, itos, eos_id=eos_id)
                true_text = ids_to_text(true_ids, itos, eos_id=eos_id)

                pred_texts.append(pred_text)
                true_texts.append(true_text)
                print(f"Predict text: {pred_text}")
                print(f"Ground truth: {true_text}")
                wer_score = wer([true_text], [pred_text])
                cer_score = cer([true_text], [pred_text])
                print(f"WER: {wer_score:.4f}, CER: {cer_score:.4f}")
                if fout and pred_text:
                    fout.write(f"Predict text: {pred_text}\n")
                    fout.write(f"Ground truth: {true_text}\n")
                    fout.write("---------------\n")

    if output_file:
        print(f"Inference complete. Results saved to {output_file}")
    else:
        print("Inference complete. Results not saved to file.")

    #===TÍNH WER VÀ CER===
    overall_wer = wer(true_texts, pred_texts)
    overall_cer = cer(true_texts, pred_texts)
    print(f"Word Error Rate (WER): {overall_wer:.4f}")
    print(f"Character Error Rate (CER): {overall_cer:.4f}")

if __name__ == '__main__':
    main()