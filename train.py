import torch
from utils.dataset import Speech2Text, speech_collate_fn
from models.model import AcousticModel
from tqdm import tqdm
from models.loss import build_loss, CTCLoss
import argparse
import yaml
import os 
from models.optim import Optimizer
from speechbrain.nnet.schedulers import NoamScheduler
import logging

def reload_model(model, optimizer, checkpoint_path, model_name):
    past_epoch = 0
    path_list = [path for path in os.listdir(checkpoint_path)]
    # print(path_list)
    if len(path_list) > 0:
        for path in path_list:
            try:
                past_epoch = max(int(path.split("_")[-1]), past_epoch)
            except:
                continue
        
        load_path = os.path.join(checkpoint_path, f"{model_name}_epoch_{past_epoch}")
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Reloaded model from {load_path} at epoch {past_epoch}")
    else:
        logging.info("No checkpoint found. Starting from scratch.")
    
    return past_epoch + 1, model, optimizer


def train_one_epoch(model, dataloader, optimizer, criterion, criterion2, ctc_weight, device, scheduler, accum_steps=1):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🔁 Training", leave=False)

    for id, batch in enumerate(progress_bar):
        speech = batch["fbank"].to(device)
        speech_mask = batch["fbank_mask"].to(device)
        text_mask = batch["text_mask"].to(device)
        fbank_len = batch["fbank_len"].to(device)
        text_len = batch["text_len"].to(device)
        target_text = batch["text"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        tokens = batch["tokens"].to(device)
        tokens_lens = batch["tokens_lens"].to(device)

        enc_out, ctc_out, dec_out = model(speech, fbank_len.long(), decoder_input, text_len.long(), speech_mask, train=True, tfr=0.8)

        loss_kldiv = criterion(dec_out, target_text, fbank_len, text_len)
        loss_ctc = criterion2(ctc_out, tokens, fbank_len, tokens_lens)
        loss = loss_ctc * ctc_weight + loss_kldiv * (1- ctc_weight)
        loss.backward()

        if ((id + 1) % accum_steps == 0) or ((id + 1) == len(dataloader)):
            optimizer.step()

            curr_lr, _ = scheduler(optimizer.optimizer)

            optimizer.zero_grad()  

        total_loss += loss.item()
        progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Average training loss: {avg_loss:.4f}")
    return avg_loss, curr_lr


def evaluate(model, dataloader, criterion, criterion2, ctc_weight, device):
    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(dataloader, desc="🧪 Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            speech = batch["fbank"].to(device)
            speech_mask = batch["fbank_mask"].to(device)
            text_mask = batch["text_mask"].to(device)
            fbank_len = batch["fbank_len"].to(device)
            text_len = batch["text_len"].to(device)
            target_text = batch["text"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            tokens = batch["tokens"].to(device)
            tokens_lens = batch["tokens_lens"].to(device)

            enc_out, ctc_out, dec_out = model(speech, fbank_len.long(), decoder_input, text_len.long(), speech_mask, train=True, tfr=1.0)
            
            loss_kldiv = criterion(dec_out, target_text, fbank_len, text_len)
            loss_ctc = criterion2(ctc_out, tokens, fbank_len, tokens_lens)
            loss = loss_ctc * ctc_weight + loss_kldiv * (1- ctc_weight)

            total_loss += loss.item()
            progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Average validation loss: {avg_loss:.4f}")
    return avg_loss


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config['training']

    # ==== Logger ====
    if not os.path.exists(training_cfg['log_path']):
        os.makedirs(training_cfg['log_path'])
    log_file = training_cfg['log_path'] + '/saa.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.Fil eHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )

    # ==== Load Dataset ====
    train_dataset = Speech2Text(
        json_path=training_cfg['train_path'],
        vocab_path=training_cfg['vocab_path'],
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        collate_fn=speech_collate_fn,
        num_workers=2,
        pin_memory=False
    )

    dev_dataset = Speech2Text(
        json_path=training_cfg['dev_path'],
        vocab_path=training_cfg['vocab_path'],
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        collate_fn=speech_collate_fn,
        num_workers=2,
        pin_memory=False
    )

    vocab_size = len(train_dataset.vocab)

    # ==== Model ====
    model = AcousticModel(config['model'], vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ==== Loss ====
    kldiv_loss = build_loss(config["loss"])
    ctc_loss = CTCLoss(blank=config["model"]["blank_id"], reduction=config["loss"]["ctc_reduction"])
    ctc_weight = config["loss"]["ctc_weight"]

    # ==== Optimizer ====
    optimizer = Optimizer(model.parameters(), config['optim'])

    # ==== Scheduler ====
    if not config['training']['reload']:
        scheduler = NoamScheduler(
            n_warmup_steps=config['scheduler']['n_warmup_steps'],
            lr_initial=config['scheduler']['lr_initial']
        )
    else:
        scheduler = NoamScheduler(
            n_warmup_steps=config['scheduler']['n_warmup_steps'],
            lr_initial=config['scheduler']['lr_initial']
        )
        scheduler.load(config['training']['save_path'] + '/scheduler.ckpt')

    # ==== Reload checkpoint if needed ====
    start_epoch = 1
    if training_cfg['reload']:
        checkpoint_path = training_cfg['save_path']
        start_epoch, model, optimizer = reload_model(model, optimizer, checkpoint_path, config['model']['name'])

    # ==== Training loop ====
    num_epochs = training_cfg["epochs"]
    accumulation_steps = training_cfg["accumulation_steps"]

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, lr = train_one_epoch(model, train_loader, optimizer, kldiv_loss, ctc_loss, ctc_weight, device, scheduler, accumulation_steps)
        val_loss = evaluate(model, dev_loader, kldiv_loss, ctc_loss, ctc_weight, device)

        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {lr:6f}")

        # Save model checkpoint
        if not os.path.exists(training_cfg['save_path']):
            os.makedirs(training_cfg['save_path'])
        model_filename = os.path.join(
            training_cfg['save_path'],
            f"{config['model']['name']}_epoch_{epoch}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_filename)

        scheduler.save(config['training']['save_path'] + '/scheduler.ckpt')


if __name__ == "__main__":
    main()