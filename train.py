import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu
from jiwer import wer

from models import UniSignPrompt
from losses.multi_objective_forgetting_loss import MultiObjectiveForgettingLoss
from datasets.how2sign_dataset import How2SignDataset
from datasets.rwth_phoenix_dataset import RWTHPhoenixDataset
from datasets.isl_csltr_dataset import ISLCSLTRDataset

# ---------------------- Config ---------------------- #
DATASET = "How2Sign"  # Change to RWTH-PHOENIX14T or ISL-CSLTR as needed
DATA_PATH = "datasets/"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 2e-4
DEVICE = torch.device("cuda")
LOG_DIR = "runs/UniSignPrompt/"
CHECKPOINT = "checkpoints/UnisignPrompt_best.pth"

# ---------------------- Dataset ---------------------- #
if DATASET == "How2Sign":
    train_set = How2SignDataset(DATA_PATH, split="train")
    val_set = How2SignDataset(DATA_PATH, split="val")
elif DATASET == "RWTH-PHOENIX14T":
    train_set = RWTHPhoenixDataset(DATA_PATH, split="train")
    val_set = RWTHPhoenixDataset(DATA_PATH, split="val")
elif DATASET == "ISL-CSLTR":
    train_set = ISLCSLTRDataset(DATA_PATH, split="train")
    val_set = ISLCSLTRDataset(DATA_PATH, split="val")
else:
    raise NotImplementedError("Dataset not implemented")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------------- Model + Optimizer ---------------------- #
model = UniSignPrompt().to(DEVICE)
criterion = MultiObjectiveForgettingLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)
writer = SummaryWriter(LOG_DIR)

best_bleu4 = 0

# ---------------------- Training Loop ---------------------- #
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        visual = batch['visual'].to(DEVICE)
        signer = batch['signer_onehot'].to(DEVICE)
        language = batch['language_onehot'].to(DEVICE)
        gloss = batch['gloss'].to(DEVICE)
        text = batch['text'].to(DEVICE)

        output = model(visual, signer, language)
        log_probs = nn.functional.log_softmax(output['text_logits'], dim=-1)
        loss_dict = criterion(log_probs, text, output['routing_scores'], signer.argmax(dim=1), None, None, output['routing_scores'], None, [])
        loss = loss_dict['total_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    print(f"Epoch {epoch+1}: Train Loss={{avg_loss:.4f}}")

    # ---------------------- Validation ---------------------- #
    model.eval()
    refs, hyps, wer_refs, wer_hyps = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            visual = batch['visual'].to(DEVICE)
            signer = batch['signer_onehot'].to(DEVICE)
            language = batch['language_onehot'].to(DEVICE)
            text = batch['text'].to(DEVICE)

            output = model(visual, signer, language)
            pred_tokens = output['text_logits'].argmax(dim=-1).cpu().tolist()
            true_tokens = text.cpu().tolist()

            refs.extend([[ref] for ref in true_tokens])
            hyps.extend(pred_tokens)
            wer_refs.extend([" ".join(map(str, ref)) for ref in true_tokens])
            wer_hyps.extend([" ".join(map(str, hyp)) for hyp in pred_tokens])

    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25)) * 100
    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0)) * 100
    wer_score = wer(wer_refs, wer_hyps) * 100

    writer.add_scalar('Val/BLEU4', bleu4, epoch)
    writer.add_scalar('Val/BLEU1', bleu1, epoch)
    writer.add_scalar('Val/WER', wer_score, epoch)
    print(f"Validation BLEU4={{bleu4:.2f}}, BLEU1={{bleu1:.2f}}, WER={{wer_score:.2f}}")

    if bleu4 > best_bleu4:
        torch.save(model.state_dict(), CHECKPOINT)
        best_bleu4 = bleu4
        print(f"Checkpoint saved at Epoch {{epoch+1}}")

writer.close()
