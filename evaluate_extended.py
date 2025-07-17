import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from jiwer import wer

from models import UniSignPrompt
from datasets.how2sign_dataset import How2SignDataset
from datasets.rwth_phoenix_dataset import RWTHPhoenixDataset
from datasets.isl_csltr_dataset import ISLCSLTRDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['How2Sign', 'RWTH-PHOENIX14T', 'ISL-CSLTR'])
parser.add_argument('--split', type=str, default='test', choices=['val', 'test', 'zero_shot', 'few_shot'])
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

DEVICE = torch.device("cuda")

if args.dataset == "How2Sign":
    dataset = How2SignDataset("datasets/", split=args.split)
elif args.dataset == "RWTH-PHOENIX14T":
    dataset = RWTHPhoenixDataset("datasets/", split=args.split)
elif args.dataset == "ISL-CSLTR":
    dataset = ISLCSLTRDataset("datasets/", split=args.split)
else:
    raise NotImplementedError

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

model = UniSignPrompt().to(DEVICE)
model.load_state_dict(torch.load(args.checkpoint))
model.eval()

refs, hyps, wer_refs, wer_hyps = [], [], [], []
routing_sparsity_list, latency_list = [], []
signer_preds, signer_labels = [], []

with torch.no_grad():
    for batch in dataloader:
        visual = batch['visual'].to(DEVICE)
        signer = batch['signer_onehot'].to(DEVICE)
        language = batch['language_onehot'].to(DEVICE)
        text = batch['text'].to(DEVICE)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        output = model(visual, signer, language)

        end_event.record()
        torch.cuda.synchronize()
        latency_list.append(start_event.elapsed_time(end_event) / visual.shape[1])

        pred_tokens = output['text_logits'].argmax(dim=-1).cpu().tolist()
        true_tokens = text.cpu().tolist()

        refs.extend([[ref] for ref in true_tokens])
        hyps.extend(pred_tokens)
        wer_refs.extend([" ".join(map(str, ref)) for ref in true_tokens])
        wer_hyps.extend([" ".join(map(str, hyp)) for hyp in pred_tokens])

        if 'routing_scores' in output:
            routing_sparsity_list.append((output['routing_scores'] > 0.01).float().mean().item())

        if output['forgetting_loss'] is not None:
            signer_logits = output['forgetting_loss']['signer_logits'].cpu().argmax(dim=-1)
            signer_labels_batch = signer.argmax(dim=1).cpu()
            signer_preds.extend(signer_logits.tolist())
            signer_labels.extend(signer_labels_batch.tolist())

bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25)) * 100
bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0)) * 100
meteor = sum([meteor_score([" ".join(map(str, ref[0]))], " ".join(map(str, hyp))) for ref, hyp in zip(refs, hyps)]) / len(refs) * 100
rouge_l = sum([len(set(ref[0]) & set(hyp)) / len(set(ref[0])) for ref, hyp in zip(refs, hyps)]) / len(refs) * 100
wer_score = wer(wer_refs, wer_hyps) * 100

print(f"BLEU-1: {bleu1:.2f}, BLEU-4: {bleu4:.2f}, METEOR: {meteor:.2f}, ROUGE-L: {rouge_l:.2f}, WER: {wer_score:.2f}")

if signer_preds:
    signer_acc = (torch.tensor(signer_preds) == torch.tensor(signer_labels)).float().mean().item() * 100
    print(f"Signer Classification Accuracy: {signer_acc:.2f}%")

if routing_sparsity_list:
    avg_sparsity = sum(routing_sparsity_list) / len(routing_sparsity_list) * 100
    print(f"Routing Sparsity (% active prompts): {avg_sparsity:.2f}%")

avg_latency = sum(latency_list) / len(latency_list)
print(f"Average Inference Latency (ms/frame): {avg_latency:.2f}")
