
# UniSign-Prompt: Signer Bias Unlearning via Multimodal Prompt Tuning for Cross-Lingual Sign Language Translation

UniSign-Prompt introduces a novel architecture for continuous sign language translation (SLT), designed to mitigate signer bias and improve cross-lingual generalization. The model integrates Prompt-Injected Sign Transformer (PI-ST+), Hierarchical Cross-Lingual Prompt Bank (H-CLPB), Temporal-Aware Prompt Injection (TAP), Prompt Routing Mechanism (PRM), and Prompt Forgetting Module (PFM).

## Architecture Overview

- **PI-ST+**: Visual encoder with temporal prompt injection and prompt-centric residual connections.
- **H-CLPB**: Hierarchical prompts conditioned on language family and signer identity.
- **TAP**: Segment-specific temporal prompt generation.
- **PRM**: Gumbel-softmax based prompt routing.
- **PFM**: Signer bias unlearning via adversarial classifier and decorrelation.
- **Dual-Branch Decoder**: Joint gloss and text decoding.

## Datasets

- **How2Sign (ASL)**
- **RWTH-PHOENIX14T (DGS)**
- **ISL-CSLTR (ISL zero-shot and few-shot splits)**

### Expected Dataset Structure
All datasets are organized inside the `datasets/` folder with metadata CSV files and vocabulary text files:
```
datasets/
├── how2sign_train_meta.csv, how2sign_val_meta.csv, how2sign_test_meta.csv
├── how2sign_src_vocab.txt, how2sign_trg_vocab.txt
├── rwth_train_meta.csv, rwth_val_meta.csv, rwth_test_meta.csv
├── rwth_src_vocab.txt, rwth_trg_vocab.txt
├── isl_train_meta.csv, isl_val_meta.csv, isl_test_meta.csv
├── isl_zero_shot_meta.csv, isl_few_shot_train_meta.csv, isl_few_shot_val_meta.csv, isl_few_shot_test_meta.csv
├── isl_src_vocab.txt, isl_trg_vocab.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

Specify the dataset inside `train.py` by setting:
```python
DATASET = "How2Sign"
```

Run training:
```bash
python train.py
```

The best checkpoint will be saved in:
```
checkpoints/{DATASET}_UniSignPrompt_best.pth
```
e.g., `checkpoints/How2Sign_UniSignPrompt_best.pth`

## Evaluation (BLEU, WER, Cross-Lingual, Signer Accuracy)

```bash
python evaluate_extended.py --dataset ISL-CSLTR --split zero_shot --checkpoint checkpoints/ISL-CSLTR_UniSignPrompt_best.pth
```

## Inference on Single Video

Coming soon via `inference_demo.ipynb`

## Results Summary

| Dataset         | BLEU-4 ↑          | WER ↓ | Signer Acc ↓ | Params ↓ | Latency ↓    |
|-----------------|-------------------|-------|--------------|----------|--------------|
| ASL             | +1.1 vs baselines | -1.3  | -19.4%       | -17.9%   | -4.2 ms/frame |
| DGS             | +1.1              | -3.3  | -19.5%       | -17.9%   | -3.9 ms/frame |
| ISL (Zero-shot) | +0.7              | -1.6  | -18.2%       | -23.2%   | -2.9 ms/frame |
| ISL (Few-shot)  | +0.7              | -1.9  | -17.2%       | -23.2%   | -2.9 ms/frame |

## License

MIT License

## Citation

```bibtex
@article{unistprompt2025,
  title={Signer Bias Unlearning via Multimodal Prompt Tuning for Cross-Lingual Sign Language Translation},
  author={Elakkiya R},
  year={2025}
}
```
