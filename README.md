# English to French Neural Machine Translation with Transformer

This project implements a Transformer-based neural machine translation model for translating English to French using PyTorch.

## Project Overview

The project focuses on implementing a Transformer architecture from scratch for English-French translation, following the architecture described in "Attention is All You Need" paper. The implementation includes custom-built encoder-decoder architecture, self-attention mechanisms, and positional encodings without using PyTorch's built-in transformer modules.

## Project Structure

```
Neural-Machine-Translation-with-Transformer/
│
├── README.md                # Project documentation
├── test.out                 # Model output on test data
├── testbleu.txt             # BLEU scores for test sentences
├── transformer.pt           # Pretrained model weights
├── ANLP_Assignment_2.pdf    # Assignment details and requirements
│
└── src/                     # Source code directory
    ├── encoder.py           # Encoder implementation
    ├── decoder.py           # Decoder and Transformer implementation
    ├── train.py             # Training script
    ├── test.py              # Testing and evaluation script
    └── utils.py             # Utility functions and helper classes
```

## Prerequisites

- Python 3.7+
- PyTorch 1.7+
- torchtext
- nltk
- tokenizers
- matplotlib
- tqdm

Install the required packages:

```bash
pip install torch torchtext nltk tokenizers matplotlib tqdm
```

## Usage

### Training the Model

1. Ensure you have the training data (train.en, train.fr) in the correct location
2. Run the training script:

```bash
cd src
python train.py
```

The script will:

- Load and preprocess the English-French parallel corpus
- Train the Transformer model
- Save the best model weights to `transformer.pt`
- Generate training logs and loss curves

### Testing the Model

1. Ensure you have the pretrained model (`transformer.pt`) and test data (test.en, test.fr)
2. Run the testing script:

```bash
cd src
python test.py
```

This will:

- Load the pretrained model
- Generate translations for test sentences
- Calculate and save BLEU scores in `testbleu.txt`
- Output the overall corpus BLEU score

## Implementation Details

- Uses Byte-Pair Encoding (BPE) for tokenization
- Maximum sequence length: 300 tokens
- Vocabulary size: 5000 (both languages)
- Default architecture: 3 layers, 4 attention heads, embedding dimension 256
- Uses teacher forcing during training
- Implements beam search (beam size=5) for inference
- Adam optimizer with learning rate 0.0001
- Trained for 30 epochs with early stopping

## Pretrained Model

The pretrained model weights are available at:
[Download Pretrained Model](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/gaurav_bhole_research_iiit_ac_in/EXa_W5VRCtxAhX2rVkTYlcIBz6S5Ow37wFBJgSjjuXYoBA?e=VqwtS6)

Place the downloaded `transformer.pt` file in the project root directory.

## Dataset

The project uses a subset of the IWSLT 2016 English-French translation dataset:

- Training: 30,000 sentence pairs
- Development: 887 sentence pairs
- Test: 1,305 sentence pairs

## Acknowledgments

- Based on the "Attention is All You Need" paper
- Implementation guidance from "The Illustrated Transformer"
- Course materials from Advanced Natural Language Processing
