import torch
from tqdm import tqdm
import torch.nn.functional as F
from decoder import Transformer
from utils import read_data, create_bpe_tokenizer, TranslateDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu


def beam_search(model, src, source_tokenizer, target_tokenizer, max_len, device, beam_size=5):
    model.eval()
    src = src.to(device)
    src_mask = model.make_source_mask(src)
    enc_output = model.encoder(src, src_mask)

    sos_token = target_tokenizer.token_to_id("[SOS]")
    eos_token = target_tokenizer.token_to_id("[EOS]")

    beam = [(torch.tensor([sos_token], device=device), 0)]
    completed_sequences = []

    for _ in range(max_len):
        candidates = []
        for seq, score in beam:
            if seq[-1].item() == eos_token:
                completed_sequences.append((seq, score))
                continue

            trg_mask = model.make_target_mask(seq.unsqueeze(0))
            output = model.decoder(seq.unsqueeze(0), enc_output, src_mask, trg_mask)
            output = model.final_linear(output[:, -1])
            probabilities = F.log_softmax(output, dim=-1)
            
            top_probs, top_indices = probabilities.topk(beam_size)
            
            for prob, idx in zip(top_probs[0], top_indices[0]):
                candidates.append((torch.cat([seq, idx.unsqueeze(0)]), score + prob.item()))

        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        if len(completed_sequences) >= beam_size:
            break

    if not completed_sequences:
        completed_sequences = beam

    completed_sequences = sorted(completed_sequences, key=lambda x: x[1], reverse=True)
    best_seq = completed_sequences[0][0][1:]  # Remove SOS token
    return best_seq

def translate(model, sentence, source_tokenizer, target_tokenizer, device, max_len=100, beam_size=5):
    model.eval()
    tokens = source_tokenizer.encode(sentence).ids
    src = torch.tensor([tokens], device=device)
    
    translation_tensor = beam_search(model, src, source_tokenizer, target_tokenizer, max_len, device, beam_size)
    translation = target_tokenizer.decode(translation_tensor.tolist())
    
    return translation

def calculate_bleu_score(model, test_loader, source_tokenizer, target_tokenizer, device):
    model.eval()
    hypotheses = []
    references = []
    smoothing = SmoothingFunction().method1

    with torch.no_grad(), open("testbleu.txt", "w", encoding="utf-8") as f:
        for batch in tqdm(test_loader, desc="Calculating BLEU Score"):
            src = batch["source_ids"].to(device)
            trg = batch["target_ids"].to(device)
            
            for i in range(src.shape[0]):
                src_sentence = source_tokenizer.decode(src[i].tolist())
                trg_sentence = target_tokenizer.decode(trg[i].tolist())
                
                translation = translate(model, src_sentence, source_tokenizer, target_tokenizer, device)
                
                hypothesis = translation.split()
                reference = [trg_sentence.split()]
                
                bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing)
                
                hypotheses.append(hypothesis)
                references.append(reference)

                f.write(f"{src_sentence} ||| {translation} ||| {bleu_score}\n")

    return hypotheses, references

if __name__ == "__main__":
    configs = {
        "test_source_data": "ted-talks-corpus/test.en",
        "test_target_data": "ted-talks-corpus/test.fr",
        "source_max_seq_len": 300,
        "target_max_seq_len": 300,
        "batch_size": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "embedding_dim": 256,
        "n_layers": 3,
        "n_heads": 4,
        "dropout": 0.1,
        "vocab_size": 5000,
        "model_path": "transformer.pt"
    }

    # Load test data
    test_src, test_trg = read_data(configs["test_source_data"], configs["test_target_data"])

    # Create BPE tokenizers
    source_tokenizer = create_bpe_tokenizer(test_src, configs["vocab_size"])
    target_tokenizer = create_bpe_tokenizer(test_trg, configs["vocab_size"])

    # Create test dataset and loader
    test_dataset = TranslateDataset(source_tokenizer, target_tokenizer, test_src, test_trg, configs["source_max_seq_len"], configs["target_max_seq_len"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs["batch_size"])

    # Initialize and load the model
    model = Transformer(
        source_vocab_size=source_tokenizer.get_vocab_size(),
        target_vocab_size=target_tokenizer.get_vocab_size(),
        source_max_seq_len=configs["source_max_seq_len"],
        target_max_seq_len=configs["target_max_seq_len"],
        embedding_dim=configs["embedding_dim"],
        num_heads=configs["n_heads"],
        num_layers=configs["n_layers"],
        dropout=configs["dropout"]
    ).to(configs["device"])

    model.load_state_dict(torch.load(configs["model_path"]))

    # Calculate BLEU scores and save results
    hypotheses, references = calculate_bleu_score(model, test_loader, source_tokenizer, target_tokenizer, configs["device"])

    # Calculate simple BLEU score using SmoothingFunction
    corpus_bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method7)

    print(f"Corpus BLEU Score: {corpus_bleu_score:.4f}")