---
library_name: transformers
pipeline_tag: summarization
---
# Populism Detection & Summarization

This checkpoint is a BART-based, LoRA-fine-tuned model that does two things:

Summarizes party press releases (and, when relevant, explains where populist framing appears), and

Classifies whether the text contains populist language (Is_Populist ∈ {0,1}).

Weights here are the merged LoRA result—no adapters required.

The model was trained on ~10k official party press releases from 12 countries (Italy, Sweden, Switzerland, Netherlands, Germany, Denmark, Spain, UK, Austria, Poland, Ireland, France) that were labeled and summarized via a Palantir AIP Ontology step using GPT-4o.

## Model Details

Pretrained Model: facebook/bart-base (seq2seq) fine-tuned with LoRA and then merged.
Instruction Framing: Two prefixes:

Summarize: summarize: <original_text>

Classify: classify_populism: <original_text> → model outputs 0 or 1 (or you can argmax over first decoder step logits for tokens “0” vs “1”).

Tokenization: BART’s subword tokenizer (Byte-Pair Encoding).

Input Processing: Text is truncated to 1024 tokens; summaries capped at 128 tokens.

Output Generation (summarization): beam search (typically 5 beams), mild length penalty, and no-repeat bigrams to reduce redundancy.

Key Parameters:

Max Input Length: 1024 tokens — fits long releases while controlling memory.

Max Target Length: 128 tokens — concise summaries with good coverage.

Beam Search: ~5 beams — balances quality and speed.

Classification Decoding: read the first generated token (0/1) or take first-step logits for a deterministic argmax.

Generation Process (high level)

Input Tokenization: Convert text to subwords and build the encoder input.

Beam Search (summarize): Explore multiple candidate sequences, pick the most probable.

Output Decoding: Map token IDs back to text, skipping special tokens.

Model Hub: tdickson17/Populism_detection

Repository: https://github.com/tcdickson/Populism.git

## Training Details

Data Collection:
Press releases were scraped from official party websites to capture formal statements and policy messaging. A Palantir AIP Ontology step (powered by GPT-4o) produced:

Is_Populist (binary) — whether the text exhibits populist framing (e.g., “people vs. elites,” anti-institutional rhetoric).

Summaries/Explanations — concise abstracts; when populism is present, the text explains where/how it appears.

Preprocessing:
HTML/boilerplate removal, normalization, and formatting into pairs:

Input: original release text (title optional at inference)

Targets: (a) abstract summary/explanation, (b) binary label

Training Objective:
Supervised fine-tuning for joint tasks:

Abstractive summarization (seq2seq cross-entropy)

Binary classification (decoded 0/1 via the same seq2seq head)

Training Strategy:

Base: facebook/bart-base

Method: LoRA on attention/FFN blocks (r=16, α=32, dropout=0.05), then merged into base.

Decoding: beam search for summaries; argmax or short generation for labels.

Evaluation signals: ROUGE for summaries; Accuracy/Precision/Recall/F1 for classification.

This setup lets one checkpoint handle both analysis (populism flag) and explanation (summary) with simple instruction prefixes.

## Citation:

@article{dickson2024going,
  title={Going against the grain: Climate change as a wedge issue for the radical right},
  author={Dickson, Zachary P and Hobolt, Sara B},
  journal={Comparative Political Studies},
  year={2024},
  publisher={SAGE Publications Sage CA: Los Angeles, CA}
}
