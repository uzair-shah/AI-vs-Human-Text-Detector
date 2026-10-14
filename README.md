# AI vs Human Text Detector

A neural network that predicts whether a paragraph was written by a human or by AI.

## How it works

1. **Data:** each sentence comes pre-embedded as a 100 × 768 array of token embeddings, zero-padded. Training uses about 17.7k sentences: human ones labelled 0, AI ones labelled 1, including some synthetic examples.
2. **EDA:** the notebook compares sentence-length distributions. AI sentences are slightly longer on average but more uniform than human ones.
3. **Model:** a feed-forward network.
   - Flatten → Dense(300, ReLU) → Dense(100, ReLU) → Dense(1, sigmoid)
   - Binary cross-entropy, the Nadam optimizer, an exponential learning-rate schedule and early stopping
4. **Paragraph scoring:** each sentence in a paragraph is scored, and the probabilities are averaged into a single paragraph score.
5. **Output:** predictions for the test set are written to `submission_FNN_model_4.csv`, with columns `id` and `y_prob`.

## Results

| Level | Validation AUC |
|-----------|------|
| Sentence  | ~0.95 |
| Paragraph | 0.90 |

## Setup

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

Expected folder layout:

```
train_files/
  train_human.npy, train_ai.npy, train_human_synth.npy, train_ai_synth.npy
  validation.jsonl, test_features.jsonl
Saved_Models/
notebooks/
  Final_Notebook.ipynb
```

Then run the notebook top to bottom.
