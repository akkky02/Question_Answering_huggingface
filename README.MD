# Extractive Question Answering Given Context with SQuad

## Overview
This Jupyter notebook trains a model to perform extractive question answering using the SQuAD dataset. The model takes a context paragraph and a question as input, and identifies the answer span from the context paragraph.

## Outline
The notebook covers the following topics:

1. **Environment Setup**: Installs necessary libraries like Transformers, datasets, etc. and sets up paths.

2. **Data Exploration**: Loads and explores the SQuAD dataset to understand its structure. Visualizes length distribution of context paragraphs.

3. **Data Preprocessing**: Tokenizes the questions and context paragraphs using DistilBERT tokenizer. Handles chunking of long context paragraphs. Aligns answer start/end positions to token indices.

4. **Model Training**: Fine-tunes a DistilBERT model on the processed SQuAD data for extractive question answering. Saves checkpoints during training.

5. **Inference**: Loads the best model checkpoint and demonstrates model predictions on a few sample questions/context.

## Requirements
The main libraries used are:
- Transformers
- Datasets
- Accelerate

The notebook requires a Python environment and access to a GPU for model training.

## Usage
To run the notebook end-to-end:

1. Install requirements
2. Mount Google Drive (for data storage) 
3. Run all cells in order

The trained model checkpoints can be used later for inference/deployment.
