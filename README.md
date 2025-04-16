# Tokenizer
This repository contains an implementation of a Byte Pair Encoding (BPE) tokenizer, designed for efficient text tokenization. The tokenizer is trained on the TinyStoriesV2 dataset and supports multiprocessing for faster pre-tokenization during training.
## Overview
The BPE tokenizer is implemented to process text data by iteratively merging the most frequent pair of bytes (or characters) into a single token. This implementation includes:

- **Multiprocessing**: Utilizes Python's multiprocessing to parallelize the pre-tokenization step, significantly speeding up the training process.
- **Dataset**: Trained on the TinyStoriesV2 dataset from Hugging Face.
- **Special Tokens**: Supports special tokens like <|endoftext|>, which can be specified during training and inference.

## Requirements

Python 3.7+
Python 3.8+



Required packages:
- `regex` (for advanced regular expression handling)
- `tqdm` (for progress bars during training)
- Standard Python libraries: os, `multiprocessing`, `functools`, `typing`

## Training the Tokenizer
To train the BPE tokenizer, use the run_train_bpe function. The training script processes the input text file, applies multiprocessing for pre-tokenization, and generates a vocabulary and merge rules.
Example
```python
results = run_train_bpe('data/TinyStoriesV2-GPT4-valid.txt', 10000, ['<|endoftext|>'])
```


### ***Arguments***:
- ***input_file***: Path to the training text file (e.g., TinyStoriesV2-GPT4-valid.txt).
- ***vocab_size***: Desired vocabulary size (e.g., 10000).
special_tokens: List of special tokens to include (e.g., ['<|endoftext|>']).

### ***Output***:
A tuple containing the vocabulary and merge rules, saved as a .pkl file (e.g., valid_results.pkl).


The training process uses the TinyStoriesV2 dataset, which can be downloaded from Hugging Face.
## Inference with the Tokenizer
To use the trained BPE tokenizer for encoding text, load the saved vocabulary and merge rules, then instantiate the BPETokenizer class.
Example:
```python
import pickle

# Load the .pkl file
with open('valid_results.pkl', 'rb') as f:
data = pickle.load(f)  # or torch.load(f) if it's a PyTorch model

vocab = data[0]
merges = data[1]

# Initialize the tokenizer
tok = BPETokenizer(vocab, merges, ['<|endoftext|>'])

# Encode a sample text
encoded = tok.encode("<|endoftext|>It was a sunny day and two friends decided to travel.<|endoftext|>")
print(encoded)
```

### ***Input***:
- vocab: The vocabulary dictionary from training.
- merges: The merge rules from training.
- special_tokens: List of special tokens used during training.


### ***Output***:
A list of token IDs representing the encoded input text.



## Dataset
The tokenizer was trained on the TinyStoriesV2 dataset, a collection of short stories designed for language modeling. The dataset is available at:

TinyStoriesV2-GPT4-train.txt

Ensure the dataset file is placed in the appropriate directory (e.g., data/) before training.
## Notes

- Multiprocessing: The pre-tokenization step leverages multiprocessing to parallelize text processing, making training faster on multi-core systems.
- Storage: Trained models are saved as .pkl files using pickle. If you prefer PyTorch serialization, use torch.save and torch.load instead.
- Special Tokens: Ensure the same special tokens used during training are provided during inference to maintain consistency.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
