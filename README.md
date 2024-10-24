
# Transformer Model

This repository provides an implementation of a Transformer model using PyTorch. The model is designed for sequence-to-sequence learning tasks, such as machine translation, and consists of key components: an Encoder, a Decoder, and several essential layers for effective training.

## Components

- **[components/Head.py](components/Head.py)**: Implements a single attention head, which is a core part of the multi-head attention mechanism.
- **[components/inputEmbedding.py](components/inputEmbedding.py)**: Provides input embeddings for the model, leveraging a pre-trained SentenceTransformer model for better initialization.
- **[components/MultiHead.py](components/MultiHead.py)**: Implements multi-head attention, a crucial part of the self-attention mechanism that allows the model to focus on different parts of the input sequence simultaneously.
- **[components/PositionEmbedding.py](components/PositionEmbedding.py)**: Implements positional embeddings to help the model understand the order of the input tokens, as the Transformer itself is permutation-invariant.
- **[components/positionWiseFeedForward.py](components/positionWiseFeedForward.py)**: Implements the position-wise feed-forward network used in both the encoder and decoder layers.
- **[components/scalarDotProduct.py](components/scalarDotProduct.py)**: Implements scalar dot-product attention, which calculates attention scores between tokens in the sequence.

## Layers

- **[layers/DecoderLayer.py](layers/DecoderLayer.py)**: Implements a single layer of the Transformer decoder, consisting of multi-head attention, layer normalization, and position-wise feed-forward networks.
- **[layers/EncoderLayer.py](layers/EncoderLayer.py)**: Implements a single layer of the Transformer encoder, following the same structure as the decoder but without cross-attention.

## Main Modules

- **[Decoder.py](Decoder.py)**: Implements the Transformer decoder by stacking multiple `DecoderLayer` instances.
- **[Encoder.py](Encoder.py)**: Implements the Transformer encoder by stacking multiple `EncoderLayer` instances.
- **[transformer.py](transformer.py)**: Implements the full Transformer model by combining the encoder and decoder modules into a single architecture.

## Installation

Follow the steps below to set up and run the Transformer model:

1. Clone the repository:
    ```bash
    git clone https://github.com/vinay-852/AttentionIsAllYouNeed.git
    cd AttentionIsAllYouNeed
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use the Transformer model, you can import and initialize the relevant classes, set the necessary parameters, and feed input sequences through the model. Here's an example of how to initialize and run the Transformer:

```python
import torch
from transformer import Transformer

# Example model parameters
N = 6  # Number of layers in encoder and decoder
d_model = 512  # Dimension of the model
num_heads = 8  # Number of attention heads
d_in = 512  # Input dimensionality
d_hid = 2048  # Hidden dimension in the feed-forward networks
output_size = 10000  # Size of the output vocabulary
dropout = 0.1  # Dropout rate

# Initialize the Transformer model
model = Transformer(N, d_model, num_heads, d_in, d_hid, output_size, dropout)

# Example input sequences
src_seq = torch.rand(10, 32, d_model)  # (sequence_length, batch_size, d_model)
tgt_seq = torch.rand(20, 32, d_model)  # (sequence_length, batch_size, d_model)

# Forward pass through the model
output = model(src_seq, tgt_seq)
print(output.shape)  # Expected output shape: (sequence_length, batch_size, output_size)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---