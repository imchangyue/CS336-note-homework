import torch
import torch.nn as nn
import torch.nn.init as init

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Constructs an embedding module.
        Args:
            num_embeddings: int, the size of the vocabulary (vocab_size)
            embedding_dim: int, the dimension of the embedding vectors (d_model)
            device: torch.device | None, device to store the parameters on
            dtype: torch.dtype | None, data type of the parameters
        """
        super(Embedding, self).__init__()
        
        # Initialize the embedding matrix (vocab_size, d_model)
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        
        # Initialize the weights with truncated normal distribution
        init.trunc_normal_(self.embedding_matrix, mean=0, std=1.0 / embedding_dim**0.5, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Perform the embedding lookup for the given token IDs.
        Args:
            token_ids: torch.Tensor, tensor of shape (batch_size, sequence_length), the token IDs to fetch embeddings for
        
        Returns:
            torch.Tensor: The embeddings for the input token IDs, shape (batch_size, sequence_length, embedding_dim)
        """
        # Use token_ids to index into the embedding matrix and get the embeddings
        return self.embedding_matrix[token_ids]
