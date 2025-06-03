from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor

# Tokenizer Class
import json
from itertools import islice

# Tokenizer Training
import regex as re
from collections import Counter
from multiprocessing import Pool
from functools import reduce
import operator
from itertools import pairwise
from dataclasses import dataclass

# Transformer modules
import torch.nn as nn
from einops import rearrange, einsum
import numpy as np


# Linear
class Linear(nn.Module):
    """Implementation of torch Linear from scratch"""
    def __init__(self, in_features, out_features, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        std = np.sqrt(2.0 / (in_features + out_features))
        weights = torch.zeros((out_features, in_features), dtype= dtype, device=device)
        nn.init.trunc_normal_(weights, std=1.0, a=-3.0*std, b=3.0*std)
        self.weights = nn.Parameter(weights)

    def forward(self, x: Tensor):
        return einsum(self.weights, x, "d_out d_in, ... d_in -> ... d_out")

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to
    
    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    linear = Linear(d_in, d_out)
    linear.weights = nn.Parameter(weights)
    return linear.forward(in_features)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        embeddings = torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)
        nn.init.trunc_normal_(embeddings, std=1.0, a=-3.0, b=3.0)
        self.embeddings = nn.Parameter(embeddings)
        
    def forward(self, token_ids: Tensor) -> Tensor:
        embeddings = self.embeddings[token_ids]
        return embeddings

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embedding = Embedding(vocab_size, d_model)
    embedding.embeddings = nn.Parameter(weights)
    return embedding.forward(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\Theta$ parameter.
        weights (dict[str, Tensor]): 
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    raise NotImplementedError


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    raise NotImplementedError


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    raise NotImplementedError

# For reaL prod environment, will have to couple vocab/ bytes_to_ids and special_tokens/ special_token_bytes, or ban rerassignments completely
# For special tokens, if there are multiple matches starting from a position, we will match with the longer token
# e.g. byte sequence = "aabbccdd", special tokens = "aab", "aabbcc", then we will match with the longer special token "aabbcc"
# We assume no overlapping between special tokens and 
class BPETokenizer:
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        
        # "Normal" Vocabulary
        self.vocab: dict[int, bytes] = vocab
        self.bytes_to_ids: dict[bytes, int] = {v: k  for k, v in vocab.items()}
        self.max_normal_vocab_length: int = max(len(v) for v in vocab.values())
        
        # Merges
        self.merges: list[tuple[bytes, bytes]] = merges
        
        # Special tokens
        self.special_tokens: list[str] = special_tokens
        self.special_tokens_bytes: list[bytes] = [token.encode('utf-8') for token in special_tokens]
        self.special_tokens_bytes_lengths = set(len(byte_sequence) for byte_sequence in self.special_token_bytes)
    
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        with open(vocab_filepath) as f:
            vocab_data = json.load(f)
        
        with open(merges_filepath) as f:
            merges_data = json.load(f)
            
        vocab = {int(k): v.encode('utf-8') for k, v in vocab_data.items()}
        merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merges_data]
        
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        text_bytes: bytes = text.encode('utf-8')
        
        # Create merges
        text_bytes_list: list[bytes] = [b for b in text_bytes]
        
        i = 0
        while i < len(text_bytes_list):
        # Check if current position starts a merge
            if i < len(text_bytes_list) - 1:
                # Match with special tokens or merges, we increment from a position once no actions is 
                # "Special Tokens"
                matchable_special_token_lengths = set(length for length in self.special_tokens_bytes_lengths if i + length <= len(text_bytes_list))
                for length in sorted(matchable_special_token_lengths, reverse=True):
                    candidate: bytes = b''.join(text_bytes_list[i: i+length])
                    if candidate in self.vocab.keys():
                        text_bytes_list = text_bytes_list[:i] + [candidate] + text_bytes_list[(i + length) + 1:]
                # "Normal" merging
                pair = (text_bytes_list[i], text_bytes_list[i + 1])
                if pair in self.merges:
                    text_bytes_list = text_bytes_list[:i] + [pair[0] + pair[1]] + text_bytes_list[(i + 1) + 1:]
                # No merges
                else:
                    i += 1
            else:
                i += 1 # Edge case for when i = len(text_bytes_list) - 1 (we arrived to the last byte), prevent infinite looping
        
        return [self.bytes_to_ids[byte] for byte in text_bytes_list]

    
    def encode_iterable(self, iterable: Iterable[str], chunk_length: int = 40000) -> Iterable[int]:
        """
        Encode an iterable of strings into token IDs, processing in chunks to save memory.
        
        Args:
            iterable (Iterable[str]): An iterable of strings (for lazy loading)
            chunk_length (int, optional): Amount of string items to load every time we exhaust our chunk. Defaults to 40000.

        Returns:
            Iterable[int]: An iterable of token IDs
        """
        max_vocab_length = max(self.max_normal_vocab_length, max(self.special_tokens_bytes_lengths) if self.special_tokens_bytes_lengths else 0)
        
        # Buffer to hold bytes that might span across chunks
        buffer = b''
        
        # Process the iterable in chunks
        for chunk_strings in self._chunk_iterable(iterable, chunk_length):
            # Combine the buffer with the new chunk of strings
            chunk_bytes = buffer + ''.join(chunk_strings).encode('utf-8')
            
            # Process all complete tokens in the chunk
            processed_length = 0
            text_bytes_list = [b for b in chunk_bytes]
            
            # Apply merges as in the encode method
            i = 0
            while i < len(text_bytes_list) - max_vocab_length + 1:
                # Check for special tokens first
                matched = False
                matchable_special_token_lengths = [length for length in self.special_tokens_bytes_lengths 
                                                if i + length <= len(text_bytes_list)]
                
                for length in sorted(matchable_special_token_lengths, reverse=True):
                    candidate = b''.join(text_bytes_list[i:i+length])
                    if candidate in self.special_tokens_bytes:
                        # Found a special token
                        token_id = self.bytes_to_ids[candidate]
                        yield token_id
                        i += length
                        matched = True
                        processed_length = i
                        break
                
                if matched:
                    continue
                    
                # Check for normal merges
                if i < len(text_bytes_list) - 1:
                    pair = (text_bytes_list[i], text_bytes_list[i + 1])
                    if pair in self.merges:
                        merged = pair[0] + pair[1]
                        text_bytes_list[i:i+2] = [merged]
                        # Don't increment i here to allow further merges
                        continue
                
                # If we get here, no merges were possible
                # Yield the token if it's in the vocabulary
                token = text_bytes_list[i]
                if token in self.bytes_to_ids:
                    yield self.bytes_to_ids[token]
                    processed_length = i + 1
                
                i += 1
            
            # Save the unprocessed part for the next chunk
            buffer = chunk_bytes[processed_length:]
        
        # Process any remaining bytes in the buffer
        if buffer:
            for token_id in self.encode(buffer.decode('utf-8', errors='replace')):
                yield token_id

    def _chunk_iterable(self, iterable: Iterable[str], chunk_size: int) -> Iterable[list[str]]:
        """
        Helper method to chunk an iterable into lists of specified size.
        
        Args:
            iterable: The input iterable
            chunk_size: Size of each chunk
            
        Returns:
            Iterable of lists containing chunks of the original iterable
        """
        iterator = iter(iterable)
        while True:
            chunk = list(islice(iterator, chunk_size))
            if not chunk:
                break
            yield chunk
        
    def decode(self, ids: list[int]) -> str:
        word_bytes: bytes = b''.join([self.vocab[word_id] for word_id in ids])
        return word_bytes.decode("utf-8")

def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPETokenizer(vocab, merges, special_tokens)


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_tokenize(text: str, pattern: str) -> Counter[str]:
    return Counter(match.group().strip() for match in re.finditer(pattern, text))

def process_chunk(input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str], pattern: str) -> Counter:
    """
    Process a single chunk of text. args: Tuple[int, int, str, List[str], str]
    
    Args:
        start (int): Start position of chunk
        end (int): End position of chunk
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        special_tokens: List of special tokens to remove
        pattern: Regex pattern for pre-tokenization
    
    Returns:
        Counter of pre-tokenized words with special tokens removed
    """
    
    # Read the chunk
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    # Remove special tokens
    for token in special_tokens:
        chunk = chunk.replace(token, " ")
    
    # Pre-tokenize and return counts
    return pre_tokenize(chunk, pattern)

@dataclass
class MergeCandidate:
    token_ids: tuple[int, int]
    bytes_data: bytes

def count_token_sequences(word_counter: Counter[str], word_to_ids: dict[str, list[int]]) -> Counter[tuple[int, int]]:
    sequence_count: Counter[tuple[int, int]] = Counter()
    for word, count in word_counter.items():
        token_ids = word_to_ids[word]
        for a, b in pairwise(token_ids):
            sequence_count[(a,b)] += count
    return sequence_count

def count_token_sequences_parallel(
    word_counter: Counter[str], 
    word_to_ids: dict[str, list[int]], 
    num_processes: int
) -> Counter[tuple[int, int]]:
    """
    Count token sequences in parallel.
    
    Args:
        word_counter: Counter mapping words to their frequencies
        word_to_ids: Dictionary mapping words to their token ID sequences
        num_processes: Number of processes to use
    
    Returns:
        Counter of token pair frequencies
    """
    # Split words into chunks for parallel processing
    all_words = list(word_counter.keys())
    chunk_size = max(1, len(all_words) // num_processes)
    word_chunks = [all_words[i:i + chunk_size] for i in range(0, len(all_words), chunk_size)]
    
    # Prepare arguments for parallel processing
    chunk_data = [(chunk, word_to_ids, word_counter) for chunk in word_chunks]
    
    # Process in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(count_token_sequences, chunk_data)
    
    # Combine results
    combined_count: Counter[tuple[int, int]] = reduce(operator.add, results, Counter())
    
    return combined_count

def replace_merged_tokens(
    word: str, 
    token_ids: list[int], 
    merge_sequences: dict[tuple[int, int], int]
) -> tuple[str, list[int]]:
    """
    Replace merged token pairs with new tokens in a word's token ID sequence.
    
    Args:
        word: The original word string
        token_ids: List of token IDs for the word
        merge_sequences: Dictionary mapping token pairs to their merged token ID
    
    Returns:
        Tuple containing:
            - word: The original word string (unchanged)
            - new_token_ids: Updated list of token IDs with merges applied
    """
    i = 0
    new_token_ids = []
    
    while i < len(token_ids):
        # Check if current position starts a merge
        if i < len(token_ids) - 1:
            pair = (token_ids[i], token_ids[i + 1])
            if pair in merge_sequences:
                # Replace with merged token
                new_token_ids.append(merge_sequences[pair])
                i += 2  # Skip both tokens that were merged
                continue
        
        # No merge at this position, keep the token
        new_token_ids.append(token_ids[i])
        i += 1
    
    return word, new_token_ids

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.
            
    Keyword Args:
        num_processes (int, optional): Number of processes to use for parallel processing.
            Defaults to 8.
        pre_tokenize_pattern (str, optional): Regex pattern to be used for pre_tokenization.
            Defaults to the GPT-2 tokenizer found [here](github.com/openai/tiktoken/pull/234/files:)
        verbose (bool, optional): Whether to print the tokenization progression or not
            Defaults to False 

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_processes: int = kwargs.get("num_processes", 8)
    PAT: str = kwargs.get("pre_tokenize_pattern", r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    # Pre-Tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))
            
    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # Create a tuple with arguments in the order expected by process_chunk
        chunk_args.append((input_path, start, end, special_tokens, PAT))
    
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        counters = pool.starmap(process_chunk, chunk_args)
        
    combined_counter: Counter[str] = reduce(operator.add, counters, Counter())
    
    # Tokenization loop
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    byte_to_id: dict[bytes, int] = {bytes([i]): i for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    
    # Vocab sizes
    target_vocab_size: int = vocab_size - len(special_tokens)
    
    # Convert words to sequences of token IDs
    word_to_ids: dict[str, list[int]] = {}
    word_counts: dict[str, int] = {}
    
    for word, count in combined_counter.items():
        word_bytes = word.encode('utf-8')
        word_counts[word] = count
        word_to_ids[word] = [byte_to_id[bytes([b])] for b in word_bytes] # OK because no merging occurred yet
    
    
    while len(vocab) < target_vocab_size:
        
        if kwargs.get("verbose", False):
            print(f"Vocabulary size: {len(vocab)}/{target_vocab_size}")
        
        # Produce sequence counts (Parallelarizable)
        sequence_count: Counter[tuple[int, int]] = count_token_sequences_parallel(combined_counter, word_to_ids, num_processes)
        
        # Token merge loop
        # The goal is to merge multiple pairs if they "have nothing to do with each other" 
        merged_ids: set[int] = set()
        merge_sequences: dict[tuple[int, int], int] = {}
        end_current_merge_iter: bool = False
        
        # Find all sequence of tokens for merging
        while not end_current_merge_iter:
            # Get candidates
            max_count = max(sequence_count.values())
            candidates: list[MergeCandidate] = []
            for item, count in sequence_count.most_common():
                if count == max_count:
                    candidates.append(MergeCandidate(item, vocab[item[0]] + vocab[item[1]]))
                else:
                    break
            candidates = sorted(candidates, key= lambda x: x.bytes_data) # now in lexicographic order
            
            # Trimming candidates list in case it could exceed the target vocabulary size
            if len(candidates) > target_vocab_size - len(vocab):
                candidates = candidates[:target_vocab_size - len(vocab)]
            
            for candidate in candidates:
                # Check if candidate contains IDs that are involved in merges in this cycle 
                if candidate.token_ids[0] in merged_ids or candidate.token_ids[1] in merged_ids:
                    end_current_merge_iter = True
                    break
                else:
                    merged_ids.update(candidate.token_ids) # Add current token ids to pairs
                    new_token_id = len(vocab)
                    vocab[new_token_id] = candidate.bytes_data
                    byte_to_id[candidate.bytes_data] = new_token_id
                    merges.append((vocab[candidate.token_ids[0]],vocab[candidate.token_ids[1]]))
                    merge_sequences[candidate.token_ids] = new_token_id
                    sequence_count.pop(candidate.token_ids)
            
        # Replace merged tokens with the new token for every word (Parallelarizable)
        if merge_sequences:  # Only proceed if we have merges to apply
            # Prepare arguments for parallel processing
            word_data = [(word, token_ids, merge_sequences) 
                         for word, token_ids in word_to_ids.items()]
            
            # Process in parallel using starmap instead of map
            with Pool(processes=num_processes) as pool:
                results = pool.starmap(replace_merged_tokens, word_data)
            
            # Update word_to_ids with the results
            word_to_ids = dict(results)
        else:
            break
    
    # Add special tokens to vocabulary
    for special_token in special_tokens:
        token_bytes = special_token.encode('utf-8')
        if token_bytes not in byte_to_id:
            token_id = len(vocab)
            vocab[token_id] = token_bytes
            byte_to_id[token_bytes] = token_id
    
    return vocab, merges
        
                    