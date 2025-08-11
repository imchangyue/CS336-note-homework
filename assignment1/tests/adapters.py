from __future__ import annotations
import multiprocessing
import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int
import time
import numpy.typing as npt
import torch
from torch import Tensor
import regex as re
from collections import defaultdict, Counter
import heapq
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.Tokenizer import BPETokenizer
from cs336_basics.Linear import Linear
from cs336_basics.Embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLUFeedForward import SwiGLUFeedForward
from cs336_basics.RoPE import RotaryPositionalEmbedding
from cs336_basics.AdamW import AdamW
import math
from einops import einsum,rearrange
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    linear_layer = Linear(d_in, d_out)
    linear_layer.W.data = weights
    return linear_layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."], #……代表的是"batch,sequence_length"
) -> Float[Tensor, " ... d_model"]:
    embedding_layer = Embedding(vocab_size, d_model)
    embedding_layer.embedding_matrix.data = weights
    return embedding_layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    swiglu = SwiGLUFeedForward(d_model, d_ff)
    
    # 加载权重 (注意转置以匹配类内部的维度)
    with torch.no_grad():
        swiglu.W1.copy_(w1_weight.T)  # [d_ff,d_model] -> [d_model,d_ff]
        swiglu.W2.copy_(w2_weight.T)  # [d_model,d_ff] -> [d_ff,d_model]
        swiglu.W3.copy_(w3_weight.T)  # [d_ff,d_model] -> [d_model,d_ff]
    
    # 调用forward方法
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    scores = einsum(Q, K, '... q d_k, ... k d_k -> ... q k')

    # 获取 d_k，用于缩放
    d_k = Q.shape[-1]
    
    # 2. 缩放注意力得分
    scores = scores / (d_k ** 0.5)

    # 3. 应用掩码
    if mask is not None:
        # 将 mask 为 False 的位置填充为 -infinity
        scores = scores.masked_fill(mask == False, float('-inf'))

    # 4. 对最后一个维度进行 softmax，得到注意力权重
    attention_weights = torch.softmax(scores, dim=-1)

    # 5. 将注意力权重与值向量 V 相乘
    # `... queries keys, ... keys d_v -> ... queries d_v`
    output = einsum(attention_weights, V, '... q k, ... k d_v -> ... q d_v')
    return output



def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    # 获取维度信息
    *batch_dims, seq_len, d_in = in_features.shape
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    d_head_k = d_k // num_heads
    d_head_v = d_v // num_heads
    
    # 1. 将输入特征投影到 Q, K, V
    # 这一步一次性为所有头计算出 Q, K, V
    # in_features: `... seq_len d_in`
    # proj_weight: `d_k d_in` 或 `d_v d_in`
    # Q: `... seq_len d_k`, K: `... seq_len d_k`, V: `... seq_len d_v`
    Q = einsum(in_features, q_proj_weight, "... s d_in, d_k d_in -> ... s d_k")
    K = einsum(in_features, k_proj_weight, "... s d_in, d_k d_in -> ... s d_k")
    V = einsum(in_features, v_proj_weight, "... s d_in, d_v d_in -> ... s d_v")

    # 2. 重塑 Q, K, V 以便分离各个头
    # 从 `... seq_len d_k` 变为 `... num_heads seq_len d_head_k`
    Q = Q.view(*batch_dims, seq_len, num_heads, d_head_k).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_head_k).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_head_v).transpose(-3, -2)
    
    # 3. 创建因果掩码（causal mask）
    # 防止注意力机制关注未来的 token。True 表示允许关注。
    # 形状为 `(seq_len, seq_len)`，可以自动广播到 `... num_heads seq_len seq_len`
    mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool))

    # 4. 并行计算所有头的缩放点积注意力
    # Q, K, V 的头维度被当作批处理维度
    # attn_output 形状: `... num_heads seq_len d_head_v`
    attn_output = run_scaled_dot_product_attention(Q, K, V, mask=mask)
    
    # 5. 拼接所有头的注意力输出
    # 从 `... num_heads seq_len d_head_v` 变回 `... seq_len d_v`
    # 首先，换回维度顺序
    attn_output = attn_output.transpose(-3, -2).contiguous()
    # 然后，重塑以合并头维度和 d_head_v 维度
    concatenated_output = attn_output.view(*batch_dims, seq_len, d_v)

    # 6. 应用最终的线性投影
    # concatenated_output: `... seq_len d_v`
    # o_proj_weight:      `d_model d_v`
    # output:             `... seq_len d_model`
    output = einsum(concatenated_output, o_proj_weight, "... s d_v, d_model d_v -> ... s d_model")

    return output



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
    # 确保 d_model 可以被 num_heads 整除
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    # 获取维度信息
    *batch_dims, seq_len, d_in = in_features.shape
    d_k = q_proj_weight.shape[0]
    d_v = v_proj_weight.shape[0]
    d_head_k = d_k // num_heads
    d_head_v = d_v // num_heads

    # 1. 将输入特征投影到 Q, K, V
    Q = einsum(in_features, q_proj_weight, "... s d_in, d_k d_in -> ... s d_k")
    K = einsum(in_features, k_proj_weight, "... s d_in, d_k d_in -> ... s d_k")
    V = einsum(in_features, v_proj_weight, "... s d_in, d_v d_in -> ... s d_v")

    # 2. 重塑 Q, K, V 以便分离各个头
    Q = Q.view(*batch_dims, seq_len, num_heads, d_head_k).transpose(-3, -2)
    K = K.view(*batch_dims, seq_len, num_heads, d_head_k).transpose(-3, -2)
    V = V.view(*batch_dims, seq_len, num_heads, d_head_v).transpose(-3, -2)

    # 3. 对 Q 和 K 应用 RoPE
    # 如果没有提供 token_positions，则创建它
    if token_positions is None:
        token_positions = torch.arange(seq_len, device=in_features.device)
        # 扩展以匹配输入的批处理维度
        view_shape = [1] * len(batch_dims) + [seq_len]
        token_positions = token_positions.view(*view_shape).expand(*batch_dims, seq_len)
    
    # 为 RoPE 广播准备位置张量。
    # `token_positions` 形状: `... seq_len` -> `... 1 seq_len`
    # 这样它就可以在 `num_heads` 维度上广播
    rope_pos = token_positions.unsqueeze(-2)

    Q_roped = run_rope(
        d_k=d_head_k,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=Q,
        token_positions=rope_pos,
    )
    K_roped = run_rope(
        d_k=d_head_k,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=K,
        token_positions=rope_pos,
    )

    # 4. 创建因果掩码
    mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool))
    
    # 5. 使用经过 RoPE 处理的 Q 和 K 计算缩放点积注意力
    attn_output = run_scaled_dot_product_attention(Q_roped, K_roped, V, mask=mask)
    
    # 6. 拼接所有头的注意力输出
    attn_output = attn_output.transpose(-3, -2).contiguous()
    concatenated_output = attn_output.view(*batch_dims, seq_len, d_v)

    # 7. 应用最终的线性投影
    output = einsum(concatenated_output, o_proj_weight, "... s d_v, d_model d_v -> ... s d_model")
    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope_module = RotaryPositionalEmbedding(
        theta=theta, 
        d_k=d_k, 
        max_seq_len=max_seq_len, 
        device=in_query_or_key.device
    )

    # 调用 forward 方法
    return rope_module(in_query_or_key, token_positions)




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
    
    给定一个 pre-norm Transformer 块的权重和输入特征，
    返回在该输入特征上运行 Transformer 块的输出。

    此版本直接调用您提供的 run_... 系列函数。
    """
    # 提取序列长度和设备，用于创建 token_positions
    seq_len = in_features.shape[1]
    device = in_features.device
    
    # --- 第一个子层: 多头自注意力 (Multi-Head Self-Attention) ---
    # 完整计算流: y = x + MHA(RMSNorm(x))
    
    # 1. 对输入进行 RMSNorm
    # 对应于 RMSNorm(x)
    normed_for_attn = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,  # 使用一个标准的 epsilon 值
        weights=weights["ln1.weight"],
        in_features=in_features
    )
    
    # 2. 执行带有 RoPE 的多头自注意力
    # 对应于 MHA(...)
    # 注意: RoPE 需要知道每个 token 的绝对位置
    token_positions = torch.arange(seq_len, device=device).unsqueeze(0)
    
    attn_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        in_features=normed_for_attn,
        token_positions=token_positions
    )
    
    # 3. 添加第一个残差连接
    # 对应于 x + ...
    residual_after_attn = in_features + attn_output
    
    # --- 第二个子层: 前馈网络 (Feed-Forward Network) ---
    # 完整计算流: z = y + FFN(RMSNorm(y))
    
    # 4. 对第一个子层的输出进行 RMSNorm
    # 对应于 RMSNorm(y)
    normed_for_ffn = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln2.weight"],
        in_features=residual_after_attn
    )
    
    # 5. 执行 SwiGLU 前馈网络
    # 对应于 FFN(...)
    ffn_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=weights["ffn.w1.weight"],
        w2_weight=weights["ffn.w2.weight"],
        w3_weight=weights["ffn.w3.weight"],
        in_features=normed_for_ffn
    )
    
    # 6. 添加第二个残差连接
    # 对应于 y + ...
    final_output = residual_after_attn + ffn_output
    
    return final_output


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
    x = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=weights["token_embeddings.weight"],
        token_ids=in_indices
    )

    # 步骤 2: 依次通过 N 个 Transformer Block
    # 这是一个循环，依次处理 num_layers 个 Transformer 层
    for i in range(num_layers):
        # 为当前的 Transformer Block 准备其专属的权重字典
        block_weights = {}
        prefix = f"layers.{i}."
        for key, value in weights.items():
            if key.startswith(prefix):
                # 移除前缀 (例如 'layers.0.ln1.weight' -> 'ln1.weight')
                # 以便 run_transformer_block 函数能够识别
                new_key = key[len(prefix):]
                block_weights[new_key] = value

        # 调用我们之前实现的 run_transformer_block 函数
        x = run_transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length, # max_seq_len 对应于 context_length
            theta=rope_theta,
            weights=block_weights,
            in_features=x
        )

    # 步骤 3: 最终的层归一化 (Final Norm)
    # 在所有 Transformer Block 之后，进行最后一次归一化
    # Transformer Block -> Norm
    x = run_rmsnorm(
        d_model=d_model,
        eps=1e-5,
        weights=weights["ln_final.weight"],
        in_features=x
    )

    # 步骤 4: 线性输出层 (LM Head)
    # 将最终的隐藏状态映射到词汇表空间，得到 logits
    # Norm -> Linear (Output Embedding)
    logits = run_linear(
        d_in=d_model,
        d_out=vocab_size,
        weights=weights["lm_head.weight"],
        in_features=x
    )
    
    # 步骤 5: 返回 Logits
    # 根据函数文档要求，返回未归一化的 logits，而不是经过 Softmax 的概率
    # 所以我们在这里直接返回 logits
    return logits


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
    # 初始化 RMSNorm 模块
    rmsnorm = RMSNorm(d_model=d_model, eps=eps)
    
    # 将增益参数设置为给定的权重
    rmsnorm.gain.data = weights
    
    # 执行 RMSNorm 操作
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    # SiLU(x) = x * sigmoid(x)
    return in_features * torch.sigmoid(in_features)


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
    # 获取数据集的总长度
    dataset_len = len(dataset)
    
    # 1. 随机生成 `batch_size` 个起始位置的索引
    # 一个完整的样本需要 `context_length + 1` 个 token (输入 + 1个未来token)。
    # 因此，最后一个可能的起始位置是 `dataset_len - context_length - 1`。
    # `torch.randint` 的 `high` 参数是上界（不包含），所以 `high` 设置为 `dataset_len - context_length`。
    start_indices = torch.randint(low=0, high=dataset_len - context_length, size=(batch_size,))
    
    # 2. 使用向量化的方式一次性构建所有输入和目标序列的索引
    # `offsets` 是一个行向量: [0, 1, 2, ..., context_length-1]
    offsets = torch.arange(context_length)
    
    # `start_indices.unsqueeze(1)` 将起始索引变为一个列向量。
    # 利用 PyTorch 的广播机制，将列向量和行向量相加，直接得到一个索引矩阵。
    # 每一行都是一个完整的输入序列的索引。
    input_indices = start_indices.unsqueeze(1) + offsets
    
    # 目标序列的索引就是输入序列的索引全部加 1
    target_indices = input_indices + 1
    
    # 3. 从数据集中提取数据
    # 为了使用 PyTorch 的高级索引，先将 numpy 数组转换为 tensor
    dataset_tensor = torch.from_numpy(dataset)
    
    # 直接使用索引矩阵从数据张量中高效地抓取所有数据
    inputs = dataset_tensor[input_indices]
    targets = dataset_tensor[target_indices]
    
    # 4. 将张量移动到指定的设备，并确保类型为 LongTensor
    inputs = inputs.to(device, dtype=torch.long)
    targets = targets.to(device, dtype=torch.long)
    
    return inputs, targets


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
    max_vals, _ = torch.max(in_features, dim=dim, keepdim=True)
    stabilized_in_features = in_features - max_vals
    
    # Calculate the exponentiated values
    exp_vals = torch.exp(stabilized_in_features)
    
    # Sum the exponentiated values along the specified dimension
    exp_sum = torch.sum(exp_vals, dim=dim, keepdim=True)
    
    # Clamp the sum to avoid dividing by zero
    exp_sum = torch.clamp(exp_sum, min=1e-8)
    
    # Return the softmax normalized values
    return exp_vals / exp_sum



def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
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
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values

    # Step 2: Calculate the log of the sum of exponentials (the denominator of the softmax).
    # We use the log-sum-exp trick formula: log(sum(exp(x_i))) = c + log(sum(exp(x_i - c)))
    # where c is the maximum value of x_i.
    # `stable_inputs` has shape [batch_size, vocab_size].
    stable_inputs = inputs - max_logits
    # `log_sum_exp` has shape [batch_size].
    log_denominator = max_logits.squeeze(-1) + torch.log(torch.sum(torch.exp(stable_inputs), dim=-1))

    # Step 3: Get the logit score for the correct target class for each example.
    # `targets.unsqueeze(-1)` changes the shape from [batch_size] to [batch_size, 1] for `gather`.
    # `gather` selects the elements from `inputs` along dim 1 using the indices from `targets`.
    # `squeeze(-1)` changes the shape back to [batch_size].
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Step 4: Compute the cross-entropy loss for each example using the simplified formula.
    # `loss` will have shape [batch_size].
    loss_per_example = log_denominator - target_logits

    # Step 5: Return the average loss across the batch.
    return loss_per_example.mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    # Clip gradients if norm exceeds max_l2_norm
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + 1e-6)  # Adding epsilon for numerical stability
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


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
    if it < warmup_iters:
        # Linear warm-up
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing
        cosine_factor = 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
        return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)
    else:
        # Post-annealing
        return min_learning_rate

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
    # 创建一个字典来存储所有需要保存的状态
    checkpoint_state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    
    # 使用 torch.save 将字典保存到指定的文件或文件对象
    torch.save(checkpoint_state, out)


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
    # 使用 torch.load 从指定的文件或文件对象加载检查点字典
    checkpoint_state = torch.load(src)
    
    # 使用 load_state_dict 方法恢复模型和优化器的状态
    model.load_state_dict(checkpoint_state['model_state_dict'])
    optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    
    # 返回保存的迭代次数
    return checkpoint_state['iteration']


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return BPETokenizer(vocab, merges, special_tokens)
    # """Given a vocabulary, a list of merges, and a list of special tokens,
    # return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    # Args:
    #     vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
    #         to bytes (token bytes)
    #     merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
    #         representing that <token1> was merged with <token2>.
    #         Merges are ordered by order of creation.
    #     special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
    #         be split into multiple tokens, and will always be kept as a single token.

    # Returns:
    #     A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    # """
    # raise NotImplementedError




# 我现在考虑在计算相邻两个byte的频率的时候，把token的id也存下来
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 读取数据集
    delimiter_pattern = "|".join(re.escape(s) for s in special_tokens)
    with open(input_path, 'r', encoding='utf-8') as f:
        data = f.read()
    parts = re.split(delimiter_pattern, data)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_number = {}

    for part in parts:
        part_tokens = re.findall(PAT, part)
        for token in part_tokens:
            token_bytes = tuple(token.encode('utf-8'))  # 直接转换为字节元组
            token_number[token_bytes] = token_number.get(token_bytes, 0) + 1  # 更新频率


    # 初始化词汇表
    vocab = {i: bytes([i]) for i in range(256)}  # 直接初始化单字节字符
    current_id = 256
    # 添加特殊标记到词汇表
    for token_str in special_tokens:
        vocab[current_id] = token_str.encode("utf-8")
        current_id += 1

    # 频率统计
    freq_dict = {}
    merges = []
    freq_dict = Counter()  # 重新统计频率
    # pair_to_token_ids = {}
    for token,count in token_number.items():
        number = len(token) - 1
        for i in range (number):
            pair = (token[i], token[i + 1])  # 生成字节对
            freq_dict[pair] += count # 累加频率
            # if pair not in pair_to_token_ids:
            #     pair_to_token_ids[pair] = set()
            # pair_to_token_ids[pair].add(i)  # 保存字节对对应的 token id
    # 现在是初始的累加频率
    while current_id < vocab_size:
        print(f"Current vocab size: {len(vocab)}, current_id: {current_id}, merges: {len(merges)}")
        # 找到频率最高的字符对
        max_pair = max(freq_dict.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]))
        # 将最高频的字符对添加到 merges 列表
        merges.append((vocab[max_pair[0][0]], vocab[max_pair[0][1]]))
        freq_dict[max_pair[0]] = 0
        # 将合并后的字符对添加到 vocab
        vocab[current_id] = vocab[max_pair[0][0]] + vocab[max_pair[0][1]]
        current_id += 1

        # 更新 token，将频率最高的字符对替换为新合并的 token
        new_tokens = {}
        for token,count in token_number.items():
            updated_token = []
            i = 0
            number = len(token) - 1
            while i < number:
                pair = (token[i], token[i + 1])
                if pair == max_pair[0]:
                    updated_token.append(current_id - 1)  # 替换为合并后的 token
                    #此时需要更新的大小是(如果i > 0)，pair(token[i - 1],token[i])--，pair(token[i - 1],current_id - 1)++
                    #还有就是(如果i+1 <number)，pair(token[i + 1],token[i + 2])--,pair(current_id - 1,token[i + 2])++
                    if i > 0:
                        freq_dict[(token[i - 1],token[i])]-=count
                        freq_dict[(token[i - 1],current_id - 1)] = freq_dict.get((token[i - 1],current_id - 1), 0) + count
                    if i+1 <number:
                        freq_dict[(token[i + 1],token[i + 2])]-=count
                        freq_dict[(current_id - 1,token[i + 2])] = freq_dict.get((current_id - 1,token[i + 2]), 0) + count
                    i += 2  # 此时要更新freq_dict的大小
                else:
                    updated_token.append(token[i])
                    i += 1

            # 处理最后一个字符
            if i < len(token):
                updated_token.append(token[i])

            new_tokens[tuple(updated_token)] = count

        # 更新 token_number 为新的 token
        token_number = new_tokens


    return vocab, merges

