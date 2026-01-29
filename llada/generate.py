# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import torch.nn.functional as F
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
    layer_skip: bool = False,
    layer_cos_thr: float = 0.97,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        threshold: If set, uses confidence threshold instead of scheduled transfer counts.
        factor: If set, uses dynamic transfer scheduling.
        layer_skip/layer_cos_thr: Layer-level compute skipping controls (passed to the model).
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    # ---- Stats for accuracy-vs-FLOPs (proxy) curves ----
    total_layers = 0
    total_layers_skipped = 0

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)

            out = model(x, layer_skip=layer_skip, layer_cos_thr=layer_cos_thr)
            logits = out.logits

            # Accumulate layer-skip stats if available
            if hasattr(model, "_last_layer_total") and hasattr(model, "_last_layer_skipped"):
                total_layers += int(getattr(model, "_last_layer_total"))
                total_layers_skipped += int(getattr(model, "_last_layer_skipped"))

            # only allow filling within current block (everything after current block masked out)
            mask_index[:, block_end:] = 0

            if factor is None:
                # FIX: if the while-loop runs longer than `steps`, i can exceed the schedule width.
                # Fall back to transferring all remaining masked tokens in the current mask_index.
                if threshold is None:
                    if i < num_transfer_tokens.shape[1]:
                        quota = num_transfer_tokens[:, i]
                    else:
                        quota = mask_index.sum(dim=1)  # remaining masked tokens per batch item
                else:
                    quota = None

                x0, transfer_index = get_transfer_index(
                    logits, temperature, remasking, mask_index, x, quota, threshold
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x, None, factor
                )

            x[transfer_index] = x0[transfer_index]
            i += 1

            # done when the current block contains no masks
            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

    stats = {
        "layers_total": int(total_layers),
        "layers_skipped": int(total_layers_skipped),
        "flops_reduction": float(total_layers_skipped / max(1, total_layers)),
    }
    return x, nfe, stats


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
):
    """
    Cached generation path. (Layer skipping should NOT be used here because KV-cache correctness.)
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)

        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i_layer in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i_layer])):
                new_past_key_values[i_layer] += (past_key_values[i_layer][j][:, :, :current_block_start],)

        past_key_values = new_past_key_values
        nfe += 1

        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break

            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if factor is None:
                # FIX: guard i beyond schedule width
                if threshold is None:
                    if i < num_transfer_tokens.shape[1]:
                        quota = num_transfer_tokens[:, i]
                    else:
                        quota = mask_index.sum(dim=1)
                else:
                    quota = None

                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:],
                    quota,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x[:, current_block_start:], None, factor
                )

            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


@torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
    factor=None,
):
    """
    Dual-cache generation path. (Layer skipping should NOT be used here because KV-cache correctness.)
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True, replace_position=current_block_start)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0

        if factor is None:
            x0, transfer_index = get_transfer_index(
                output.logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, 0] if threshold is None else None,
                threshold,
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)

        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i_layer in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i_layer])):
                new_past_key_values[i_layer] += (past_key_values[i_layer][j][:, :, :current_block_start],)

        past_key_values = new_past_key_values
        nfe += 1

        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break

            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(
                x[:, current_block_start:],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=current_block_start,
            ).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if factor is None:
                # FIX: guard i beyond schedule width
                if threshold is None:
                    if i < num_transfer_tokens.shape[1]:
                        quota = num_transfer_tokens[:, i]
                    else:
                        quota = mask_index.sum(dim=1)
                else:
                    quota = None

                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x[:, current_block_start:],
                    quota,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_index, x[:, current_block_start:], None, factor
                )

            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe


def add_gumbel_noise(logits, temperature=0.0, eps=1e-10):
    if temperature == 0:
        return logits
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    return logits + temperature * g


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    cols = torch.arange(steps, device=mask_num.device).unsqueeze(0)  # (1, steps)
    num_transfer_tokens = torch.ceil(mask_num * (cols + 1) / steps).long()
    num_transfer_tokens[:, 1:] = num_transfer_tokens[:, 1:] - num_transfer_tokens[:, :-1]
    return num_transfer_tokens


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == "low_confidence":
        # MPS does not support float64; float32 is plenty for softmax here.
        p = F.softmax(logits.to(torch.float32), dim=-1)
        confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
    elif remasking == "random":
        confidence = torch.rand_like(x0, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)

    if threshold is not None:
        transfer_index = (confidence > threshold) & mask_index
    else:
        # num_transfer_tokens is a (B,) tensor; this implementation assumes batch size 1 in most scripts.
        _, transfer_index = torch.topk(confidence, k=num_transfer_tokens.item(), dim=-1)
        transfer_index = torch.scatter(torch.zeros_like(x0, dtype=torch.bool), dim=-1, index=transfer_index, value=True)
        transfer_index = transfer_index & mask_index

    x0 = torch.where(mask_index, x0, x)
    return x0, transfer_index


def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float32), dim=-1)
        confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
    elif remasking == "random":
        confidence = torch.rand_like(x0, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)

    for j in range(confidence.shape[0]):
        conf = confidence[j][mask_index[j]]
        num_transfer_tokens = int(np.round(conf.shape[0] / (factor)))
        _, transfer_index = torch.topk(conf, k=num_transfer_tokens, dim=-1)
        transfer_index = torch.scatter(torch.zeros_like(conf, dtype=torch.bool), dim=-1, index=transfer_index, value=True)
        confidence[j][mask_index[j]] = transfer_index

    transfer_index = (confidence > 0.0) & mask_index

    x0 = torch.where(mask_index, x0, x)
    return x0, transfer_index


def log_prob_sum(logits, x0, temperature=0.0):
    if temperature == 0:
        logprob = F.log_softmax(logits.to(torch.float32), dim=-1)
        logp = torch.gather(logprob, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
    else:
        probs = F.softmax(logits.to(torch.float32) / temperature, dim=-1)
        logp = torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(-1)
        logp = torch.log(logp)
    return logp.sum()


def get_transfer_index_parallel(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float32), dim=-1)
        confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
    elif remasking == "random":
        confidence = torch.rand_like(x0, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)

    if threshold is not None:
        transfer_index = (confidence > threshold) & mask_index
    else:
        _, transfer_index = torch.topk(confidence, k=num_transfer_tokens.item(), dim=-1)
        transfer_index = torch.scatter(torch.zeros_like(x0, dtype=torch.bool), dim=-1, index=transfer_index, value=True)
        transfer_index = transfer_index & mask_index

    x0 = torch.where(mask_index, x0, x)
    return x0, transfer_index
