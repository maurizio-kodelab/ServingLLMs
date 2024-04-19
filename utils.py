import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch
import torch.nn.functional as F


def recursive_tensor_device_check(obj, device):
    if ((obj!=None) and isinstance(obj, (list, tuple))):
        for element in obj:
            recursive_tensor_device_check(element, device)

    if ((obj!=None) and isinstance(obj, dict)):
        for k,v in obj.items():
            recursive_tensor_device_check(v, device)
    if isinstance(obj, torch.Tensor):
        if obj.device != device:
            raise ValueError(f"Tensor {obj} on device {obj.device}")




def get_next_inputs(batch, next_token_ids, past_key_values, next_tokens, device='cpu'):
    
    return {
        "input_ids": next_token_ids.reshape((-1, 1)),  # '-1' here means the remaining elements for this dim
        "position_ids": batch["position_ids"][:, -1].unsqueeze(-1) + 1,  # increment last, discard the rest
        "attention_mask": torch.cat([
            batch["attention_mask"],
            torch.ones((next_token_ids.shape[0], 1), device=device),  # concatenate vector of 1's with shape [batch_size]
        ], dim=1),
        "past_key_values": past_key_values,
        "responses": [r1 + r2 for r1, r2 in zip(batch["responses"], next_tokens)],
        "tokens_remaining": [v - 1 for v in batch["tokens_remaining"]],
    }


def init_batch(tokenizer, requests, device='cpu'):
    prompts = [r[0] for r in requests]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    
    attention_mask = inputs["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    
    return {
        "position_ids": position_ids,
        "responses": copy.copy(prompts),
        "tokens_remaining": [r[1] for r in requests],
        **inputs
    }



def generate_batch_tokens_with_past(model, inputs, device='cpu'):
    
    # Check that inputs are all on same devide
    #print("Generating from model, checking inputs....")
    #recursive_tensor_device_check(inputs, device)
    #print("Iinputs are ok.")
    

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    last_logits = logits[:, -1, :]
    next_token_ids = last_logits.argmax(dim=1)
    
    #print("Checking generated tensors....")
    #recursive_tensor_device_check(outputs, device)
    #print("Generated tensors are correct.")
    return next_token_ids.to(device), outputs.past_key_values


def generate_next_token(model, tokenizer, batch, device='cpu'):
    inputs = copy.copy(batch)
    inputs.pop("responses")
    inputs.pop("tokens_remaining")
    
    next_token_ids, past_key_values = generate_batch_tokens_with_past(model, inputs, device=device)
    next_tokens = tokenizer.batch_decode(next_token_ids)
    return get_next_inputs(batch, next_token_ids, past_key_values, next_tokens, device=device)


def merge_batches(batch1, batch2, device='cpu'):
    # first find the max sequence length of the two batches
    # this can be obtained from the second dimension of the attention mask
    attn_mask1 = batch1["attention_mask"]
    attn_mask2 = batch2["attention_mask"]
    max_seq_len = max(attn_mask1.shape[1], attn_mask2.shape[1])
    
    # pad each mask (on the left) to the max sequence length
    # attention mask uses 0 for padding
    padding1 = max_seq_len - attn_mask1.shape[1]
    padding2 = max_seq_len - attn_mask2.shape[1]
    attn_mask1 = F.pad(attn_mask1, (padding1, 0), "constant", 0)
    attn_mask2 = F.pad(attn_mask2, (padding2, 0), "constant", 0)
    
    # because we only append batches post decoding, we don't need to pad input_ids
    # or position_ids. these are always length 1 in the sequence dimension
    # however, we do need to pad the past_key_values, which have shape:
    # [batch_size, num_heads, sequence_length, head_dim]
    past_kv1 = batch1["past_key_values"]
    past_kv2 = batch2["past_key_values"]
    
    padded_kv1 = []
    for i in range(len(past_kv1)):
        k, v = past_kv1[i]
        k = F.pad(k, (0, 0, padding1, 0), "constant", 0).to(device)
        v = F.pad(v, (0, 0, padding1, 0), "constant", 0).to(device)     
        padded_kv1.append((k, v))
    
    padded_kv2 = []
    for i in range(len(past_kv2)):
        k, v = past_kv2[i]
        k = F.pad(k, (0, 0, padding2, 0), "constant", 0).to(device)
        v = F.pad(v, (0, 0, padding2, 0), "constant", 0).to(device)     
        padded_kv2.append((k, v))
        
    # now that everything has been padded to have consistent shapes, let's merge
    input_ids = torch.concat([batch1["input_ids"], batch2["input_ids"]], dim=0).to(device)
    position_ids = torch.concat([batch1["position_ids"], batch2["position_ids"]], dim=0) .to(device)
    attn_mask = torch.concat([attn_mask1, attn_mask2], dim=0).to(device)
    
    past_kv = []
    for i in range(len(padded_kv1)):
        k1, v1 = padded_kv1[i]
        k2, v2 = padded_kv2[i]
        k = torch.concat([k1, k2], dim=0).to(device)
        v = torch.concat([v1, v2], dim=0).to(device)
        past_kv.append((k, v))
    
    #recursive_tensor_device_check(input_ids, device)
    #recursive_tensor_device_check(position_ids, device)
    #recursive_tensor_device_check(attn_mask, device)
    #recursive_tensor_device_check(past_kv, device)
    #print(f"Batch merged.")
    
    
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attn_mask,
        "past_key_values": past_kv,
        "responses": batch1["responses"] + batch2["responses"],
        "tokens_remaining": batch1["tokens_remaining"] + batch2["tokens_remaining"],
    }


def filter_batch(batch, device='cpu'):
    #print("Filtering batch....")
    # mark all rows with 0 tokens remaining for removal
    remove_indices = []
    for i, tokens_remaining in enumerate(batch["tokens_remaining"]):
        if tokens_remaining <= 0:
            remove_indices.append(i)
            
    completed_responses = [
        r 
        for i, r in enumerate(batch["responses"])
        if i in remove_indices
    ]
    
    # first, define a mask used to subselect the indices to keep
    # from each tensor, given the indices to remove
    batch_size = batch["input_ids"].size(0)
    mask = torch.ones(batch_size, dtype=torch.bool)
    mask[remove_indices] = False

    # index into the tensors using the mask to remove rows
    input_ids = batch["input_ids"][mask]
    position_ids = batch["position_ids"][mask]
    attention_mask = batch["attention_mask"][mask]
    responses = [
        r 
        for i, r in enumerate(batch["responses"])
        if i not in remove_indices
    ]
    tokens_remaining = [
        v 
        for i, v in enumerate(batch["tokens_remaining"])
        if i not in remove_indices
    ]

    past_key_values = batch["past_key_values"]
    new_past_key_values = []
    for i in range(len(past_key_values)):
        k, v = past_key_values[i]
        k = k[mask]
        v = v[mask]
        new_past_key_values.append((k, v))
    past_key_values = new_past_key_values
    
    if input_ids.size(0) > 0:
        # next, as an optimization to avoid wasting compute cycles on padding tokens,
        # we will left truncate the attention_mask and past_key_values to the longest
        # remaining sequence length
        # we obtain the longest sequence length by looking for the min first non-zero index
        # of the attention mask
        zero_mask = attention_mask == 0
        cumprod = zero_mask.cumprod(dim=1)  # cumprod ensures we stop accumulating when we see a 1
        leading_zeros_count = cumprod.sum(dim=1)
        min_leading_zeros = torch.min(leading_zeros_count)
        truncation_offset = min_leading_zeros.item()

        # do the trunction
        attention_mask = attention_mask[:, truncation_offset:]
        past_key_values = past_key_values
        new_past_key_values = []
        for i in range(len(past_key_values)):
            k, v = past_key_values[i]
            k = k[:, :, truncation_offset:, :]
            v = v[:, :, truncation_offset:, :]
            new_past_key_values.append((k, v))
        past_key_values = new_past_key_values
    
    # return the new batch
    #print(f"Batch filtered.")
    #recursive_tensor_device_check(input_ids, device)
    #recursive_tensor_device_check(position_ids, device)
    #recursive_tensor_device_check(attention_mask, device)
    #recursive_tensor_device_check(past_key_values, device)

    #print(f"Batch filtered. Tensor are correct.")
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "responses": responses,
        "tokens_remaining": tokens_remaining,
    }, remove_indices, completed_responses


def generate(model, tokenizer, requests):
    # seed the random number generator so our results are deterministic
    random.seed(42)
    device = model.device
    # constants
    batch_size = 8
    request_queue = copy.copy(requests)
    
    responses = [None] * len(requests)

    # and run the initial prefill step
    batch = init_batch(tokenizer, request_queue[:batch_size], device=device)
    cached_batch = generate_next_token(model, tokenizer, batch, device=device)
    request_queue = request_queue[batch_size:]

    # continue until both the request queue is fully drained and every input
    # within the cached_batch has completed generation
    while len(request_queue) > 0 or cached_batch["input_ids"].size(0) > 0:
        batch_capacity = batch_size - cached_batch["input_ids"].size(0)
        if batch_capacity > 0 and len(request_queue) > 0:
            # prefill
            new_batch = init_batch(tokenizer, request_queue[:batch_capacity])
            new_batch = generate_next_token(model, tokenizer, new_batch, device=device)
            request_queue = request_queue[batch_capacity:]

            # merge
            cached_batch = merge_batches(cached_batch, new_batch, device=device)

        # decode
        cached_batch = generate_next_token(model, tokenizer, cached_batch, device=device)

        # remove any inputs that have finished generation
        cached_batch, removed_indices, completed_responses = filter_batch(cached_batch, device=device)

        for idx, resp in zip(removed_indices, completed_responses):
            responses[idx] = resp
    
    return responses