# -*- coding: utf-8 -*-
# torchrun --standalone --nnodes=1 --nproc-per-node=8 generate.py 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import pandas as pd
import os
from torch.distributed import is_initialized as dist_is_initialized
from torch.distributed import barrier as dist_barrier
from torch import distributed as dist
from datasets import Dataset
from torch.utils.data import DataLoader

# -------------------- os environment --------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["SWANLAB_MODE"] = "disabled"
os.environ["SWANLAB_DISABLED"] = "true"
os.environ['NCCL_DEBUG'] = 'WARN'

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

def mprint(pstr):
    if is_main_process():
        return print(pstr)

def initialize_distributed():
    if not dist.is_initialized() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

def wait_for_everyone():
    if torch.distributed.is_available() and dist_is_initialized():
        dist_barrier()

def process_func(example):
    system_input = "You are an assistant that translates special tokens into meaningful English sentences. The special tokens represent the number of characters in each token (including any leading space). Example: token `\"The\"` formalized as Special Tokens: _3, token `\" quick\"` formalized as Special Tokens: _6  (note the leading space), token `\".\"` formalized as Special Tokens: _1. Special Tokens: _2 _3 _3 _9 _6 _1  can translate like: As an AI language model,\nYou need to reconstruct the original sentence from these special tokens."
    
    #====== merged =======
    system_input = (
    "You are an assistant that reconstructs fluent English text from Special Tokens. "
    "Each Special Token `Mk` represents the total length (in characters, including any leading space) "
    "of two consecutive original tokens merged together, except possibly the last one which may be a single token. "
    "Recreate the original text by splitting each `Mk` into two tokens that match natural English spacing and punctuation patterns: "
    "The first token of a sentence has no leading space, later tokens usually do. "
    "Punctuation tokens (.,!?:;) typically have no leading space and length 1. "
    "A newline `\\n` counts as length 1. "
    "When multiple reconstructions are possible, choose the most natural and grammatical one. "
    "Output only the reconstructed English, without counts or explanations. "
    "Example: Special Tokens: M5 M12 M7 → "
    "As an AI language model,"
)
    prompt = (
        f"<s><|im_start|>system\n{system_input}<|im_end|>\n"
        f"<|im_start|>user\n{example['Encoding']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


    return {
        "prompt": prompt,
        "label": example['Sentence']
    }


def collate_fn(batch, tokenizer):
    prompts = [item["prompt"] for item in batch]
    labels = [item["label"] for item in batch]
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    return tokenized, prompts, labels


if __name__ == "__main__":
    batch_size = 16 
    max_new_tokens = 80
    # ========COT============
    # max_new_tokens = 1000 
    num_beams = 16

    model_path = "/final_model"
    df_test = pd.read_csv("/first_sentence/test.csv")
    
    output_dir = "/train/res"
    file_name = "merged_output"

    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    mprint(world_size)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    # ==== load tokenizer ====
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'

    mprint("token embedding:")
    for i in range(1):
        token_id = tokenizer.convert_tokens_to_ids(f'_{i}')
        print(f"Rank: {rank} Token: _{i} ID: {token_id}")

    mprint(tokenizer.pad_token_id)
    mprint(tokenizer.eos_token_id)
    wait_for_everyone()

    # ==== load dataset ====
    dataset = Dataset.from_pandas(df_test)
    dataset = dataset.map(process_func, remove_columns=dataset.column_names, num_proc=8)
    dataset = dataset.shard(num_shards=world_size, index=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, tokenizer))

    # ==== load model ====
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    mprint(f"Model dtype: {next(model.parameters()).dtype}")
    mprint(f"Model devices: {set(str(p.device) for p in model.parameters())}")
    
    mprint("Starting generation...")

    local_outputs = {'Sentence_0':[], 'Generated_0':[]}
    for batch_idx, (batch_inputs, batch_prompts, batch_labels) in enumerate(tqdm(dataloader)):
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                # output_scores=True,
                # return_dict_in_generate=True,
                # num_beam_groups=16,
                # diversity_penalty=0.8,
                num_beams=num_beams,
                # no_repeat_ngram_size=2,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        outputs = outputs[:, batch_inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        local_outputs['Generated_0'] += list(texts)
        local_outputs['Sentence_0'] += list(batch_labels)

        if batch_idx % 5 == 0:
            mprint(f"==== Batch {batch_idx} Output Preview ====")
            for pred, label in zip(texts[:2], batch_labels[:2]):
                mprint(f"Pred: {pred}\nLabel: {label}\n")
            mprint("=======================================\n")


    pd.DataFrame(local_outputs).to_csv(f"{output_dir}/res_rank_{rank}.csv", index=False)

    dist.barrier()

    if is_main_process():
        res_df = pd.read_csv(f"{output_dir}/res_rank_0.csv")
        for i in range(1, world_size):
            cur_df = pd.read_csv(f"{output_dir}/res_rank_{i}.csv")
            # res_df = res_df.append(cur_df)
            res_df = pd.concat([res_df,cur_df], ignore_index=True)
        res_df.to_csv(f"{output_dir}/res_{file_name}.csv",index=False)

        for i in range(world_size):
            csv_file_path = f"{output_dir}/res_rank_{i}.csv"
            if os.path.exists(csv_file_path):
                os.remove(csv_file_path)
