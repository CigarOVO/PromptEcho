# torchrun --standalone --nnodes=1 --nproc-per-node=8 merged_decomposition.py 2>&1 | tee logs/.log
import os
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from datasets import Dataset
import pandas as pd
from torch import nn
from torch.distributed import is_initialized as dist_is_initialized
from torch.distributed import barrier as dist_barrier
from torch import distributed as dist
import evaluate
from sentence_transformers import SentenceTransformer
import numpy as np
import Levenshtein
import warnings
import logging
import time 
import shutil
from transformers import TrainerCallback, TrainerControl, TrainerState
from torch.serialization import add_safe_globals
import re, random
from torch.optim import AdamW
import torch.nn.functional as F

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>\n\n"

add_safe_globals([
    np.core.multiarray._reconstruct,
    np.ndarray,
    np.dtype,
    np.float16, np.float32, np.float64,
    np.int32,
    np.int64,
    np.bool_,
    np.ufunc,
    np.void,
    np.int8, np.int16,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.complex64, np.complex128,
    np.str_, np.bytes_,
    np.object_
    ])

_original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = patched_load
# -------------------- os --------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["SWANLAB_MODE"] = "disabled"
os.environ["SWANLAB_DISABLED"] = "true"

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

os.environ['NCCL_DEBUG'] = 'WARN'


class BestModelBackupCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        best_ckpt = state.best_model_checkpoint
        if best_ckpt and os.path.exists(best_ckpt):
            backup_path = os.path.join(self.output_dir, "best_model_backup")
            if os.environ.get("LOCAL_RANK", "0") == "0":
                print(f"[Callback] best model: {best_ckpt}")
                try:
                    shutil.copytree(best_ckpt, backup_path, dirs_exist_ok=True)
                    print(f"[Callback] copy: {backup_path}")
                except Exception as e:
                    print(f"[Callback] failed: {e}")

def rationale_linear_decay(step, total, start=0.3, end=0.0):
    r = min(step / max(1, total), 1.0)
    return start + (end - start) * r


def strip_think_block(text: str) -> str:
    # return re.sub(rf"{re.escape(THINK_OPEN)}.*?{re.escape(THINK_CLOSE)}", "", text, flags=re.S)
    return re.sub(rf"{re.escape(THINK_OPEN)}.*?{re.escape(THINK_CLOSE)}", THINK_OPEN + "\n\n" + THINK_CLOSE, text, flags=re.S)

def skeletonize_think_block(text: str, max_chars=120) -> str:
    def _skel(m):
        inner = m.group(0)[len(THINK_OPEN):-len(THINK_CLOSE)]
        inner = re.sub(r"\s+", " ", inner).strip()[:max_chars]
        return f"{THINK_OPEN}{inner}{THINK_CLOSE}"
    return re.sub(rf"{re.escape(THINK_OPEN)}.*?{re.escape(THINK_CLOSE)}", _skel, text, flags=re.S)

def cot_dropout(cot_text: str, p=0.5, mode="remove", max_chars=120, rng=None):
    """
    p: probability
    mode: "remove" | "skeleton"
    rng: random.Random(seed) 
    """
    r = (rng.random() if rng is not None else random.random())
    if r >= p:
        return cot_text
    if mode == "remove":
        return strip_think_block(cot_text)
    elif mode == "skeleton":
        return skeletonize_think_block(cot_text, max_chars=max_chars)
    return cot_text

def setup_logging():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if local_rank == 0:
        logging.basicConfig(level=logging.INFO)
        warnings.filterwarnings("ignore", message=".*tokenizer is now deprecated.*")
        warnings.filterwarnings("ignore", message=".*do_sample.*top_p.*")
        warnings.filterwarnings("ignore", message=".*do_sample.*top_k.*")
    else:
        logging.basicConfig(level=logging.ERROR)
        warnings.filterwarnings("ignore")


setup_logging()

def create_data_collator(tokenizer, padding_side):
    tokenizer.padding_side = padding_side
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

def is_main_process():
    return int(os.environ.get("LOCAL_RANK", 0)) == 0

def wait_for_everyone():
    if torch.distributed.is_available() and dist_is_initialized():
        dist_barrier()

def initialize_distributed():
    if not dist.is_initialized() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")

def process_func(example):
    system_input = """
You are an assistant that reconstructs fluent English from merged special tokens. 

Token conventions:
- Primitive tokens: `_x` represent learned pieces whose length metadata `x` was used during preprocessing (including any leading space).
- Merged tokens: `Mk` are opaque single tokens that correspond to one or two primitive tokens: `_x`.

Decoding policy:
- When receiving merged tokens (`Mk`), infer the original primitive sequence based on learned patterns, then decode to English.
- When provided with a `<think>` block, follow the primitive sequence from `_x` and reconstruct the English sentence accordingly.
- If no `<think>` block is provided, internally derive the primitive sequence from the merged tokens and reconstruct the English sentence accordingly.

Your task is to reconstruct the sentence from the provided merged special tokens.
"""

    prompt = (
        f"<s><|im_start|>system\n{system_input}<|im_end|>\n"
        f"<|im_start|>user\n{example['Encoding']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    instruction = tokenizer(prompt, add_special_tokens=False)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    response = tokenizer(f"<think>\n{example['Cot']}\n</think>\n\n{example['Sentence']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [im_end_id]
    attention_mask = instruction["attention_mask"] +response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [im_end_id]


    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def val_process_func(example):
    system_input = """
You are an assistant that reconstructs fluent English from merged special tokens. 

Token conventions:
- Primitive tokens: `_x` represent learned pieces whose length metadata `x` was used during preprocessing (including any leading space).
- Merged tokens: `Mk` are opaque single tokens that correspond to one or two primitive tokens: `_x`.

Decoding policy:
- When receiving merged tokens (`Mk`), infer the original primitive sequence based on learned patterns, then decode to English.
- When provided with a `<think>` block, follow the primitive sequence from `_x` and reconstruct the English sentence accordingly.
- If no `<think>` block is provided, internally derive the primitive sequence from the merged tokens and reconstruct the English sentence accordingly.

Your task is to reconstruct the sentence from the provided merged special tokens.
"""

    prompt = (
        f"<s><|im_start|>system\n{system_input}<|im_end|>\n"
        f"<|im_start|>user\n{example['Encoding']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(f"{example['Sentence']}", add_special_tokens=False)

    labels = [-100] * (len(instruction["input_ids"]) - len(response["input_ids"])) + response["input_ids"]
    # labels = [-100] * len(instruction["input_ids"])

    labels_text = example["Sentence"]  

    return {
        "input_ids": instruction["input_ids"],
        "attention_mask": instruction["attention_mask"],
        "labels": labels
        # "target_text": labels_text
    }

class CustomTrainer(Trainer): 
    def _build_masks_from_ids(self, input_ids, tokenizer):
        
        B, L = input_ids.shape
        device = input_ids.device
        mask_r = torch.zeros((B, L), dtype=torch.float32, device=device)
        mask_f = torch.zeros((B, L), dtype=torch.float32, device=device)

        tid_think_beg = tokenizer.convert_tokens_to_ids("<think>")
        tid_think_end = tokenizer.convert_tokens_to_ids("</think>")
        tid_im_end    = tokenizer.convert_tokens_to_ids("<|im_end|>")

        ids = input_ids.tolist()
        for b in range(B):
            seq = ids[b]
            
            try:
                i1 = seq.index(tid_think_beg)
                i2 = seq.index(tid_think_end, i1 + 1)  
            except ValueError:
                
                i1 = i2 = -1

            try:
                i_end = seq.index(tid_im_end)
            except ValueError:
                i_end = L   

            if 0 <= i1 < i2 < L:
                mask_r[b, i1: i2 + 1] = 1.0

                left = min(i2 + 1, L)
                right = max(left, i_end)  # [left, i_end)
                mask_f[b, left: i_end] = 1.0
            else:
                mask_f[b, :i_end] = 1.0

        return mask_r, mask_f

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None))
        logits = outputs.logits  # [B, L, V]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # shift_input_ids = inputs["input_ids"][:, 1:].contiguous()

        tok = getattr(self, "processing_class", None)
        assert tok is not None

        mask_r, mask_f = self._build_masks_from_ids(inputs["input_ids"], tok)
        shift_mask_r = mask_r[:, 1:]
        shift_mask_f = mask_f[:, 1:]

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                      shift_labels.view(-1))
        ce = ce.view(shift_labels.size())


        step = self.state.global_step
        total = self.state.max_steps or (step + 1)
        w_r = rationale_linear_decay(step, total, start=0.3, end=0.0)
        w_f = 1.0

        weights = w_r * shift_mask_r + w_f * shift_mask_f  # [B, L-1]
        denom = weights.sum().clamp_min(1.0)
        loss = (ce * weights).sum() / denom

        return (loss, outputs) if return_outputs else loss
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        # print("Debug inputs:", inputs)
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
        
            try:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1000,  
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.convert_tokens_to_ids("<|im_end|>"),
                    do_sample=False,     
                    # temperature=1.0,
                    num_beams=16,
                    num_return_sequences=1,
                )


                input_length = input_ids.shape[1]
                generated_ids = generated_ids[:, input_length:]
                

                vocab_size = len(self.processing_class)
                generated_ids = torch.clamp(generated_ids, min=0, max=vocab_size-1)
                
            except Exception as e:

                print(f"Generation failed: {e}")
                generated_ids = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
                generated_ids.fill_(self.processing_class.pad_token_id)
            
        return (loss, generated_ids, inputs['labels'])


def compute_metrics_qwen_final(eval_pred, tokenizer):
    preds, labels = eval_pred
    
    total_samples = len(preds)
    valid_pairs = []
    empty_count = 0
    decode_errors = 0
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:  
        print(f"\n=== eval (sample: {total_samples}) ===")
    pre_time = time.time()
    
    for i, (pred, label) in enumerate(zip(preds, labels)):
        try:
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            if len(pred) > 0 and not all(p == tokenizer.pad_token_id for p in pred):
                vocab_size = len(tokenizer)
                valid_pred = []
                
                for token_id in pred:
                    if isinstance(token_id, (int, np.integer)) and 0 <= token_id < vocab_size:
                        valid_pred.append(int(token_id))
                
                if valid_pred:
                    try:
                        ptxt = tokenizer.decode(valid_pred, skip_special_tokens=True)
                        # think_token = "</think>\n\nAnswer: "
                        think_token = "</think>\n\n"
                        if think_token in ptxt:
                            ptxt = ptxt.split(think_token,-1)[-1].strip()

                        ptxt = ptxt.replace("<|im_end|>", "").strip()
                    except Exception as decode_error:
                        decode_errors += 1
                        if i < 3 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            print(f"  failed {i}: {decode_error}")
            
            label_text = ""
            label_tokens = [t for t in label if t != -100]
            if label_tokens:
                try:
                    label_text = tokenizer.decode(label_tokens, skip_special_tokens=True).strip()
                except:
                    decode_errors += 1
            
            if ptxt.strip() and label_text.strip():
                valid_pairs.append((ptxt.strip(), label_text.strip()))
                
                if i < 10 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"  succeeded {i}: '{ptxt}' vs '{label_text}'")
            else:
                empty_count += 1
                if i < 10 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    print(f"  empty {i}: pred='{ptxt}', label='{label_text}'")
                
        except Exception as e:
            empty_count += 1
            if i < 10 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"  failed {i}: {e}")

    success_rate = len(valid_pairs) / total_samples * 100 if total_samples > 0 else 0
    empty_rate = empty_count / total_samples * 100 if total_samples > 0 else 0
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"  succeeded: {len(valid_pairs)}/{total_samples} ({success_rate:.1f}%)")
        print(f"  empty: {empty_count}/{total_samples} ({empty_rate:.1f}%)")
        print(f"  decode errors: {decode_errors}")
    
    print(f'decode time: {time.time() - pre_time}')
    pre_time = time.time()

    metrics = {
        "success_rate": success_rate,
        "empty_rate": empty_rate,
        "valid_samples": len(valid_pairs)
    }
    
    if len(valid_pairs) > 0:
        valid_preds, valid_labels = zip(*valid_pairs)
        
        try:
            split_re = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
            preds_r = ["\n".join(split_re.split(p)) for p in valid_preds]
            labels_r = ["\n".join(split_re.split(l)) for l in valid_labels]
            
            rouge_raw = evaluate.load(rouge_path).compute(
                predictions=preds_r, references=labels_r, use_stemmer=True
            )

            metrics.update({
                "rouge1": rouge_raw["rouge1"] * 100,
                "rouge2": rouge_raw["rouge2"] * 100, 
                "rougeL": rouge_raw["rougeL"] * 100,
            })
        except Exception as e:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"  failed rouge: {e}")
            metrics.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})

        pre_time = time.time()

        try:
            st = SentenceTransformer(sentence_transformer_model_path).eval()
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)(
                st.encode(list(valid_preds), convert_to_tensor=True),
                st.encode(list(valid_labels), convert_to_tensor=True)
            ).cpu().numpy()
            metrics["cos_0.5"] = sum(cos > 0.5) / len(cos)
            metrics["cos_0.9"] = sum(cos > 0.9) / len(cos)
            metrics["cos_1"] = sum(cos > 0.9999) / len(cos)
        except Exception as e:
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                print(f"  failed cos: {e}")
            metrics["cos_0.5"] = 0.0

        pre_time = time.time()


        metrics["gen_len"] = np.mean([len(p.split()) for p in valid_preds])
        
        metrics["weighted_rougeL"] = metrics["rougeL"] * (success_rate / 100)
        
    else:

        metrics.update({"cos_0.5": 0})

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"final: COS_0.5={metrics['cos_0.5']:.4f}, COS_0.9={metrics['cos_0.9']:.4f}")
        print("="*50)
    
    return {k: round(v, 4) for k, v in metrics.items()}


def verify_process_func():

    test_example = {
        'Encoding': 'test encoding',
        'Sentence': 'test sentence response <|im_start|> <think> </think> _1 M3',
        'Cot': '_1 means length 1'
    }
    
    result = process_func(test_example)
    

    input_text = tokenizer.decode(result['input_ids'])
    print("prompt:")
    print(input_text)
    print()
    
    label_tokens = [token for token in result['labels'] if token != -100]
    label_text = tokenizer.decode(label_tokens)
    print("label:")
    print(label_text)
    print()
    
    print("location:")
    for i, (input_id, label) in enumerate(zip(result['input_ids'], result['labels'])):
        if label != -100:
            token = tokenizer.decode([input_id])
            print(f"loc {i}: token='{token}', label={label}")


# =========================================================
if __name__ == "__main__":
    per_device_batch_size = 8
    gradient_accumulation_steps = 8
    num_train_epochs = 30
    learning_rate = 1e-5  
    resume_from_checkpoint = False
    # resume_from_checkpoint = True 

    # base_model_path = "/Qwen/Qwen3-8B"
    base_model_path = "/output/first_r32_64_2lr/final_model/"
    prepared_model_path = "/train/prepared_model_xM_base_r32_64/"
    model_checkpoint_path = "/output/merged_first_xcot_r16_32_rational/checkpoints/"
    final_model_path = "/output/merged_first_xcot_r16_32_rational/final_model/"
    
    
    sentence_transformer_model_path = "/eval/hf_cache/models/all-MiniLM-L12-v1"
    rouge_path = "/eval/hf_cache/modules/rouge"

    initialize_distributed()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    if local_rank != 0:
        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.ERROR)

    
    # ---------- save ----------
    if is_main_process():
        print("main process: add_tokens/resize/save...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
        new_tokens = [f"M{i}" for i in range(60)]
        tokenizer.add_tokens(new_tokens)
        tokenizer.save_pretrained(prepared_model_path)

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            nn.init.normal_(model.get_input_embeddings().weight[-len(new_tokens):], mean=0.0, std=0.02)
        model.config.vocab_size = len(tokenizer)
        model.save_pretrained(prepared_model_path, torch_dtype=torch.bfloat16)
        print("main process: save tokenizer and expanded model completed.")
    wait_for_everyone()

    print(f"{local_rank=}: load tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained(prepared_model_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(prepared_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    # ---------- LoRA ----------
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens"],
        r=16, lora_alpha=32, lora_dropout=0.05, inference_mode=False
    )
    model = get_peft_model(model, lora_cfg)


    model.config.use_cache = False

    # ---------- unfreeze new token embedding ----------
    new_token_ids = [tokenizer.convert_tokens_to_ids(f"M{i}") for i in range(60)]
    emb_layer = model.get_input_embeddings()
    emb_layer.weight.requires_grad = True
    mask = torch.zeros_like(emb_layer.weight, device=emb_layer.weight.device, dtype=emb_layer.weight.dtype)
    for idx in new_token_ids:
        mask[idx, :] = 1.0

    def embedding_grad_hook(grad):
        return grad * mask

    emb_layer.weight.register_hook(embedding_grad_hook)

    if is_main_process():
        model.print_trainable_parameters()


    df_train = pd.read_json("/data/merged_train_xcot.jsonl", lines=True)
    df_val = pd.read_csv("/data/test_merged_first_M.csv")

    ds_train = Dataset.from_pandas(df_train)
    ds_val   = Dataset.from_pandas(df_val)

    tokenized_train = ds_train.map(process_func, remove_columns=ds_train.column_names, num_proc=8)
    tokenized_val   = ds_val.map(val_process_func, remove_columns=ds_val.column_names, num_proc=8)


    emb_group = [model.get_input_embeddings().weight]
    lora_params = [p for n,p in model.named_parameters() if p.requires_grad and ("lora" in n.lower())]
    # sem_params = list(sem_head.parameters())

    optimizer = AdamW(
        [
            {"params":emb_group, "lr":2e-4, "weight_decay":0.0},
            {"params":lora_params, "lr":1e-5, "weight_decay":0.0},
            # {"params":sem_params, "lr":1e-3, "weight_decay":0.0}
        ],
        betas = (0.9, 0.98), eps=1e-8,
    )
    
    train_args = TrainingArguments(
        output_dir                   = model_checkpoint_path,
        per_device_train_batch_size  = per_device_batch_size,
        per_device_eval_batch_size   = per_device_batch_size,
        gradient_accumulation_steps  = gradient_accumulation_steps,
        num_train_epochs             = num_train_epochs,
        learning_rate                = learning_rate,
        # weight_decay                 = 0.01,
        # warmup_ratio                 = 0.03,
        # warmup_ratio                 = 0.05,
        weight_decay                 = 0.00,
        warmup_ratio                 = 0.06,
        lr_scheduler_type            = "cosine", 
        eval_strategy                = "epoch",
        save_strategy                = "epoch",
        logging_strategy             = "epoch",
        # max_steps = 1,                        
        # eval_strategy                = "steps",
        # save_strategy                = "steps",
        # logging_strategy             = "steps",
        # eval_steps                   = 5,
        # save_steps                   = 10,
        # logging_steps                = 10,
        # save_total_limit             = 5,
        load_best_model_at_end       = True,
        metric_for_best_model        = "cos_0.5",
        greater_is_better            = True,
        optim                        = "adamw_torch",
        adam_beta1                   = 0.9,
        adam_beta2                   = 0.999,
        adam_epsilon                 = 1e-8,
        max_grad_norm                = 1.0,
        save_on_each_node            = False,
        gradient_checkpointing       = False,  
        dataloader_pin_memory        = False,
        label_names                  = ["labels"],
        report_to                    = "none",
        local_rank                   = local_rank,
        # ddp_find_unused_parameters   = True,
        ddp_find_unused_parameters   = False,
        fp16                         = False,
        bf16                         = True,
        dataloader_num_workers       = 32,
        remove_unused_columns        = False,
        dataloader_drop_last         = True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        trainer = CustomTrainer( 
            model           = model,
            args            = train_args,
            train_dataset   = tokenized_train,
            eval_dataset    = tokenized_val,
            data_collator   = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            # tokenizer       = tokenizer,
            # data_collator   = create_data_collator(tokenizer, "right"),
            # val_data_collator= create_data_collator(tokenizer, "left"),
            processing_class = tokenizer,
            # compute_metrics = lambda e: compute_metrics_qwen_final(e, tokenizer),
            compute_metrics = lambda e: compute_metrics_qwen_final(e, tokenizer),
            # callbacks       = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)],
            callbacks       = [
                                BestModelBackupCallback(output_dir=model_checkpoint_path),
                            ],
            optimizers = (optimizer, None),

        )
    
    if is_main_process():
        print("validation...")
        verify_process_func()

    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    torch.cuda.empty_cache()
    # ---------- 保存 ----------
    if is_main_process():
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
 
