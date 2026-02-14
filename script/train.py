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
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import Levenshtein
import warnings
import logging
import time 

# -------------------- os --------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["SWANLAB_MODE"] = "disabled"
os.environ["SWANLAB_DISABLED"] = "true"

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

os.environ['NCCL_DEBUG'] = 'WARN'


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

# -------------------- sample process --------------------
def process_func(example):
    system_input = "You are an assistant that translates special tokens into meaningful English sentences. The special tokens represent the number of characters in each token (including any leading space). Example: token `\"The\"` formalized as Special Tokens: _3, token `\" quick\"` formalized as Special Tokens: _6  (note the leading space), token `\".\"` formalized as Special Tokens: _1. Special Tokens: _2 _3 _3 _9 _6 _1  can translate like: As an AI language model,\nYou need to reconstruct the original sentence from these special tokens."
    prompt = (
        f"<s><|im_start|>system\n{system_input}<|im_end|>\n"
        f"<|im_start|>user\n{example['Encoding']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(f"{example['Sentence']}", add_special_tokens=False)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    input_ids = instruction["input_ids"] + response["input_ids"] + [im_end_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [im_end_id]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def val_process_func(example):
    system_input = "You are an assistant that translates special tokens into meaningful English sentences. The special tokens represent the number of characters in each token (including any leading space). Example: token `\"The\"` formalized as Special Tokens: _3, token `\" quick\"` formalized as Special Tokens: _6  (note the leading space), token `\".\"` formalized as Special Tokens: _1. Special Tokens: _2 _3 _3 _9 _6 _1  can translate like: As an AI language model,\nYou need to reconstruct the original sentence from these special tokens."
    prompt = (
        f"<s><|im_start|>system\n{system_input}<|im_end|>\n"
        f"<|im_start|>user\n{example['Encoding']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    instruction = tokenizer(prompt, add_special_tokens=False)
    response = tokenizer(f"{example['Sentence']}", add_special_tokens=False)
    labels = [-100] * (len(instruction["input_ids"]) - len(response["input_ids"])) + response["input_ids"]
    return {
        "input_ids": instruction["input_ids"],
        "attention_mask": instruction["attention_mask"],
        "labels": labels
    }

class CustomTrainer(Trainer): 
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            # loss
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            
            
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            
            try:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,  
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.convert_tokens_to_ids("<|im_end|>"),
                    do_sample=False,     
                    # temperature=1.0,
                    num_beams=1,
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
    
    # 详细的诊断信息
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:  
        print(f"\n=== evaluate (sample num: {total_samples}) ===")
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
                        think_token = "</think>"
                        if ptxt.lstrip().startswith(think_token):
                            ptxt = ptxt[ptxt.find(think_token) + len(think_token):]

                        ptxt = ptxt.replace("<|im_end|>", "").strip()
                    except Exception as decode_error:
                        decode_errors += 1
                        if i < 3 and int(os.environ.get("LOCAL_RANK", 0)) == 0:
                            print(f"  fail {i}: {decode_error}")
            
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
                    print(f"  success {i}: '{ptxt}' vs '{label_text}'")
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
        print(f"  success: {len(valid_pairs)}/{total_samples} ({success_rate:.1f}%)")
        print(f"  empty: {empty_count}/{total_samples} ({empty_rate:.1f}%)")
        print(f"  failed: {decode_errors}")
    
    print(f'time spent：{time.time() - pre_time}')
    pre_time = time.time()

    metrics = {
        "success_rate": success_rate,
        "empty_rate": empty_rate,
        "valid_samples": len(valid_pairs)
    }
    
    if len(valid_pairs) > 0:
        valid_preds, valid_labels = zip(*valid_pairs)
        
        # ROUGE 
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
                print(f"  ROUGE failed: {e}")
            metrics.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})

        pre_time = time.time()

        # cosine similarity
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
                print(f"  cosine failed: {e}")
            metrics["cos_0.5"] = 0.0

        pre_time = time.time()

        # ED
        # try:
        #     ed = np.mean([
        #         (max(len(p), len(l)) - Levenshtein.distance(p, l)) / max(len(p), len(l), 1)
        #         for p, l in zip(valid_preds, valid_labels)
        #     ])
        #     metrics["edit_distance"] = ed
        # except:
        #     metrics["edit_distance"] = 0.0

        # print(f'ED time spent：{time.time() - pre_time}')
        # pre_time = time.time()

        # length
        metrics["gen_len"] = np.mean([len(p.split()) for p in valid_preds])
        
        # metric combination
        metrics["weighted_rougeL"] = metrics["rougeL"] * (success_rate / 100)
        
    else:
        metrics.update({"cos_0.5": 0})

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"final metric: COS_0.5={metrics['cos_0.5']:.4f}, COS_0.9={metrics['cos_0.9']:.4f}")
        print("="*50)
    
    return {k: round(v, 4) for k, v in metrics.items()}


def verify_process_func():

    test_example = {
        'Encoding': 'test encoding',
        'Sentence': 'test sentence response _1'
    }
    
    result = process_func(test_example)

    input_text = tokenizer.decode(result['input_ids'])
    print("input:")
    print(input_text)
    print()
    
    label_tokens = [token for token in result['labels'] if token != -100]
    label_text = tokenizer.decode(label_tokens)
    print("label:")
    print(label_text)
    print()

    print("label location:")
    for i, (input_id, label) in enumerate(zip(result['input_ids'], result['labels'])):
        if label != -100:
            token = tokenizer.decode([input_id])
            print(f"location {i}: token='{token}', label={label}")


# =========================================================
if __name__ == "__main__":
    per_device_batch_size = 8
    gradient_accumulation_steps = 8
    num_train_epochs = 1
    learning_rate = 3e-5
    resume_from_checkpoint = False

    base_model_path = "/model/qwen-8b"
    prepared_model_path = "/model/prepared_model/"
    model_checkpoint_path = "/model/checkpoints/"
    final_model_path = "/model/final_model/"
    sentence_transformer_model_path = "/model/all-MiniLM-L12-v1"
    rouge_path = "/scripts/rouge"

    initialize_distributed()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    if local_rank != 0:
        warnings.filterwarnings("ignore")
        logging.basicConfig(level=logging.ERROR)

    # ---------- tokenizer ----------
    # if is_main_process():
    #     print("main process: add_tokens/resize/save...")
    #     tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, trust_remote_code=True)
    #     new_tokens = [f"s{i}" for i in range(20)]
    #     new_tokens = [f"M{i}" for i in range(60)]
    #     tokenizer.add_tokens(new_tokens)
    #     tokenizer.save_pretrained(prepared_model_path)

    #     model = AutoModelForCausalLM.from_pretrained(
    #         base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    #     )
    #     model.resize_token_embeddings(len(tokenizer))
    #     with torch.no_grad():
    #         nn.init.normal_(model.get_input_embeddings().weight[-len(new_tokens):], mean=0.0, std=0.02)
    #     model.config.vocab_size = len(tokenizer)
    #     model.save_pretrained(prepared_model_path, torch_dtype=torch.bfloat16)
    #     print("main process: save tokenizer")
    wait_for_everyone()

    # ---------- load ----------
    print(f"{local_rank=}: load tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(prepared_model_path, use_fast=False, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(prepared_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)

    # ---------- LoRA ----------
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj", "lm_head"],
        modules_to_save=["embed_tokens"],
        r=8, lora_alpha=16, lora_dropout=0.1, inference_mode=False
    )
    model = get_peft_model(model, lora_cfg)

    # ---------- unfreeze token embedding ----------
    new_token_ids = [tokenizer.convert_tokens_to_ids(f"_{i}") for i in range(20)]
    with torch.no_grad():
        emb_layer = model.get_input_embeddings()
        for param in emb_layer.parameters():
            param.requires_grad = False
        for idx in new_token_ids:
            emb_layer.weight[idx, :].requires_grad = True

    if is_main_process():
        model.print_trainable_parameters()

    # ---------- load data ----------
    df_train = pd.read_json("/data/train.jsonl", lines=True)
    df_val = pd.read_csv("/data/val.csv")
    
    ds_train = Dataset.from_pandas(df_train)
    ds_val   = Dataset.from_pandas(df_val)

    tokenized_train = ds_train.map(process_func, remove_columns=ds_train.column_names, num_proc=8)
    tokenized_val   = ds_val.map(val_process_func, remove_columns=ds_val.column_names, num_proc=8)

    # ---------- Training parameters ----------
    train_args = TrainingArguments(
        output_dir                   = model_checkpoint_path,
        per_device_train_batch_size  = per_device_batch_size,
        per_device_eval_batch_size   = per_device_batch_size,
        gradient_accumulation_steps  = gradient_accumulation_steps,
        num_train_epochs             = num_train_epochs,
        learning_rate                = learning_rate,
        weight_decay                 = 0.01,
        warmup_ratio                 = 0.03,
        lr_scheduler_type            = "cosine", 
        eval_strategy                = "epoch",
        save_strategy                = "epoch",
        logging_strategy             = "epoch",
        # max_steps = 10,                        
        # eval_strategy                = "steps",
        # save_strategy                = "steps",
        # logging_strategy             = "steps",
        # eval_steps                   = 5,
        # save_steps                   = 10,
        # logging_steps                = 10,
        save_total_limit             = 5,
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
        ddp_find_unused_parameters   = True,
        fp16                         = False,
        bf16                         = True,
        dataloader_num_workers       = 4,
        remove_unused_columns        = False,
        dataloader_drop_last         = True,
    )

    # ---------- custom Trainer ----------
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
            compute_metrics = lambda e: compute_metrics_qwen_final(e, tokenizer),
            # callbacks       = [EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.001)],
        )
    
    if is_main_process():
        print("data process validation...")
        verify_process_func()
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    torch.cuda.empty_cache()
    # ---------- save ----------
    if is_main_process():
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
 
