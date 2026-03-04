#!/usr/bin/env python
# evaluate.py
import argparse, torch, math, Levenshtein,evaluate
from datasets import load_dataset
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import os
from tqdm import tqdm

os.environ["HF_HOME"] = "/eval/hf_cache"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_EVALUATE_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------- helpers ----------
def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

def distinct_n(sent: str, n: int = 2) -> float:
    toks = sent.split()
    if len(toks) < n:
        return 0.0
    ngrams = [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return len(set(ngrams)) / len(ngrams)

def read_and_print_csv(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    print("=" * 80)
    print("EVALUATION RESULTS - Sentence_0 vs Generated_0")
    print("=" * 80)
    print(f"Total text pairs evaluated: {len(df)}")
    print()
    
    # 1. Cosine Similarity
    cosine_values = df["Cosine"].values
    print("📊 COSINE SIMILARITY")
    print(f"   Average: {cosine_values.mean():.4f}")
    print(f"   Perfect Match (= 1.0): {sum(1 for x in cosine_values if x >= 0.999) / len(cosine_values) * 100:.2f}%")
    print(f"   High Similarity (> 0.9): {sum(1 for x in cosine_values if x > 0.9) / len(cosine_values) * 100:.2f}%")
    print(f"   Good Similarity (> 0.5): {sum(1 for x in cosine_values if x > 0.5) / len(cosine_values) * 100:.2f}%")
    print(f"   Low Similarity (< 0.3): {sum(1 for x in cosine_values if x < 0.3) / len(cosine_values) * 100:.2f}%")
    print()
    
    # 2. ROUGE-1
    rouge1_values = df["ROUGE1"].values
    print("📊 ROUGE-1 (Unigram Overlap)")
    print(f"   Average: {rouge1_values.mean():.4f}")
    print(f"   Perfect Match (= 1.0): {sum(1 for x in cosine_values if x >= 0.999) / len(cosine_values) * 100:.2f}%")
    print(f"   Excellent (> 0.9): {sum(1 for x in rouge1_values if x > 0.8) / len(rouge1_values) * 100:.2f}%")
    print(f"   Good (> 0.5): {sum(1 for x in rouge1_values if x > 0.5) / len(rouge1_values) * 100:.2f}%")
    print(f"   Fair (> 0.3): {sum(1 for x in rouge1_values if x > 0.3) / len(rouge1_values) * 100:.2f}%")
    print()
    
    # 3. ROUGE-L
    rougeL_values = df["ROUGEL"].values
    print("📊 ROUGE-L (Longest Common Subsequence)")
    print(f"   Average: {rougeL_values.mean():.4f}")
    print(f"   Perfect Match (= 1.0): {sum(1 for x in rougeL_values if x >= 0.999) / len(rougeL_values) * 100:.2f}%")
    print(f"   Excellent (> 0.9): {sum(1 for x in rougeL_values if x > 0.8) / len(rougeL_values) * 100:.2f}%")
    print(f"   Good (> 0.5): {sum(1 for x in rougeL_values if x > 0.5) / len(rougeL_values) * 100:.2f}%")
    print(f"   Fair (> 0.3): {sum(1 for x in rougeL_values if x > 0.3) / len(rougeL_values) * 100:.2f}%")
    print()
    
    # 4. Edit Distance
    edit_similarity_values = df["EditSimilarity"].values
    raw_edit_values = df["EditDistance"].values  
    norm_edit_values = df["NormalizedEditDistance"].values
    
    print("📊 EDIT DISTANCE ANALYSIS")
    print(f"   📈 Edit Similarity (0-1, higher=better):")
    print(f"      Average: {edit_similarity_values.mean():.4f}")
    print(f"      Nearly Identical (> 0.9): {sum(1 for x in edit_similarity_values if x > 0.9) / len(edit_similarity_values) * 100:.2f}%")
    print(f"      Highly Similar (> 0.8): {sum(1 for x in edit_similarity_values if x > 0.8) / len(edit_similarity_values) * 100:.2f}%")
    print()
    print(f"   📉 Raw Edit Distance (0+, lower=better):")
    print(f"      Average: {raw_edit_values.mean():.2f}")
    print(f"      Perfect Match (= 0): {sum(1 for x in raw_edit_values if x == 0) / len(raw_edit_values) * 100:.2f}%")
    print(f"      Very Close (≤ 1): {sum(1 for x in raw_edit_values if x <= 1) / len(raw_edit_values) * 100:.2f}%")
    print(f"      Close (≤ 3): {sum(1 for x in raw_edit_values if x <= 3) / len(raw_edit_values) * 100:.2f}%")
    print(f"      Moderate (≤ 5): {sum(1 for x in raw_edit_values if x <= 5) / len(raw_edit_values) * 100:.2f}%")
    print()
    print(f"   📊 Normalized Edit Distance (0-1, lower=better):")
    print(f"      Average: {norm_edit_values.mean():.4f}")
    print(f"      Perfect Match (= 0.0): {sum(1 for x in norm_edit_values if abs(x) < 1e-6) / len(norm_edit_values) * 100:.2f}%")
    print(f"      Excellent (≤ 0.1): {sum(1 for x in norm_edit_values if x <= 0.1) / len(norm_edit_values) * 100:.2f}%")
    print(f"      Good (≤ 0.2): {sum(1 for x in norm_edit_values if x <= 0.2) / len(norm_edit_values) * 100:.2f}%")
    print(f"      Fair (≤ 0.3): {sum(1 for x in norm_edit_values if x <= 0.3) / len(norm_edit_values) * 100:.2f}%")
    print()
    
    # 5. Jaccard Similarity 
    jaccard_values = df["Jaccard"].values
    print("📊 JACCARD SIMILARITY (Word Set Overlap)")
    print(f"   Average: {jaccard_values.mean():.4f}")
    print(f"   High Overlap (> 0.8): {sum(1 for x in jaccard_values if x > 0.8) / len(jaccard_values) * 100:.2f}%")
    print(f"   Good Overlap (> 0.5): {sum(1 for x in jaccard_values if x > 0.5) / len(jaccard_values) * 100:.2f}%")
    print(f"   Some Overlap (> 0.3): {sum(1 for x in jaccard_values if x > 0.3) / len(jaccard_values) * 100:.2f}%")
    print()
    
    # 6. Distinct-2 
    distinct2_values = df["distinct2"].values
    print("📊 DISTINCT-2 (Generated Text Diversity)")
    print(f"   Average: {distinct2_values.mean():.4f}")
    print(f"   High Diversity (> 0.8): {sum(1 for x in distinct2_values if x > 0.8) / len(distinct2_values) * 100:.2f}%")
    print(f"   Medium Diversity (> 0.5): {sum(1 for x in distinct2_values if x > 0.5) / len(distinct2_values) * 100:.2f}%")
    print(f"   Low Diversity (< 0.3): {sum(1 for x in distinct2_values if x < 0.3) / len(distinct2_values) * 100:.2f}%")
    print()
    
    # overall
    print("🎯 ATTACK SUCCESS ANALYSIS")
    high_cosine = sum(1 for x in cosine_values if x > 0.5)
    high_rouge1 = sum(1 for x in rouge1_values if x > 0.5)
    high_edit = sum(1 for x in edit_similarity_values if x > 0.7)
    
    print(f"   Strong Attack Success (Cosine > 0.5): {high_cosine / len(df) * 100:.2f}%")
    print(f"   Multi-metric Success (Cosine>0.5 & ROUGE1>0.5 & Edit>0.7): ", end="")
    
    success_count = 0
    for i in range(len(df)):
        if cosine_values[i] > 0.5 and rouge1_values[i] > 0.5 and edit_similarity_values[i] > 0.7:
            success_count += 1
    print(f"{success_count / len(df) * 100:.2f}%")
    
    print("=" * 80)

# ---------- main ----------
def main(args):
    acc = Accelerator()
    device = acc.device
    if acc.is_main_process:
        print(f"[evaluate] world_size={acc.num_processes}, device={device}")

    st = SentenceTransformer("/eval/hf_cache/models/all-MiniLM-L12-v1", device=device).eval()
    
    if torch.cuda.is_available() and device == "cuda":
        st = st.to(device, dtype=torch.bfloat16)

    rouge = evaluate.load("/eval/hf_cache/modules/rouge")

    ds = load_dataset("csv", data_files={"full": args.input})["full"]
    ds = ds.rename_columns({args.ref_col: "ref", args.hyp_col: "hyp"})
  
    ds = ds.filter( 
    lambda x: x["ref"] and x["hyp"],  
    num_proc=acc.num_processes
    )
    

    if acc.is_main_process:
        print(f"[evaluate] dataset size: {len(ds)}")
        print(f"[evaluate] columns: {ds.column_names}")
    ds = ds.shard(acc.num_processes, acc.process_index)

    def _metric(example):
        ref, hyp = example["ref"] or "", example["hyp"] or ""

        # Sentence-BERT cosine
        with torch.no_grad():
            emb = st.encode([ref, hyp], convert_to_tensor=True,
                            normalize_embeddings=True, device=device)
            cosine = torch.nn.functional.cosine_similarity(
                emb[0], emb[1], dim=0).item()

        # ROUGE
        rouge_res = rouge.compute(predictions=[hyp], references=[ref], use_aggregator=False)
        rouge1, rougeL = rouge_res["rouge1"][0], rouge_res["rougeL"][0]

        # Levenshtein 
        raw_edit_distance = Levenshtein.distance(ref, hyp)
        maxlen = max(len(ref), len(hyp)) or 1
        edit_similarity = (maxlen - raw_edit_distance) / maxlen  
        normalized_edit_distance = raw_edit_distance / maxlen    

        # Jaccard
        jac = jaccard(ref, hyp)

        # distinct-2
        d2 = distinct_n(hyp, 2)

        return {
            "Cosine": cosine, "ROUGE1": rouge1, "ROUGEL": rougeL,
            "EditSimilarity": edit_similarity, "EditDistance": raw_edit_distance, 
            "NormalizedEditDistance": normalized_edit_distance, "Jaccard": jac,
            "distinct2": d2
        }


    results = []
    for i, example in enumerate(tqdm(ds)):
        try:
            result = _metric(example)
            
            result.update(example)
            results.append(result)
            
            # if acc.is_main_process and (i + 1) % 100 == 0:
            #     print(f"Processed {i + 1} examples...")
        except Exception as e:
            if acc.is_main_process:
                print(f"Error processing example {i}: {e}")
            continue


    temp_file = f"{args.output}.tmp.{acc.process_index}"
    df_local = pd.DataFrame(results)
    df_local.to_csv(temp_file, index=False)
    

    acc.wait_for_everyone()
    

    if acc.is_main_process:
        all_results = []
        for rank in range(acc.num_processes):
            temp_file = f"{args.output}.tmp.{rank}"
            if os.path.exists(temp_file):
                df_temp = pd.read_csv(temp_file)
                all_results.append(df_temp)
                os.remove(temp_file)  
        
        if all_results:
            df_final = pd.concat(all_results, ignore_index=True)
            df_final.to_csv(args.output, index=False)
            print(f"[evaluate] saved → {args.output}")
            read_and_print_csv(args.output)
        else:
            print("No results to save")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with Sentence_0 & Generated_0")
    ap.add_argument("--output", required=True, help="Metrics CSV to write")
    ap.add_argument("--ref_col", default="Sentence_0")
    ap.add_argument("--hyp_col", default="Generated_0")
    args = ap.parse_args()
    main(args)
