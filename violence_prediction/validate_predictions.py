#!/usr/bin/env python3
import os
import pandas as pd
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import time
import sys
import json
import gc  # Add garbage collection

# --- Tee class for logging to both console and file ---
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

# timing accumulators for profiling
timers = {
    'read': 0.0,
    'prompt': 0.0,
    'tokenize': 0.0,
    'generate': 0.0,
    'parse': 0.0,
    'misc': 0.0
}


from tqdm import tqdm
from prompt_template import CHECK_PROMPT_TEMPLATE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef
)
from sklearn.metrics import precision_recall_fscore_support

def parse_yes_no(resp: str) -> str:
    """
    Extracts the boolean 'actual' field from the model output in JSON format.
    """
    try:
        # Look for JSON object with optional whitespace and comments
        json_match = re.search(r'\{[^}]*"actual":\s*(true|false)[^}]*\}', resp, re.IGNORECASE | re.DOTALL)
        if json_match:
            # Clean up any trailing commas or comments before parsing
            json_str = re.sub(r'//.*$', '', json_match.group(), flags=re.MULTILINE)
            json_str = re.sub(r',(\s*})', r'\1', json_str)
            parsed = json.loads(json_str)
            if isinstance(parsed.get("actual"), bool):
                return parsed["actual"]
    except Exception as e:
        print(f"⚠️ JSON parse failed: {e}")
        print(f"Raw response was: {resp}")
    return "InvalidFormat"

def parse_justification(resp: str) -> str:
    """
    Extracts the snippet assigned to `justification = "..."` from the model output,
    capturing everything between the first pair of quotes.
    """
    for line in resp.splitlines():
        if "justification" in line:
            # get the part after the first '='
            parts = line.split("=", 1)
            if len(parts) < 2:
                continue
            val = parts[1].strip()
            # strip surrounding backticks
            if val.startswith("```") and val.endswith("```"):
                val = val.strip("`")
            # strip matching surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            # unescape any escaped quotes
            val = val.replace('\\"', '"').replace("\\'", "'")
            # collapse whitespace and return
            return " ".join(val.split())
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["local","server"], default="local",
                        help="Choose environment: local or server")
    parser.add_argument("--model_name", default=None, help="HuggingFace model name to use")
    args = parser.parse_args()
    start_time = time.time()

    # --- paths setup ---
    if args.env == "server":
        metadata_dirs = [
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TXT1_META_DATA-1745406168",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT2_META_DATA-1745406167",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT3_META_DATA-1745406168",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT4_META_DATA-1745406168",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TXT5_META_DATA-1745406168",
        ]
        text_dirs = [
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT1-1745402454",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT2-1745402454",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT3-1745406167",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT4-1745406168",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT5-1745406168",
        ]
        model_path = "mistralai/Mistral-7B-Instruct-v0.2"
        device = "cuda"
    else:
        metadata_dirs = [
            "/Users/hilac/Downloads/mazor/research-main/uploads/TXT1_META_DATA-1745406168",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT2_META_DATA-1745406167",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT3_META_DATA-1745406168",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT4_META_DATA-1745406168",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TXT5_META_DATA-1745406168",
        ]
        text_dirs = [
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT1-1745402454",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT2-1745402454",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT3-1745406167",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT4-1745406168",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT5-1745406168",
        ]
        model_path = "mistralai/Mistral-7B-Instruct-v0.2"
        device = "mps"

    # --- load predictions ---
    script_dir = os.path.dirname(__file__)
    # ensure we have a dedicated results folder
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    # Redirect all print output (stdout and stderr) to a runtime log file, but also show on console
    run_log_path = os.path.join(results_dir, "run.log")
    run_log_f = open(run_log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = Tee(orig_stdout, run_log_f)
    sys.stderr = Tee(orig_stderr, run_log_f)
    preds_df = pd.read_csv(os.path.join(script_dir, "predictions.csv"))

    # --- load metadata ---
    meta_frames = []
    for d in metadata_dirs:
        for f in os.listdir(d):
            if f.endswith(".csv"):
                meta_frames.append(pd.read_csv(os.path.join(d, f)))
    meta_df = pd.concat(meta_frames, ignore_index=True).drop_duplicates()

    # allow overriding the local model with a HF hub model name
    model_source = args.model_name if args.model_name else model_path
    print (f"device: {device}, model source: {model_source}")
    # --- prepare LLM ---
    # Run garbage collection before loading model
    gc.collect()

    print(f"Loading model from {model_source}...")
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",  # Better device management
        low_cpu_mem_usage=True  # Reduce memory usage during loading
    )

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    results = []
    for _, row in tqdm(preds_df.iterrows(), total=len(preds_df), desc="Admissions"):
        # Run garbage collection at the start of each major iteration
        gc.collect()

        demog = row["DEMOG_REC_ID"]
        adm_fn = row["FILENAME"].replace(".txt", "")
        pred_label = row["LABEL"]

        # find admission number for the admission file
        adm_meta = meta_df[
            (meta_df["DEMOG_REC_ID"] == demog) &
            (meta_df["FILENAME"] == adm_fn)
        ]
        if adm_meta.empty:
            print(f"Warning: Admission file {adm_fn} not found in metadata for patient {demog}")
            continue
        admission_no = adm_meta["ADMISSION_NO"].iloc[0]

        # only consider records from the same admission
        other_meta = meta_df[
            (meta_df["DEMOG_REC_ID"] == demog) &
            (meta_df["ADMISSION_NO"] == admission_no) &
            (meta_df["FILENAME"] != adm_fn)
        ]
        other_files = other_meta["FILENAME"].unique().tolist()
        if args.env == "local":
            other_files = other_files[:30]
        actual = False
        actual_file = ""
        justification = ""
        violent_text = ""
        for fn in tqdm(other_files, desc=f"Checking files for {demog}", leave=False):
            txt_fn = fn + ".txt"
            print (f"Checking {txt_fn} for DEMOG_REC_ID {demog}...")
            path = next((os.path.join(td, txt_fn) for td in text_dirs if os.path.exists(os.path.join(td, txt_fn))), None)
            if not path:
                continue

            # --- Profile EMR reading ---
            t0 = time.time()
            with open(path, "r", encoding="utf-8") as f:
                emr_text = f.read()
            print(f"EMR size for {txt_fn}: {len(emr_text)} characters")
            timers['read'] += time.time() - t0

            # --- Skip if EMR text is too long ---
            if len(emr_text) > 2000:
                print(f"Skipping {txt_fn} because EMR size is too large ({len(emr_text)} chars)")
                results.append({
                    "DEMOG_REC_ID": demog,
                    "ADMISSION_FILE": txt_fn,
                    "PREDICTION": pred_label,
                    "ACTUAL": "TooLong",
                    "ACTUAL_FILE": path,
                    "JUSTIFICATION": "",
                    "CONTENT": emr_text
                })
                continue

            # --- Profile prompt construction ---
            t0 = time.time()
            prompt = CHECK_PROMPT_TEMPLATE.format(emr_text=emr_text)
            timers['prompt'] += time.time() - t0

            # --- Profile tokenization ---
            t0 = time.time()
            # No explicit tokenization needed with pipeline; keep timer for compatibility
            timers['tokenize'] += time.time() - t0

            # --- Profile generation ---
            t0 = time.time()
            with torch.no_grad():  # Prevent gradient computation
                raw = gen_pipeline(prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
            print(f"Raw response for {txt_fn}:\n{raw}\n")
            timers['generate'] += time.time() - t0

            # --- Profile parsing ---
            t0 = time.time()
            resp_label = parse_yes_no(raw)
            resp_just = parse_justification(raw)
            timers['parse'] += time.time() - t0

            # Free memory after we're done with the raw response
            del raw
            gc.collect()

            print(f"actual label for {txt_fn} is {resp_label}")
            if resp_label is True:
                actual = True
                actual_file = path
                # extract justification snippet from the EMR text directly
                justification = resp_just
                violent_text = emr_text
                print (f"Justification for {txt_fn}: {justification}")
                timers['misc'] += time.time() - t0_overall_iteration_start
                break
            elif resp_label is False:
                actual = False

            # Account for misc time for this iteration if we didn't break
            timers['misc'] += time.time() - t0_overall_iteration_start

    results.append({
        "DEMOG_REC_ID": demog,
        "ADMISSION_FILE": adm_fn + ".txt",
        "PREDICTION": pred_label,
        "ACTUAL": actual if resp_label != "InvalidFormat" else "InvalidFormat",
        "ACTUAL_FILE": actual_file or "N/A",
        "JUSTIFICATION": justification,
        "CONTENT": violent_text
    })

    # --- write out ---
    out_path = os.path.join(results_dir, "validation_results.csv")
    # write main results including actual file and justification
    results_df = pd.DataFrame(results)
    results_df = results_df[["DEMOG_REC_ID", "ADMISSION_FILE", "PREDICTION", "ACTUAL", "ACTUAL_FILE", "JUSTIFICATION"]]

    # Handle boolean conversion more carefully
    def convert_to_boolean(val):
        if isinstance(val, bool):
            return val
        elif isinstance(val, str) and val in ["TooLong", "InvalidFormat"]:
            return val
        return None

    # Convert values one by one using the safe converter
    results_df["ACTUAL"] = results_df["ACTUAL"].apply(convert_to_boolean)

    results_df.to_csv(out_path, index=False)
    # Count only actual boolean True/False values
    yes_count = sum(1 for x in results_df["ACTUAL"] if x is True)
    no_count = sum(1 for x in results_df["ACTUAL"] if x is False)
    print(f"Saved {len(results)} rows to {out_path} — actual violence=True: {yes_count}, False: {no_count}")

    # שמור קבצים שלא קיבלו תשובה בפורמט תקני או שהיו ארוכים מדי
    invalids_df = pd.DataFrame([r for r in results if r["ACTUAL"] in ["InvalidFormat", "TooLong"]])
    invalids_path = os.path.join(results_dir, "invalid_format_responses.csv")
    invalids_df.to_csv(invalids_path, index=False)
    print(f"\n⚠️ Saved {len(invalids_df)} responses with invalid format or too long to {invalids_path}")

    # --- label distribution ---
    import json
    results_df = pd.read_csv(out_path)
    pred_dist = results_df["PREDICTION"].value_counts().reset_index()
    pred_dist.columns = ["Label", "Count"]
    pred_dist_path = os.path.join(results_dir, "label_distribution_predictions.csv")
    pred_dist.to_csv(pred_dist_path, index=False)
    actual_dist = results_df["ACTUAL"].value_counts(dropna=False).reset_index()
    actual_dist.columns = ["Label", "Count"]
    actual_dist_path = os.path.join(results_dir, "label_distribution_actual.csv")
    actual_dist.to_csv(actual_dist_path, index=False)
    print(f"Prediction label distribution saved to {pred_dist_path}")
    print(f"Actual label distribution saved to {actual_dist_path}")

    # --- compute validation metrics ---
    metrics_df = pd.read_csv(out_path)
    # drop rows with missing predictions
    missing_count = metrics_df["PREDICTION"].isna().sum()
    if missing_count > 0:
        print(f"Warning: {missing_count} predictions missing, excluding from metrics.")
    metrics_df = metrics_df.dropna(subset=["PREDICTION"])
    # map actual labels to binary
    y_true = metrics_df["ACTUAL"].astype("boolean").astype("int")
    # convert likelihood labels to binary predictions (only "High" counts as positive)
    pred_map = {"High": 1, "Medium": 0, "Low": 0}
    y_pred = metrics_df["PREDICTION"].map(pred_map)

    # drop any rows where prediction remains unmapped
    missing_pred = y_pred.isna().sum()
    if missing_pred > 0:
        print(f"Warning: {missing_pred} rows with unmapped PREDICTION values, excluding from metrics.")
        valid = y_pred.notna()
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    print("\nValidation Metrics:")
    print(f"  Accuracy   : {acc:.3f}")
    print(f"  Precision  : {prec:.3f}")
    print(f"  Recall     : {rec:.3f}")
    print(f"  Specificity: {spec:.3f}")
    print(f"  F1 Score   : {f1:.3f}")
    print(f"  MCC        : {mcc:.3f}")

    # --- save metrics to CSV ---
    metrics_dict = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1 Score": f1,
        "MCC": mcc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_csv_path = os.path.join(results_dir, "validation_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Validation metrics saved to {metrics_csv_path}")
    print("\nConfusion Matrix:")
    print(f"  TN={tn}  FP={fp}")
    print(f"  FN={fn}  TP={tp}")
    print("\nClassification Report:")
    # Print single binary precision/recall/f1
    prfs = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    precision_bin, recall_bin, f1_bin, _ = prfs
    print(f"Precision (binary): {precision_bin:.3f}")
    print(f"Recall (binary)   : {recall_bin:.3f}")
    print(f"F1 Score (binary) : {f1_bin:.3f}")
    print(f"Total run time: {time.time() - start_time:.2f} seconds")
    print("\n🕒 Profiling Timers:")
    for name, val in timers.items():
        print(f"  {name.capitalize():<8}: {val:.2f} sec")
    # Close the runtime log file
    run_log_f.close()
    # print(classification_report(y_true, y_pred, target_names=["No","Yes"]))