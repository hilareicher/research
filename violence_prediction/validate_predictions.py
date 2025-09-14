#!/usr/bin/env python3
import os
import pandas as pd
import argparse
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re
import time
import sys
import json

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
    Handles various forms of true/false values.
    """
    try:
        # Look for JSON object with optional whitespace and comments
        json_match = re.search(r'\{[^}]*"actual":\s*(true|false)[^}]*\}', resp, re.IGNORECASE | re.DOTALL)
        if json_match:
            # Clean up any trailing commas or comments before parsing
            json_str = re.sub(r'//.*$', '', json_match.group(), flags=re.MULTILINE)
            json_str = re.sub(r',(\s*})', r'\1', json_str)

            # Handle case variations before parsing
            json_str = re.sub(r':\s*true\s*([,}])', r':true\1', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r':\s*false\s*([,}])', r':false\1', json_str, flags=re.IGNORECASE)

            parsed = json.loads(json_str)
            actual_val = parsed.get("actual")

            # Handle both boolean and string representations
            if isinstance(actual_val, bool):
                return actual_val
            elif isinstance(actual_val, str):
                val_lower = actual_val.lower()
                if val_lower in ('true', 'yes', '1'):
                    return True
                elif val_lower in ('false', 'no', '0'):
                    return False
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
    parser.add_argument("--manual_csv", default=None,
                        help="Path to CSV with a 'manual inspection' column to correct ACTUAL and to drive resume runs")
    parser.add_argument("--resume_false_positives", action="store_true",
                        help="Only re-check admissions where prior ACTUAL=Yes but manual inspection=False; skip the previously flagged file(s)")
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
    # --- normalize IDs and filenames so resume filtering matches ---
    def _to_int_id(x):
        try:
            return int(float(x))
        except Exception:
            return None
    preds_df["DEMOG_REC_ID_norm"] = preds_df["DEMOG_REC_ID"].apply(_to_int_id)
    # ensure we have a `.txt` filename column to compare with manual CSV
    preds_df["FILENAME_txt"] = preds_df["FILENAME"].astype(str).str.replace(".txt", "", regex=False) + ".txt"
    # --- optional: prepare resume sets from manual CSV ---
    fp_pairs = set()   # {(DEMOG_REC_ID, ADMISSION_FILE)} admissions to re-check
    skip_map = {}      # {DEMOG_REC_ID: set([basename_without_ext, ...])} files to skip when scanning follow-up
    if args.manual_csv and os.path.exists(args.manual_csv):
        try:
            manual_df = pd.read_csv(args.manual_csv)
            # Normalize expected columns
            if "ADMISSION_FILE" not in manual_df.columns and "ADMISSION_FILE_x" in manual_df.columns:
                manual_df = manual_df.rename(columns={"ADMISSION_FILE_x": "ADMISSION_FILE"})
            # Identify rows that were labeled ACTUAL=Yes previously but refuted by manual inspection
            mask_fp = manual_df["ACTUAL"].astype(str).str.lower().isin(["yes","true"]) & (manual_df["manual inspection"] == False)
            fps = manual_df.loc[mask_fp, ["DEMOG_REC_ID","ADMISSION_FILE","ACTUAL_FILE"]].dropna(subset=["DEMOG_REC_ID","ADMISSION_FILE"])
            def _basename_txt(p):
                p = str(p).strip()
                base = os.path.basename(p)
                return base if base.endswith(".txt") else base + ".txt"
            for _, rr in fps.iterrows():
                demog = _to_int_id(rr["DEMOG_REC_ID"])  # robust cast
                adm_file_txt = _basename_txt(rr["ADMISSION_FILE"])  # basename + .txt
                fp_pairs.add((demog, adm_file_txt))
                prev_path = str(rr.get("ACTUAL_FILE", "")).strip()
                if prev_path and prev_path != "N/A":
                    prev_base = os.path.splitext(os.path.basename(prev_path))[0]
                    skip_map.setdefault(demog, set()).add(prev_base)
            print(f"Loaded manual CSV for resume: {len(fp_pairs)} admissions to re-check")
        except Exception as e:
            print(f"Warning: failed to parse manual_csv {args.manual_csv}: {e}")

    # --- print resume admissions to process (detailed listing) ---
    if args.resume_false_positives and fp_pairs:
        # Count how many rows in predictions match the resume keys
        preds_keys = set((rid, fn) for rid, fn in zip(preds_df["DEMOG_REC_ID_norm"], preds_df["FILENAME_txt"]))
        to_process = len(preds_keys & fp_pairs)
        print(f"[resume] Will process {to_process} admissions out of {len(fp_pairs)} requested (based on predictions.csv overlap)")
        intersection = sorted(list(preds_keys & fp_pairs))
        if not intersection:
            print("[resume] No matching admissions found between manual CSV and predictions.csv — nothing to run.")
        else:
            print("[resume] Admissions scheduled to run (DEMOG_REC_ID, ADMISSION_FILE):")
            max_show = 50
            for i, (rid, afile) in enumerate(intersection[:max_show], 1):
                print(f"  {i:>3}. {rid}, {afile}")
            if len(intersection) > max_show:
                print(f"  ... and {len(intersection) - max_show} more")

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
    # Move results.append outside the loop - it's currently only saving the last admission
    for _, row in tqdm(preds_df.iterrows(), total=len(preds_df), desc="Admissions"):
        demog = row["DEMOG_REC_ID_norm"]
        adm_fn = row["FILENAME"].replace(".txt", "")
        adm_txt = adm_fn + ".txt"
        pred_label = row["LABEL"]

        # If resuming, only process admissions that were refuted positives
        if args.resume_false_positives:
            if (demog, adm_txt) not in fp_pairs:
                continue

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
        # Remove previously flagged (but refuted) file(s) for this patient
        if demog in skip_map and skip_map[demog]:
            other_files = [fn for fn in other_files if fn not in skip_map[demog]]
        if args.env == "local":
            other_files = other_files[:30]
        actual = False
        actual_file = ""
        justification = ""
        violent_text = ""
        resp_label = False  # Initialize with a default value

        for fn in tqdm(other_files, desc=f"Checking files for {demog}", leave=False):
            # Start timing the overall iteration
            t0_overall_iteration_start = time.time()

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
            if len(emr_text) > 6000:
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

            # Simply remove the raw response when done
            del raw

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

        # Store result for this admission (moved inside the main loop)
        results.append({
            "DEMOG_REC_ID": demog,
            "ADMISSION_FILE": adm_txt,
            "PREDICTION": pred_label,
            "ACTUAL": actual if resp_label != "InvalidFormat" else "InvalidFormat",
            "ACTUAL_FILE": actual_file or "N/A",
            "JUSTIFICATION": justification,
            "CONTENT": violent_text
        })

    # --- write out ---
    out_filename = "validation_results_resumed.csv" if args.resume_false_positives else "validation_results.csv"
    out_path = os.path.join(results_dir, out_filename)
    # write main results including actual file and justification
    results_df = pd.DataFrame(results)
    results_df = results_df[["DEMOG_REC_ID", "ADMISSION_FILE", "PREDICTION", "ACTUAL", "ACTUAL_FILE", "JUSTIFICATION"]]

    # Handle boolean conversion more carefully
    def convert_to_boolean(val):
        """
        Converts various true/false representations to proper boolean values.
        Preserves special string values like TooLong and InvalidFormat.
        """
        if isinstance(val, bool):
            return val
        elif isinstance(val, str):
            if val in ["TooLong", "InvalidFormat"]:
                return val
            val_lower = val.lower()
            if val_lower in ('true', 'yes', '1', 't', 'y'):
                return True
            elif val_lower in ('false', 'no', '0', 'f', 'n'):
                return False
        return None

    # Convert values one by one using the safe converter
    results_df["ACTUAL"] = results_df["ACTUAL"].apply(convert_to_boolean)

    results_df.to_csv(out_path, index=False)
    # Count only actual boolean True/False values
    yes_count = sum(1 for x in results_df["ACTUAL"] if x is True)
    no_count = sum(1 for x in results_df["ACTUAL"] if x is False)
    print(f"Saved {len(results)} rows to {out_path} — actual violence=True: {yes_count}, False: {no_count}")

    # --- recompute metrics on full data with manual corrections (prefers full results in results/) ---
    try:
        if args.manual_csv and os.path.exists(args.manual_csv):
            full_results_path = os.path.join(results_dir, "validation_results.csv")
            eval_path = full_results_path if os.path.exists(full_results_path) else out_path
            all_df = pd.read_csv(eval_path)

            manual_df = pd.read_csv(args.manual_csv)
            # Keep only necessary columns if present
            keep_cols = [c for c in ["DEMOG_REC_ID","ADMISSION_FILE","manual inspection"] if c in manual_df.columns]
            manual_df = manual_df[keep_cols]

            merged = all_df.merge(manual_df, on=["DEMOG_REC_ID","ADMISSION_FILE"], how="left")

            def corrected_actual(row):
                act = str(row["ACTUAL"]).strip().lower()
                mi = row.get("manual inspection")
                if act in ("yes","true"):
                    # If manual inspection explicitly False -> flip to No, else keep Yes
                    return 1 if mi is True else 0
                return 0  # No/False/other -> 0

            merged["y_true_manual"] = merged.apply(corrected_actual, axis=1).astype(int)

            # Map predictions for strict/relaxed thresholds
            pred_map_strict  = {"High":1, "Medium":0, "Low":0}
            pred_map_relaxed = {"High":1, "Medium":1, "Low":0}

            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

            def compute_all(y_true, pred_map):
                y_pred = merged["PREDICTION"].map(pred_map).fillna(0).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                mcc = matthews_corrcoef(y_true, y_pred)
                return {"Accuracy":acc, "Precision":prec, "Recall":rec, "Specificity":spec, "F1 Score":f1, "MCC":mcc, "TN":tn, "FP":fp, "FN":fn, "TP":tp}

            strict_metrics  = compute_all(merged["y_true_manual"], pred_map_strict)
            relaxed_metrics = compute_all(merged["y_true_manual"], pred_map_relaxed)

            corrected_path = os.path.join(results_dir, "validation_metrics_corrected.csv")
            pd.DataFrame([
                dict(Threshold="Strict", **strict_metrics),
                dict(Threshold="Relaxed", **relaxed_metrics),
            ]).to_csv(corrected_path, index=False)
            print(f"Corrected metrics (manual) saved to {corrected_path}")
    except Exception as e:
        print(f"Warning: failed to compute corrected metrics: {e}")

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
    try:
        metrics_df = pd.read_csv(out_path)
        print(f"\nStarting metrics computation with {len(metrics_df)} total samples")
        print(f"Initial ACTUAL value counts:\n{metrics_df['ACTUAL'].value_counts(dropna=False)}")

        # First, filter out any non-boolean values (TooLong, InvalidFormat)
        valid_responses = ~metrics_df["ACTUAL"].isin(["TooLong", "InvalidFormat"])
        metrics_df = metrics_df[valid_responses]
        print(f"After removing special values: {len(metrics_df)} samples")

        # drop rows with missing predictions or NaN values
        metrics_df = metrics_df.dropna(subset=["PREDICTION", "ACTUAL"])
        missing_count = len(valid_responses) - len(metrics_df)
        if missing_count > 0:
            print(f"Warning: {missing_count} predictions missing or invalid, excluding from metrics.")

        # Convert ACTUAL to numeric (True=1, False=0) handling string representations
        def convert_to_binary(val):
            """Convert various boolean representations to binary (0/1) values"""
            try:
                if pd.isna(val):
                    return None
                if isinstance(val, bool):
                    return 1 if val else 0
                if isinstance(val, (int, float)):
                    if val == 1:
                        return 1
                    if val == 0:
                        return 0
                    return None
                if isinstance(val, str):
                    val_lower = val.lower().strip()
                    if val_lower in ('true', 'yes', '1', 't', 'y', 'true.0', '1.0'):
                        return 1
                    if val_lower in ('false', 'no', '0', 'f', 'n', 'false.0', '0.0'):
                        return 0
                return None
            except Exception as e:
                print(f"Warning: Error converting value '{val}' ({type(val)}): {str(e)}")
                return None

        y_true = metrics_df["ACTUAL"].apply(convert_to_binary)
        print(f"After binary conversion: {len(y_true)} samples")
        print(f"Binary value counts:\n{y_true.value_counts(dropna=False)}")

        # Remove any rows where conversion failed (resulted in None/NaN)
        valid_true = y_true.notna()
        y_true = y_true[valid_true]
        metrics_df = metrics_df[valid_true]

        # convert likelihood labels to binary predictions (only "High" counts as positive)
        pred_map = {"High": 1, "Medium": 0, "Low": 0}
        y_pred = metrics_df["PREDICTION"].map(pred_map)
        print(f"Prediction value counts:\n{y_pred.value_counts(dropna=False)}")

        # drop any rows where prediction remains unmapped
        valid_pred = y_pred.notna()
        y_true = y_true[valid_pred]
        y_pred = y_pred[valid_pred]
        missing_pred = (~valid_pred).sum()
        if missing_pred > 0:
            print(f"Warning: {missing_pred} rows with unmapped PREDICTION values, excluding from metrics.")

        # Final validation before computing metrics
        if len(y_true) != len(y_pred):
            raise ValueError(f"Mismatch in lengths: y_true({len(y_true)}) != y_pred({len(y_pred)})")

        # Verify we have both positive and negative samples
        n_pos = (y_true == 1).sum()
        n_neg = (y_true == 0).sum()
        print(f"\nFinal dataset composition:")
        print(f"Positive samples: {n_pos}")
        print(f"Negative samples: {n_neg}")

        # Only compute metrics if we have valid data
        if len(y_true) > 0 and len(y_pred) > 0 and n_pos + n_neg == len(y_true):
            print(f"\nComputing metrics on {len(y_true)} valid samples...")

            # Convert to numpy arrays for sklearn
            y_true_arr = np.array(y_true, dtype=int)
            y_pred_arr = np.array(y_pred, dtype=int)

            acc = accuracy_score(y_true_arr, y_pred_arr)
            prec = precision_score(y_true_arr, y_pred_arr, zero_division=0)
            rec = recall_score(y_true_arr, y_pred_arr, zero_division=0)
            f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            mcc = matthews_corrcoef(y_true_arr, y_pred_arr)

            print("\nValidation Metrics:")
            print(f"  Total valid samples: {len(y_true)}")
            print(f"  Accuracy   : {acc:.3f}")
            print(f"  Precision  : {prec:.3f}")
            print(f"  Recall     : {rec:.3f}")
            print(f"  Specificity: {spec:.3f}")
            print(f"  F1 Score   : {f1:.3f}")
            print(f"  MCC        : {mcc:.3f}")

            # --- save metrics to CSV ---
            metrics_dict = {
                "Total_Samples": len(y_true),
                "Positive_Samples": n_pos,
                "Negative_Samples": n_neg,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "Specificity": spec,
                "F1_Score": f1,
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
            prfs = precision_recall_fscore_support(y_true_arr, y_pred_arr, average='binary', zero_division=0)
            precision_bin, recall_bin, f1_bin, _ = prfs
            print(f"Precision (binary): {precision_bin:.3f}")
            print(f"Recall (binary)   : {recall_bin:.3f}")
            print(f"F1 Score (binary) : {f1_bin:.3f}")
        else:
            print("\nWarning: Invalid or insufficient data for computing metrics")
            print(f"Total samples: {len(y_true)}")
            print(f"Positive samples: {n_pos}")
            print(f"Negative samples: {n_neg}")

    except Exception as e:
        print(f"\nError during metrics computation: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

    print(f"Total run time: {time.time() - start_time:.2f} seconds")
    print("\n🕒 Profiling Timers:")
    for name, val in timers.items():
        print(f"  {name.capitalize():<8}: {val:.2f} sec")
    # Close the runtime log file
    run_log_f.close()
    # print(classification_report(y_true, y_pred, target_names=["No","Yes"]))