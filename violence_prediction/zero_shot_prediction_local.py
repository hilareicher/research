from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import pandas as pd
import re
import csv
from collections import Counter
from tqdm import tqdm


# New function for code block and label extraction
def parse_prediction(prediction):
    """Extracts the code block and label from raw LLM output."""
    # Extract the python code block if present
    if "```python" in prediction:
        code_block = prediction.split("```python")[-1].split("```")[0]
    else:
        code_block = prediction
    # Find the High/Medium/Low label
    match = re.search(r'["\'](High|Medium|Low)["\']', code_block, re.IGNORECASE)
    label = match.group(1).capitalize() if match else "Unknown"
    return code_block.strip(), label

def read_emr(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def construct_prompt(emr_text):
    prompt = f"""
You are an expert in psychiatric risk assessment, specializing in predicting violent behavior in psychiatric in-patients based on emergency department medical records.

## Task
Your goal is to analyze the following emergency medical record (EMR) of a psychiatric patient and determine whether they are likely to exhibit violent behavior towards others or in general. Violent behavior is defined based on the following criteria:

### Definition of Violent Behavior
תקופות פיזית חמושת כלפי זולת - קטגוריה זו כוללת תקופות שמתבצעת באופן פיזי (ישנה קטגוריה נפרדת לתוקפנות מינית) ולא פשוט דחפים. למשל בטויים:
תקופה פיזית, אלימות פיזית, אלימות כלפי אחרים, תקף, בתוקפנות כלפי הזולת, הכה, הרביץ, דחף, בעט, חבט, זרק, דקר, נשך, חנק, שרף, התקוטט, מעורב בקטטה, רצח, הרג

### Patient's Emergency Medical Record (EMR):
{emr_text}

## Instructions
- Based on the given EMR, assess the likelihood of the patient exhibiting violent behavior.
- Consider any relevant indications, history, or behavioral patterns described in the EMR.
- Provide a reasoned explanation for your assessment.
- Your response should be structured as follows:
  1. **Key Indicators from the EMR**
  2. **Likelihood of violent behavior** (High / Medium / Low)
  3. **Brief Explanation**

Provide your assessment below:

Please respond only with the following Python variable assignments, and do not repeat the prompt or any additional commentary:
```python
key_indicators_from_emr = [...]
likelihood_of_violent_behavior = "<High/Medium/Low>"
explanation = "..."
```
"""
    return prompt

def query_llm(prompt, tokenizer, model, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, repetition_penalty=1.0, temperature=0.1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def process_emrs_from_csv(csv_path, text_dirs, tokenizer, model, device):
    df = pd.read_csv(csv_path)
    print (f"Loaded {len(df)} admission records from {csv_path}")
    results = []
    filenames = df["FILENAME"].unique()
    total = len(filenames)
    if device == "mps":
        filenames = filenames[:5]
    for idx, filename in enumerate(tqdm(filenames, desc="Processing EMRs"), start=1):
        txt_filename = filename + ".txt"
        file_path = None
        # percent calculation and verbose print removed; tqdm shows progress
        for directory in text_dirs:
            candidate_path = os.path.join(directory, txt_filename)
            if os.path.exists(candidate_path):
                file_path = candidate_path
                break

        if file_path:
            print(f"\nFound {txt_filename} ({idx}/{total})")
            emr_text = read_emr(file_path)
            prompt = construct_prompt(emr_text)
            # print(f"\nPrompt for {txt_filename}:\n")
            # print(prompt)
            # print("\n" + "=" * 80 + "\n")
            # Run prediction 5 times and take majority vote
            labels = []
            for _ in range(5):
                pred = query_llm(prompt, tokenizer, model, device)
                _, lab = parse_prediction(pred)
                labels.append(lab)
            majority_label = Counter(labels).most_common(1)[0][0]
            print(f"Labels for {txt_filename}: {labels} -> Majority: {majority_label}")
            results.append({
                "DEMOG_REC_ID": int(df.loc[df["FILENAME"] == filename, "DEMOG_REC_ID"].iloc[0]),
                "FILENAME": txt_filename,
                "LABEL": majority_label
            })
        else:
            print(f"*** File {txt_filename} not found in either directory. *** ")

    # Write results to predictions.csv
    output_path = os.path.join(os.path.dirname(__file__), "predictions.csv")
    with open(output_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["DEMOG_REC_ID", "FILENAME", "LABEL"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved predictions to {output_path}")

    # --- label distribution ---
    preds_df = pd.read_csv(output_path)
    label_counts = preds_df["LABEL"].value_counts().sort_index()
    print("\nLabel distribution:")
    print(label_counts.to_string())
    dist_path = os.path.join(os.path.dirname(__file__), "label_distribution.csv")
    label_counts.rename_axis("LABEL").reset_index(name="COUNT").to_csv(dist_path, index=False)
    print(f"Saved label distribution to {dist_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["local", "server"], default="local", help="Choose environment: local or server")
    args = parser.parse_args()

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    device = 0
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16).to(device)

    if args.env == "server":
        text_dirs = [
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT1-1745402454",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT2-1745402454",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT3-1745406167",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT4-1745406168",
            "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/Data/Violence-Risk-143856ac/TEXT5-1745406168",
        ]
    else:
        text_dirs = [
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT1-1745402454",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT2-1745402454",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT3-1745406167",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT4-1745406168",
            "/Users/hilac/Downloads/mazor/research-main/uploads/TEXT5-1745406168",
        ]

    csv_path = os.path.join(os.path.dirname(__file__), "admission_records.csv")
    process_emrs_from_csv(csv_path, text_dirs, tokenizer, model, device)