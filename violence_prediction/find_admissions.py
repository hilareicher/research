import os
import argparse
import pandas as pd

# Choose directories based on environment
parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["local", "server"], default="local", help="Choose environment: local or server")
args = parser.parse_args()

if args.env == "server":
    directories = [
        "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/lynx-workspace/data/Violence-Risk-143856ac/TEXT1_META_DATA-1745406168",
        "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/lynx-workspace/data/Violence-Risk-143856ac/TEXT2_META_DATA-1745406167",
    ]
    text_dirs = [
        "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/lynx-workspace/data/Violence-Risk-143856ac/TEXT1-1745402454",
        "/home/hilareicher.mail.tau.ac.il/Desktop/local_share/lynx-workspace/data/Violence-Risk-143856ac/TEXT2-1745402454"
    ]
else:
    directories = [
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

# Define admission criteria
admission_depts = {"רפואה דחופה", "חדר מיון"}
target_module = "ADMDIS"
fields_to_compare = ["MODULE", "SUB_MODULE"]

# Step 1: Load all CSVs
all_rows = []
for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_csv(filepath)
                all_rows.append(df)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

if not all_rows:
    print("No data loaded.")
    exit()

# Remove exact duplicate rows after concatenation
full_df = pd.concat(all_rows, ignore_index=True).drop_duplicates()
print(f"\nTotal unique patients: {full_df['DEMOG_REC_ID'].nunique()}")

# Step 2: Filter to DEPT_DESC + MODULE of interest
admission_candidates = full_df[
    (full_df["DEPT_DESC"].isin(admission_depts)) &
    (full_df["MODULE"] == target_module)
]

# Identify patients without admission records
admission_patient_ids = set(
    full_df[
        (full_df["DEPT_DESC"].isin(admission_depts)) &
        (full_df["MODULE"] == target_module)
    ]["DEMOG_REC_ID"].unique()
)
all_patient_ids = set(full_df["DEMOG_REC_ID"].unique())
patients_without_admission = all_patient_ids - admission_patient_ids

print("\nPatients without admission records:")
for pid in sorted(patients_without_admission):
    print(f"  - {pid}")

# For patients without ADMDIS records, find their DEPT_DESC-matching records anyway
print("\nTrying to extract fallback admission candidates for patients without ADMDIS:")
fallback_records = full_df[
    (full_df["DEMOG_REC_ID"].isin(patients_without_admission)) &
    (full_df["DEPT_DESC"].isin(admission_depts))
]

if fallback_records.empty:
    print("No fallback records found.")
    # Investigate what DEPT_DESC values these patients *do* have
    print("\nListing DEPT_DESC values for patients without ADMDIS admissions:")
    patient_dept_map = (
        full_df[full_df["DEMOG_REC_ID"].isin(patients_without_admission)]
        .groupby("DEMOG_REC_ID")["DEPT_DESC"]
        .unique()
    )

    for patient_id, dept_list in patient_dept_map.items():
        dept_values = [str(d) for d in dept_list if pd.notna(d)]
        print(f"  Patient ID {patient_id}: {', '.join(dept_values) if dept_values else 'No DEPT_DESC found'}")
else:
    for patient_id, group in fallback_records.groupby("DEMOG_REC_ID"):
        print("\n" + "="*80)
        print(f"Patient ID (fallback): {patient_id}")
        print("="*80)

        group = group.sort_values("TEXT_UPD_TIME", na_position="last").reset_index(drop=True)
        for i, row in group.iterrows():
            print(f"\nRecord {i+1}:")
            print(f"  FILENAME     : {row['FILENAME']}")
            print(f"  ADMISSION_NO : {row['ADMISSION_NO']}")
            print(f"  MODULE       : {row['MODULE']}")
            print(f"  SUB_MODULE   : {row['SUB_MODULE']}")

# Step 3: Find minimal ADMISSION_NO per patient among those candidates
min_adm_map = (
    admission_candidates
    .groupby("DEMOG_REC_ID")["ADMISSION_NO"]
    .min()
    .reset_index()
)

# Step 4: Get full matching rows (by DEMOG_REC_ID, ADMISSION_NO, MODULE, DEPT_DESC)
admission_records = full_df.merge(min_adm_map, on=["DEMOG_REC_ID", "ADMISSION_NO"])
admission_records = admission_records[
    (admission_records["DEPT_DESC"].isin(admission_depts)) &
    (admission_records["MODULE"] == target_module)
]

# Step 5: Print admission records and field differences
for patient_id, group in admission_records.groupby("DEMOG_REC_ID"):
    print("\n" + "="*80)
    print(f"Patient ID: {patient_id}")
    print("="*80)

    group = group.sort_values("TEXT_UPD_TIME", na_position="last").reset_index(drop=True)

    for i, row in group.iterrows():
        print(f"\nRecord {i+1}:")
        print(f"  FILENAME     : {row['FILENAME']}")
        print(f"  ADMISSION_NO : {row['ADMISSION_NO']}")
        print(f"  MODULE       : {row['MODULE']}")
        print(f"  SUB_MODULE   : {row['SUB_MODULE']}")

    print("\nDifferences between records:")
    prev_row = None
    for i, row in group.iterrows():
        if prev_row is not None:
            for col in fields_to_compare:
                val1 = prev_row[col]
                val2 = row[col]
                if pd.isna(val1) and pd.isna(val2):
                    continue
                if val1 != val2:
                    print(f"  {col} changed: {val1} -> {val2}")
        prev_row = row


print("\nChecking for existence of .txt files for patients with admission records:")
checked_files = set()
# Counters for found and missing .txt files
found_count = 0
missing_count = 0
for patient_id, group in admission_records.groupby("DEMOG_REC_ID"):
    print(f"\nPatient ID: {patient_id}")
    for filename in group["FILENAME"].unique():
        txt_filename = filename + ".txt"
        found = False
        for text_dir in text_dirs:
            if os.path.isfile(os.path.join(text_dir, txt_filename)):
                print(f"  Found: {txt_filename} in {text_dir}")
                found = True
                break
        if not found:
            print(f"  Missing: {txt_filename} in both directories")
            missing_count += 1
        else:
            found_count += 1

# Summary
print("\nSummary of .txt file availability:")
print(f"  Found    : {found_count}")
print(f"  Missing  : {missing_count}")
print(f"  Total    : {found_count + missing_count}")

# Save admission records to CSV
output_csv_path = "../admission_records.csv"
admission_records.to_csv(output_csv_path, index=False)
print(f"\nSaved admission records to {output_csv_path}")