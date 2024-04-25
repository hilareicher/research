import os
import sys

import chardet


def convert_encoding(input_file_path, output_file_path, input_encoding, output_encoding):
    with open(input_file_path, 'r', encoding=input_encoding) as f:
        text = f.read()
    with open(output_file_path, 'w', encoding=output_encoding) as f:
        f.write(text)

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as file:
        # Read some bytes from the file. The more you read, the more accurate the detection will be.
        # However, reading the entire file might be slow for very large files.
        raw_data = file.read(50000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        return encoding



def convert_all_files_in_directory(input_dir, input_encoding, output_encoding):
    output_dir = f"{input_dir}_utf8"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        if os.path.isfile(input_file_path):
            try:
                convert_encoding(input_file_path, output_file_path, input_encoding, output_encoding)
                print(f"Converted '{filename}' to UTF-8 and saved to '{output_dir}'.")
            except Exception as e:
                print(f"Failed to convert '{filename}': {e}")


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <source_directory>")
    # else:
    #     source_directory = sys.argv[1]
    #     convert_all_files_in_directory(source_directory, 'windows-1255', 'utf-8')
    print(detect_file_encoding("nisaion_Jan2018.txt"))
