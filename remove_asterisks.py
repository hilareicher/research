import os
import re
import argparse


def remove_asterisks_from_txt_files(directory):
    """
    This function goes through all the .txt files in a given directory
    and removes all instances of double asterisks (**) from the text.

    Parameters:
        directory (str): The path to the directory containing the .txt files.
    """
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Only process .txt files
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            # Read the file's contents
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # Remove all instances of double asterisks (**)
            updated_content = re.sub(r'\*\*', '', content)

            # Write the updated content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(updated_content)

            print(f"Processed: {filename}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Remove double asterisks (**) from all .txt files in a directory.")
    parser.add_argument("directory", type=str, help="The path to the directory containing the .txt files.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided directory
    remove_asterisks_from_txt_files(args.directory)
