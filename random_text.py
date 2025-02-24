import os
import random
import argparse
import shutil


def get_texts_with_word_count(directory, word_count):
    """
    Extract text files from the given directory with the specified word count.
    """
    valid_texts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                if len(content.split()) >= word_count:
                    valid_texts.append((filename, filepath))
    return valid_texts


def select_random_texts(texts, num_to_select):
    """
    Select a random sample of texts.
    """
    return random.sample(texts, min(len(texts), num_to_select))


def save_selected_texts(selected_texts, output_directory):
    """
    Save the selected text files to the output directory.
    """
    os.makedirs(output_directory, exist_ok=True)
    for filename, filepath in selected_texts:
        shutil.copy(filepath, os.path.join(output_directory, filename))


def main():
    parser = argparse.ArgumentParser(
        description="Select random text files with a specific word count and save them to a new directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing text files.")
    parser.add_argument("word_count", type=int, help="The exact word count to filter text files.")
    parser.add_argument("num_to_select", type=int, help="Number of random texts to select.")
    parser.add_argument("output_directory", type=str, help="Path to save the selected files.")

    args = parser.parse_args()
    directory = args.directory
    word_count = args.word_count
    num_to_select = args.num_to_select
    output_directory = args.output_directory

    # Get texts with the specified word count
    valid_texts = get_texts_with_word_count(directory, word_count)

    if not valid_texts:
        print(f"No texts with exactly {word_count} words were found in the directory.")
        return

    # Select random texts
    selected_texts = select_random_texts(valid_texts, num_to_select)

    # Save the selected texts
    save_selected_texts(selected_texts, output_directory)

    print(f"Saved {len(selected_texts)} texts to '{output_directory}'.")


if __name__ == "__main__":
    main()
