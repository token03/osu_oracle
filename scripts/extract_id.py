import re
import sys


def extract_numbers(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            numbers = re.findall(r'\((\d+)\)', line)
            for number in numbers:
                outfile.write(f"{number}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_id.py input.txt output.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    extract_numbers(input_file, output_file)