import re
import sys
import os
import requests
import glob



def extract_numbers(input_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    # If the input file already contains extracted IDs, return without overwriting the file
    if re.fullmatch(r'(\d+\n)+', content):
        return

    numbers = re.findall(r'\((\d+)\)', content)

    # If there are no numbers to extract, return without overwriting the file
    if not numbers:
        return

    with open(input_file, 'w') as outfile:
        for number in numbers:
            outfile.write(f"{number}\n")


def download_beatmaps(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, 'r') as infile:
        for line in infile:
            beatmap_id = line.strip()
            url = f"https://osu.direct/api/osu/{beatmap_id}"
            response = requests.get(url)

            if response.status_code == 200:
                file_path = os.path.join(output_folder, f"{beatmap_id}.osu")
                with open(file_path, 'wb') as outfile:
                    outfile.write(response.content)
                print(f"Downloaded {beatmap_id}.osu")
            else:
                print(f"Error downloading {beatmap_id}.osu: Status code {response.status_code}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        txt_files = glob.glob("*.txt")
        for txt_file in txt_files:
            output_folder = os.path.splitext(txt_file)[0]
            extract_numbers(txt_file)
            download_beatmaps(txt_file, output_folder)
    elif len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_folder = os.path.splitext(input_file)[0]
        extract_numbers(input_file)
        download_beatmaps(input_file, output_folder)
    else:
        print("Usage: python combined_script.py [input.txt]")
        sys.exit(1)