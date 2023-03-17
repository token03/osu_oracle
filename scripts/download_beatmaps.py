import os
import sys

import requests


def download_beatmaps(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(input_file, 'r') as infile:
        for line in infile:
            beatmap_id = line.strip()
            url = f"https://kitsu.moe/api/osu/{beatmap_id}"
            response = requests.get(url)

            if response.status_code == 200:
                file_path = os.path.join(output_folder, f"{beatmap_id}.osu")
                with open(file_path, 'wb') as outfile:
                    outfile.write(response.content)
                print(f"Downloaded {beatmap_id}.osu")
            else:
                print(f"Error downloading {beatmap_id}.osu: Status code {response.status_code}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_beatmaps.py input.txt output_folder")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    download_beatmaps(input_file, output_folder)