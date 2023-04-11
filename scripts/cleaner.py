import os
import glob
import shutil

def find_overlaps():
    folders = [d for d in os.listdir() if os.path.isdir(d)]
    id_files = {}

    for folder in folders:
        osu_files = glob.glob(os.path.join(folder, '*.osu'))

        for osu_file in osu_files:
            filename = os.path.basename(osu_file)
            if filename in id_files:
                id_files[filename].append(osu_file)
            else:
                id_files[filename] = [osu_file]

    return {k: v for k, v in id_files.items() if len(v) > 1}


def move_overlapping_files(overlaps):
    if not os.path.exists("overlaps"):
        os.makedirs("overlaps")

    for files in overlaps.values():
        for file in files:
            destination = os.path.join("overlaps", os.path.basename(file))
            shutil.move(file, destination)
            print(f"Moved {file} to {destination}")


if __name__ == "__main__":
    overlaps = find_overlaps()
    move_overlapping_files(overlaps)
