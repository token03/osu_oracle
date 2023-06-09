import os
import sqlite3

def flip_beatmap(beatmap_data, horizontal=True, vertical=True):
    max_x, max_y = 512, 384
    flipped_data = beatmap_data.copy()
    for obj in flipped_data['hit_objects']:
        if horizontal:
            obj['x'] = max_x - obj['x']
        if vertical:
            obj['y'] = max_y - obj['y']
    return flipped_data

def parse_osu_file(file_path, max_slider_length = 1, max_time_diff = 1, print_info=False):
    data = {
        'beatmap_id': None,
        'hp_drain': None,
        'circle_size': None,
        'od': None,
        'ar': None,
        'slider_multiplier': None,
        'slider_tick': None,
        'hit_objects': [],
        'label': None,
    }

    parent_folder = os.path.dirname(file_path)
    data['label'] = os.path.basename(parent_folder)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        section = None

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith('[') and line.endswith(']'):
                section = line[1:-1]
                continue

            if section == 'Metadata':
                key, value = line.split(':', maxsplit=1)
                if key == 'Title' and print_info:
                    print("Title: " + value, end = ' ')
                if key == 'Artist' and print_info:
                    print("by " + value)
                if key == 'Creator' and print_info:
                    print("Mapper: " + value)
                if key == 'Version' and print_info:
                    print("Diffuculty: " + value)
                if key == 'BeatmapID':
                    data['beatmap_id'] = int(value)
                    if data['beatmap_id'] is None:
                        return None
            elif section == 'Difficulty':
                key, value = line.split(':', maxsplit=1)
                value = float(value)
                if key == 'HPDrainRate':
                    data['hp_drain'] = value / 10
                elif key == 'CircleSize':
                    data['circle_size'] = value / 10
                elif key == 'OverallDifficulty':
                    data['od'] = value / 10
                elif key == 'ApproachRate':
                    data['ar'] = value / 10
                elif key == 'SliderMultiplier':
                    data['slider_multiplier'] = value
                elif key == 'SliderTickRate':
                    data['slider_tick'] = value
            elif section == 'HitObjects':  # Move this line one level back
                    obj_data = line.split(',')
                    hit_object_type = int(obj_data[3])

                    hit_circle_flag = 0b1
                    slider_flag = 0b10

                    if hit_object_type & hit_circle_flag:
                        hit_object = {
                            'x': int(obj_data[0]),
                            'y': int(obj_data[1]),
                            'time': min(1000, int(obj_data[2])),
                            'length': float(0), # slider len
                        }
                        data['hit_objects'].append(hit_object)
                    elif hit_object_type & slider_flag:
                        hit_object = {
                            'x': int(obj_data[0]),
                            'y': int(obj_data[1]),
                            'time': min(1000, int(obj_data[2])),
                            'length': min(500, float(obj_data[7])), # slider len 
                        }
                        data['hit_objects'].append(hit_object)
                        
    # Normalize the coordinates
    max_x, max_y = 512, 384
    for obj in data['hit_objects']:
        obj['x_norm'] = obj['x'] / max_x
        obj['y_norm'] = obj['y'] / max_y

    # Compute the time differences
    if data['hit_objects']:  # Add this condition
        for i, obj in enumerate(data['hit_objects'][1:], start=1):
            obj['time_diff'] = obj['time'] - data['hit_objects'][i - 1]['time']
        data['hit_objects'][0]['time_diff'] = 0

    vectors = []
    for i, obj in enumerate(data['hit_objects'][1:], start=1):
        prev_obj = data['hit_objects'][i - 1]
        x_diff = obj['x_norm'] - prev_obj['x_norm']
        y_diff = obj['y_norm'] - prev_obj['y_norm']
        time_diff = obj['time_diff']
        length = obj['length']
        vectors.append((x_diff, y_diff, time_diff / max_time_diff, length / max_slider_length))

    data['vectors'] = vectors
    return data


def create_tables(conn):
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS beatmaps (
            id INTEGER PRIMARY KEY,
            beatmap_id INTEGER,
            category TEXT,
            hp_drain REAL,
            circle_size REAL,
            od REAL,
            ar REAL,
            slider_multiplier REAL,
            slider_tick REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS beatmap_vectors (
            id INTEGER PRIMARY KEY,
            beatmap_id INTEGER,
            x_diff REAL,
            y_diff REAL,
            time_diff REAL,
            length REAL,
            FOREIGN KEY (beatmap_id) REFERENCES beatmaps (id)
        )
    ''')


    conn.commit()


def insert_beatmap_data(conn, beatmap_data):
    cursor = conn.cursor()

    cursor.execute('INSERT INTO beatmaps (beatmap_id, category, hp_drain, circle_size, od, ar, slider_multiplier, slider_tick) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                   (beatmap_data['beatmap_id'], beatmap_data['label'], beatmap_data['hp_drain'], beatmap_data['circle_size'], beatmap_data['od'], beatmap_data['ar'],
                    beatmap_data['slider_multiplier'], beatmap_data['slider_tick']))
    beatmap_row_id = cursor.lastrowid

    for vector in beatmap_data['vectors']:
        cursor.execute('INSERT INTO beatmap_vectors (beatmap_id, x_diff, y_diff, time_diff, length) VALUES (?, ?, ?, ?, ?)', (beatmap_row_id, vector[0], vector[1], vector[2], vector[3]))
    conn.commit()
    
def main():
    root_dir = '.'  # Current directory
    beatmaps_data = []

    # Connect to the SQLite database
    conn = sqlite3.connect('./beatmaps.db')

    # Create the necessary tables
    create_tables(conn)

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.osu'):
                file_path = os.path.join(dirpath, filename)
                beatmap_data = parse_osu_file(file_path)
                if beatmap_data is not None and (len(beatmap_data['vectors']) <= 3502):
                    # Process the data (e.g., insert into the database)
                    beatmaps_data.append(beatmap_data)                # Insert the parsed beatmap data into the SQLite database
                    insert_beatmap_data(conn, beatmap_data)
                    
                    # Flip horizontally
                    flipped_beatmap_data = flip_beatmap_horizontal(beatmap_data)
                    beatmaps_data.append(flipped_beatmap_data)
                    insert_beatmap_data(conn, flipped_beatmap_data)
                    
                    # Flip vertically
                    flipped_vertical_data = flip_beatmap_vertical(beatmap_data)
                    beatmaps_data.append(flipped_vertical_data)
                    insert_beatmap_data(conn, flipped_vertical_data)

                    # Flip both horizontally and vertically
                    flipped_both_data = flip_beatmap_horizontal(flipped_vertical_data)
                    beatmaps_data.append(flipped_both_data)
                    insert_beatmap_data(conn, flipped_both_data)

                    pass

    # Close the SQLite database connection
    conn.close()

if __name__ == '__main__':
    main()