import sqlite3

db_path = './beatmaps.db'

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute('''
    SELECT beatmap_id, COUNT(*) AS vector_count
    FROM beatmap_vectors
    GROUP BY beatmap_id
    ORDER BY vector_count DESC
    LIMIT 100;
''')

rows = cursor.fetchall()
if rows:
    for row in rows:
        beatmap_id = row[0]
        vector_count = row[1]
        print("Beatmap with ID {} has {} vectors.".format(beatmap_id, vector_count))
else:
    print("No beatmaps found.")