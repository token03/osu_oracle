import sqlite3

# Connect to the database
conn = sqlite3.connect('beatmaps.db')

# Update the time_diff values greater than 1000 to 1000
cursor = conn.cursor()
cursor.execute('UPDATE beatmap_vectors SET time_diff = 1000 WHERE time_diff > 1000')
conn.commit()

# Confirm the update
print(f'{cursor.rowcount} rows updated')
