import sqlite3

# Connect to the database
conn = sqlite3.connect('beatmaps.db')

# Execute the query
cursor = conn.cursor()
cursor.execute('''
    SELECT beatmap_id, COUNT(*) AS vector_count
    FROM beatmap_vectors
    GROUP BY beatmap_id
    ORDER BY vector_count DESC
    LIMIT 1;
''')
results = cursor.fetchall()

# Print the results
print('Top 50 beatmap_ids with the most number of vectors:')
for row in results:
    print(f'beatmap_id: {row[0]}, vector_count: {row[1]}')
