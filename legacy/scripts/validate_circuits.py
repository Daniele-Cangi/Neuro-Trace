import sqlite3

conn = sqlite3.connect('circuits/atlas_circuits.db')
cursor = conn.cursor()
cursor.execute('SELECT circuit_id, task_tag, vlo_mean, num_components FROM circuits')
circuits = cursor.fetchall()

print('Circuit Registry Validation:')
print(f'  Total circuits: {len(circuits)}')
print()

for c in circuits:
    print(f'  Circuit: {c[0]}')
    print(f'    Task: {c[1]}')
    print(f'    VLO: {c[2]:.3f}')
    print(f'    Components: {c[3]}')
    print()

conn.close()
