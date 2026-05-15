import random
import csv

# --------------------------
# CONFIG (EASY TO EDIT)
# --------------------------
MAX = 100          # Time goes from 0 → 100 seconds
OUTPUT_FILE = "vid.csv"

# --------------------------
# Generate data
# --------------------------
with open(OUTPUT_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["time", "val", "increaser", "decreaser"])

    for t in range(0, MAX + 1):
        val = t - (MAX // 2)          # from -50 to 100 (when MAX=100)
        rand = random.randint(1, 10)  # small random step
        increaser = val + rand
        decreaser = val - rand
        
        writer.writerow([t, val, increaser, decreaser])

print(f"✅ CSV created: {OUTPUT_FILE}")