import itertools, csv, numpy as np

matrix_size = int(input("Enter the size of the matrix: "))
with open(f"{matrix_size}x{matrix_size}_1s0s.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"x{i}" for i in range(matrix_size**2)] + ["y"])
    for b in itertools.product([0,1], repeat=matrix_size**2):
        m = np.array(b).reshape(matrix_size, matrix_size)
        y = 0
        if (any(r.sum()==matrix_size for r in m) or
            any(c.sum()==matrix_size for c in m.T) or
            np.diag(m).sum()==matrix_size or np.diag(np.fliplr(m)).sum()==matrix_size):
            y = 1
        w.writerow(list(b) + [y])

print(f"✅ CSV generated: {matrix_size}x{matrix_size}_1s0s.csv")
