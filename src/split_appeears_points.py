import pandas as pd
from pathlib import Path

INPUT = Path("data/processed/appeears_points_14270.csv")
OUTDIR = Path("data/raw/appeears/requests")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

chunk_size = 1000
total = len(df)

for i, start in enumerate(range(0, total, chunk_size), 1):
    chunk = df.iloc[start:start + chunk_size].copy()
    output = OUTDIR / f"slidronix_2022_points_{i:02d}.csv"
    chunk.to_csv(output, index=False)

    print(
        f"Request {i:02d}: "
        f"{len(chunk):,} points -> {output}"
    )

print()
print(f"Total points : {total:,}")
print(f"Total files  : {(total + chunk_size - 1) // chunk_size}")
