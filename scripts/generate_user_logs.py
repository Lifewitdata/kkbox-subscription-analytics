"""Generate deterministic synthetic subscriber-day listening logs."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261103

def main() -> None:
    members = pd.read_csv(ROOT / "members_v3.csv", usecols=["msno"])
    rng = np.random.default_rng(SEED)
    rows = []
    for msno in members["msno"]:
        for day in (1, 11, 21):
            plays = rng.integers([5, 0, 0, 0, 8], [40, 16, 13, 9, 74])
            n25, n50, n75, n985, n100 = map(int, plays)
            total = n25 + n50 + n75 + n985 + n100
            secs = int((n25*.25+n50*.5+n75*.75+n985*.985+n100)*240+rng.integers(0,5001))
            rows.append((msno, f"201703{day:02d}", n25, n50, n75, n985, n100,
                         max(1, int(total*.8)), secs))
    frame = pd.DataFrame(rows, columns=[
        "msno", "date", "num_25", "num_50", "num_75", "num_985",
        "num_100", "num_unq", "total_secs"
    ])
    frame.to_csv(ROOT / "user_logs_v2.csv", index=False)
    print(f"Wrote {len(frame):,} rows to user_logs_v2.csv")

if __name__ == "__main__":
    main()
