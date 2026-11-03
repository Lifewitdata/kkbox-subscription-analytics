"""Regenerate the committed synthetic listening logs exactly (fixed LCG seed)."""
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261103

def main() -> None:
    state = SEED
    def random_value() -> float:
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        return state / 4294967296

    members = pd.read_csv(ROOT / "members_v3.csv", usecols=["msno"])
    rows = []
    for msno in members["msno"]:
        for day in (1, 11, 21):
            n25 = math.floor(5 + random_value() * 35)
            n50 = math.floor(random_value() * 15)
            n75 = math.floor(random_value() * 12)
            n985 = math.floor(random_value() * 8)
            n100 = math.floor(8 + random_value() * 65)
            total = n25 + n50 + n75 + n985 + n100
            seconds = math.floor((n25*.25+n50*.5+n75*.75+n985*.985+n100)*240
                                 + random_value()*5000 + 0.5)
            rows.append((msno, f"201703{day:02d}", n25, n50, n75, n985,
                         n100, max(1, math.floor(total*.8)), seconds))
    frame = pd.DataFrame(rows, columns=[
        "msno", "date", "num_25", "num_50", "num_75", "num_985",
        "num_100", "num_unq", "total_secs"
    ])
    frame.to_csv(ROOT / "user_logs_v2.csv", index=False)
    print(f"Wrote {len(frame):,} rows to user_logs_v2.csv")

if __name__ == "__main__":
    main()
