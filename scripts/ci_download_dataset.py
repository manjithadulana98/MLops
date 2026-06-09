"""Download MIT-BIH Arrhythmia dataset from PhysioNet with per-record retry logic."""
import os
import sys
import time

import wfdb

os.makedirs("data/MIT_dataset", exist_ok=True)

records = None
for idx_attempt in range(1, 6):
    try:
        records = wfdb.get_record_list("mitdb")
        break
    except Exception as exc:
        print(f"Record list attempt {idx_attempt}/5 failed: {exc}")
        if idx_attempt == 5:
            sys.exit(1)
        time.sleep(15 * idx_attempt)

for rec in records:
    for rec_attempt in range(1, 4):
        try:
            wfdb.dl_database("mitdb", dl_dir="data/MIT_dataset", records=[rec])
            print(f"Downloaded: {rec}")
            break
        except Exception as exc:
            print(f"Record {rec} attempt {rec_attempt}/3 failed: {exc}")
            if rec_attempt == 3:
                sys.exit(1)
            time.sleep(10 * rec_attempt)

print(f"Total records downloaded: {len(records)}")
