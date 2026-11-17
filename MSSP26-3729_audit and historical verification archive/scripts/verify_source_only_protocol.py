import argparse
import json
from pathlib import Path

import pandas as pd

from pgmarl.protocol import verify_source_only_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    audit = verify_source_only_manifest(pd.read_csv(args.manifest))
    print(json.dumps(audit, indent=2))
    print("PASS: D/E are target-test only and source trials/windows are disjoint.")


if __name__ == "__main__":
    main()

