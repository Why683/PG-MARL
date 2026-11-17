import argparse
from pathlib import Path

from pgmarl.protocol import build_manifest, write_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = build_manifest(args.data_root, args.output)
    audit = write_audit(frame, args.output.with_suffix(".audit.json"))
    print(audit)


if __name__ == "__main__":
    main()

