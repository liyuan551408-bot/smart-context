"""Build a complete repository index (normally use update_index.py thereafter)."""
import argparse
from update_index import update

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", help="Repository path (defaults to current directory)")
    args = parser.parse_args()
    update(root=args.root, force=True)
