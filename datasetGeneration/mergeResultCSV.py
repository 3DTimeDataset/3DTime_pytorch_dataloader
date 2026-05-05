import sys
import os
import argparse
import pandas as pd
sys.path.insert(0,"..")
import Logger
log = Logger.Logger(verbose_level=1)

def arguments(parser: argparse.ArgumentParser) -> None:
    """
    Argument definition.
    """
    parser.add_argument('--slice-csv', required=True, type=str, help=f"Path to the resulting CSV of the slicing process.")
    parser.add_argument('--other-csv', required=True, type=str, help=f"Path to the other resulting CSV file, either from the TimeKlip annotation or the vector extraction process.")

def parse(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parses the given arguments and check their integrity
    """
    args = parser.parse_args()
    if not os.path.isfile(args.slice_csv):
        log.error(f"The provided result CSV from slicing process does not exist.", 1)
    if not os.path.isfile(args.other_csv):
        log.error(f"The provided second result CSV does not exist.", 1)
    return args

def main() -> None:
    parser = argparse.ArgumentParser()
    arguments(parser)
    args = parse(parser)

    slice_csv = pd.read_csv(args.slice_csv)
    other_csv = pd.read_csv(args.other_csv)

    common_cols = list(set(slice_csv.columns) & set(other_csv.columns))

    if len(common_cols) == 0:
        log.error(f"Could not find common columns between {args.slice_csv} and {args.other_csv}.", 1)

    result = pd.merge(slice_csv, other_csv, on=common_cols, how='inner')

    result.to_csv(args.slice_csv, index=False)

if __name__ == "__main__":
    main()

"""
End of file.
"""