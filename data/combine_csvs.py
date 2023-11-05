import pandas as pd
import argparse
import sys


def read_and_validate_files(filenames):
    """Read files and ensure they all have the same columns"""
    dfs = []
    column_set = None
    for file in filenames:
        df = pd.read_csv(file)
        columns = set(df.columns)
        if column_set is None:
            column_set = columns
        elif column_set != columns:
            raise ValueError(f"File {file} does not have matching columns.")
        dfs.append(df)
    return dfs


def combine_csv(output_file, input_files):
    """Combine CSV files with the same columns"""
    dfs = read_and_validate_files(input_files)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined file saved as {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple CSV files with the same columns into one CSV file."
    )
    parser.add_argument("output_file", type=str, help="Output CSV filename")
    parser.add_argument("input_files", type=str, nargs="+", help="Input CSV filenames")

    args = parser.parse_args()

    try:
        combine_csv(args.output_file, args.input_files)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
