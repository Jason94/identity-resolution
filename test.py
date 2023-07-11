import pandas as pd
import numpy as np


def update_label(df1: pd.DataFrame, df2: pd.DataFrame, fields: List[str]):
    """
    Updates the 'label' value in df1 with the 'label' value in df2
    for rows where all columns in 'fields' list match in both dataframes.

    Parameters:
    df1 (pandas.DataFrame): The dataframe to be updated.
    df2 (pandas.DataFrame): The dataframe that contains the new label values.
    fields (list of str): The fields to be matched in both dataframes.

    Returns:
    pandas.DataFrame: The updated dataframe.
    """
    merged = pd.merge(df1, df2, on=fields, how="left", suffixes=("", "_y"))
    df1["label"] = np.where(
        pd.notna(merged["label_y"]), merged["label_y"], df1["label"]
    )
    return df1


def main():
    # Define the initial data for the dataframes
    data1 = {
        "label": ["a", "b", "c", "d"],
        "field1": [1, 2, 3, 4],
        "field2": [5, 6, 7, 8],
    }
    data2 = {"label": ["e", "f"], "field1": [1, 2], "field2": [5, 6]}
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    fields = ["field1", "field2"]

    print("Initial DataFrame df1:")
    print(df1)
    print("\nDataFrame df2:")
    print(df2)

    updated_df = update_label(df1, df2, fields)

    print("\nUpdated DataFrame:")
    print(updated_df)


if __name__ == "__main__":
    main()
