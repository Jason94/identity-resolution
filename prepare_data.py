import csv
import random


def is_valid(string: str) -> bool:
    # Remove non-ascii entries
    for c in string:
        if not (0 <= ord(c) <= 127):
            return False

    return True


def valid_row(row: dict) -> bool:
    for s in row.values():
        if not is_valid(s):
            return False

    return True


# Read data from CSV files
def read_data(filename):
    with open(filename, "r", encoding="utf8") as file:
        reader = csv.DictReader(file)
        data = [row for row in reader if valid_row(row)]
    return data


# Write data to CSV file
def write_data(filename, pairs):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "first_name_1",
                "last_name_1",
                "email1",
                "first_name_2",
                "last_name_2",
                "email2",
                "label",
            ]
        )
        for pair in pairs:
            writer.writerow(
                [
                    pair["first_name1"],
                    pair["last_name1"],
                    pair["email1"],
                    pair["first_name2"],
                    pair["last_name2"],
                    pair["email2"],
                    pair["label"],
                ]
            )


def main():
    duplicates = read_data("data/duplicates.csv")
    distincts = read_data("data/distincts.csv")

    print(f"Found {len(duplicates)} duplicate rows.")
    print(f"Found {len(distincts)} distinct rows.")

    # Balance the dataset if necessary
    if len(duplicates) > len(distincts):
        duplicates = random.sample(duplicates, len(distincts))
    elif len(distincts) > len(duplicates):
        distincts = random.sample(distincts, len(duplicates))

    all_pairs = duplicates + distincts
    random.shuffle(all_pairs)

    # Split the data into training and evaluation sets
    split_index = int(len(all_pairs) * 0.85)
    training_pairs = all_pairs[:split_index]
    eval_pairs = all_pairs[split_index:]

    # Write the data to CSV files
    write_data("data/training.csv", training_pairs)
    write_data("data/eval.csv", eval_pairs)


if __name__ == "__main__":
    main()
