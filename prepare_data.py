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


# Generate name pairs and corresponding labels
def generate_pairs(data, label):
    pairs = []
    for row in data:
        tools = [
            key.split("_")[0] for key in row.keys() if "first_name" in key and row[key]
        ]
        for i in range(len(tools) - 1):
            for j in range(i + 1, len(tools)):
                set_1 = (row[tools[i] + "_first_name"], row[tools[i] + "_last_name"])
                set_2 = (row[tools[j] + "_first_name"], row[tools[j] + "_last_name"])
                pairs.append((set_1, set_2, label))
    return pairs


# Write data to CSV file
def write_data(filename, pairs):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["first_name_1", "last_name_1", "first_name_2", "last_name_2", "label"]
        )
        for pair in pairs:
            writer.writerow([pair[0][0], pair[0][1], pair[1][0], pair[1][1], pair[2]])


def main():
    matches = read_data("data/matches.csv")
    distinct = read_data("data/distinct.csv")

    print(
        f"Found {len(matches)} match rows. (Note each row might contain multiple pairs.)"
    )
    print(
        f"Found {len(distinct)} distinct rows. (Note each row might contain multiple pairs.)"
    )

    positive_pairs = generate_pairs(matches, 1)
    negative_pairs = generate_pairs(distinct, -1)

    print(f"Found {len(positive_pairs)} matched (positive, 1 annotated) pairs.")
    print(f"found {len(negative_pairs)} distinct (negative, -1 annotated) pairs.")

    # Balance the dataset if necessary
    if len(positive_pairs) > len(negative_pairs):
        positive_pairs = random.sample(positive_pairs, len(negative_pairs))
    elif len(negative_pairs) > len(positive_pairs):
        negative_pairs = random.sample(negative_pairs, len(positive_pairs))

    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)

    # Split the data into training and evaluation sets
    split_index = int(len(all_pairs) * 0.8)
    training_pairs = all_pairs[:split_index]
    eval_pairs = all_pairs[split_index:]

    # Write the data to CSV files
    write_data("data/training.csv", training_pairs)
    write_data("data/eval.csv", eval_pairs)


if __name__ == "__main__":
    main()
