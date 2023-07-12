import random
from typing import List
import os
import csv
import argparse

from parsons import Table
from parsons.databases.redshift import Redshift


def simulate_typos(s: str, num_deletions: int = 0, num_swaps: int = 1) -> str:
    # Handle deletions
    for _ in range(num_deletions):
        if len(s) > 0:
            del_index = random.randint(0, len(s) - 1)
            s = s[:del_index] + s[del_index + 1 :]

    # Handle swaps
    for _ in range(num_swaps):
        if len(s) > 1:
            i, j = random.sample(range(len(s)), 2)
            s_list = list(s)
            s_list[i], s_list[j] = s_list[j], s_list[i]
            s = "".join(s_list)

    return s


def generate_synthetic_data(data):
    synthetic_data = []

    # Note: Rules 1 & 2 aren't really feasible with the existing architecture. It's just not possible
    # for the model to produce radically different embeddings for slight character variations if no
    # other data is present. It will probably require some kind of parallel output head that produces
    # a "matchable" flag.

    # # Rule 1: If all data is present but the first and last names are reversed, it is a duplicate.
    # record1 = {
    #     "first_name1": data["last_name"],
    #     "last_name1": data["first_name"],
    #     "email1": data["email"],
    #     "phone1": data["phone"],
    #     "state1": data["state"],
    #     "first_name2": data["first_name"],
    #     "last_name2": data["last_name"],
    #     "email2": data["email"],
    #     "phone2": data["phone"],
    #     "state2": data["state"],
    #     "label": 1,
    # }
    # synthetic_data.append(record1)

    # # Rule 2: If the first initial is present and last name is present, but all other data is absence, it is distinct.
    # record2 = {
    #     "first_name1": data["first_name"][0],
    #     "last_name1": data["last_name"],
    #     "email1": "",
    #     "phone1": "",
    #     "state1": "",
    #     "first_name2": data["first_name"],
    #     "last_name2": data["last_name"],
    #     "email2": data["email"],
    #     "phone2": data["phone"],
    #     "state2": data["state"],
    #     "label": -1,
    # }
    # synthetic_data.append(record2)

    # Rule 3: If the first initial is present and all other data is present, it is a duplicate.
    record3 = {
        "first_name1": data["first_name"][0],
        "last_name1": data["last_name"],
        "email1": data["email"],
        "phone1": data["phone"],
        "state1": data["state"],
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": data["email"],
        "phone2": data["phone"],
        "state2": data["state"],
        "label": 1,
    }
    synthetic_data.append(record3)

    # Rule 4 (part 1): If the first and last names match, and email matches,
    #                  but phone and state are absent, it is a duplicate.
    record4 = {
        "first_name1": data["first_name"],
        "last_name1": data["last_name"],
        "email1": data["email"],
        "phone1": "",
        "state1": "",
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": data["email"],
        "phone2": data["phone"],
        "state2": data["state"],
        "label": 1,
    }
    synthetic_data.append(record4)

    # Rule 4 (part 2): If the first and last names match, and phone matches,
    #                  but email and state are absent, it is a duplicate.
    record5 = {
        "first_name1": data["first_name"],
        "last_name1": data["last_name"],
        "email1": "",
        "phone1": data["phone"],
        "state1": "",
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": data["email"],
        "phone2": data["phone"],
        "state2": data["state"],
        "label": 1,
    }
    synthetic_data.append(record5)

    # Rule 5: If only names are present, then a typo in first name is distinct.
    record6 = {
        "first_name1": simulate_typos(data["first_name"]),
        "last_name1": data["last_name"],
        "email1": "",
        "phone1": "",
        "state1": "",
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": "",
        "phone2": "",
        "state2": "",
        "label": -1,
    }
    synthetic_data.append(record6)

    # Rule 6: If only names are present, then a typo in last name is distinct.
    record7 = {
        "first_name1": data["first_name"],
        "last_name1": simulate_typos(data["last_name"]),
        "email1": "",
        "phone1": "",
        "state1": "",
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": "",
        "phone2": "",
        "state2": "",
        "label": -1,
    }
    synthetic_data.append(record7)

    # Rule 7: If email and state match then a typo in first name is a duplicate.
    record8 = {
        "first_name1": simulate_typos(data["first_name"]),
        "last_name1": data["last_name"],
        "email1": data["email"],
        "phone1": "",
        "state1": data["state"],
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": data["email"],
        "phone2": "",
        "state2": data["state"],
        "label": 1,
    }
    synthetic_data.append(record8)

    # Rule 8: If email and state match then a typo in last name is a duplicate.
    record9 = {
        "first_name1": data["first_name"],
        "last_name1": simulate_typos(data["last_name"]),
        "email1": data["email"],
        "phone1": "",
        "state1": data["state"],
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": data["email"],
        "phone2": "",
        "state2": data["state"],
        "label": 1,
    }
    synthetic_data.append(record9)

    # Rule 9: If phone and state match then a typo in first name is a duplicate.
    record10 = {
        "first_name1": simulate_typos(data["first_name"]),
        "last_name1": data["last_name"],
        "email1": "",
        "phone1": data["phone"],
        "state1": data["state"],
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": "",
        "phone2": data["phone"],
        "state2": data["state"],
        "label": 1,
    }
    synthetic_data.append(record10)

    # Rule 10: If phone and state match then a typo in last name is a duplicate.
    record11 = {
        "first_name1": data["first_name"],
        "last_name1": simulate_typos(data["last_name"]),
        "email1": "",
        "phone1": data["phone"],
        "state1": data["state"],
        "first_name2": data["first_name"],
        "last_name2": data["last_name"],
        "email2": "",
        "phone2": data["phone"],
        "state2": data["state"],
        "label": 1,
    }
    synthetic_data.append(record11)

    return synthetic_data


def gather_data(n_contacts: int) -> List[dict]:
    os.environ["REDSHIFT_DB"] = os.environ["REDSHIFT_DATABASE"]
    os.environ["REDSHIFT_USERNAME"] = os.environ["REDSHIFT_CREDENTIAL_USERNAME"]
    os.environ["REDSHIFT_PASSWORD"] = os.environ["REDSHIFT_CREDENTIAL_PASSWORD"]
    os.environ["S3_TEMP_BUCKET"] = (
        os.getenv("S3_TEMP_BUCKET") or "tmc-member-indivisible"
    )

    rs = Redshift()

    data = (
        rs.query(
            f"""
        WITH ak_phones AS (
            SELECT user_id, phone
            FROM (
                SELECT
                    user_id,
                    phone,
                    ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY updated_at DESC) as rn
                FROM indivisible_ak.core_phone
            )
            WHERE rn = 1
        )
        SELECT
            LOWER(a.first_name) AS first_name,
            LOWER(a.last_name) AS last_name,
            LOWER(a.email) AS email,
            LOWER(COALESCE(a.state, '')) as state,
            LOWER(COALESCE(p.phone, '')) as phone,
            ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS rownum
        FROM indivisible_ak.core_user a
        LEFT JOIN ak_phones p
            ON p.user_id = a.id
        WHERE
            first_name IS NOT NULL
            AND last_name IS NOT NULL
            AND email IS NOT NULL
            AND state IS NOT NULL
            AND phone IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {n_contacts};
        """
        )
        or Table()
    )

    return data.to_dicts()


def main(n_contacts: int, output_csv: str):
    data = gather_data(n_contacts)
    print(f"Found {len(data)} contacts")

    synthetic_data = []
    for contact in data:
        synthetic_data.extend(generate_synthetic_data(contact))

    print(f"Generated {len(synthetic_data)} rows of synthetic data")

    # Define the fieldnames for the CSV file
    fieldnames = [
        "first_name1",
        "last_name1",
        "email1",
        "phone1",
        "state1",
        "first_name2",
        "last_name2",
        "email2",
        "phone2",
        "state2",
        "label",
    ]

    # Write the synthetic data to a CSV file
    with open(output_csv, "w", newline="", encoding="utf8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in synthetic_data:
            writer.writerow(row)


# If the script is run from the command line, parse the arguments and run the main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic contact data for deduplication."
    )
    parser.add_argument(
        "-n",
        "--n_contacts",
        type=int,
        required=True,
        help="The number of contact records to generate.",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=str,
        required=True,
        help="The path to the output CSV file.",
    )
    args = parser.parse_args()

    main(args.n_contacts, args.output_csv)
