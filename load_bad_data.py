from data import NameDataset
from model import create_char_to_int

if __name__ == "__main__":
    first_name = "shadrach"

    char_to_int, chars = create_char_to_int()
    dataset = NameDataset("data/eval.csv", char_to_int, debug=True)

    for _, _, _, (fname1, lname1, email1), (fname2, lname2, email2) in dataset:  # type: ignore
        if fname1 == first_name or fname2 == first_name:
            breakpoint()

    pass
