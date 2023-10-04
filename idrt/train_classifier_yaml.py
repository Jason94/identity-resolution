import argparse

from train_yaml import yaml_to_namespace
from train_classifier import train_classifier


def main():
    parser = argparse.ArgumentParser(
        description="Run classifier trainer with YAML arguments."
    )
    parser.add_argument("yaml_path", type=str, help="Path to the YAML file")

    args = parser.parse_args()

    yaml_args = yaml_to_namespace(args.yaml_path)
    train_classifier(yaml_args)


if __name__ == "__main__":
    main()
