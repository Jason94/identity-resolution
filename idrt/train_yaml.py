import argparse
import yaml


def yaml_to_namespace(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    namespace = argparse.Namespace(**yaml_data)

    return namespace


if __name__ == "__main__":
    from train import train

    parser = argparse.ArgumentParser(description="Run trainer with YAML arguments.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML file")

    args = parser.parse_args()

    yaml_args = yaml_to_namespace(args.yaml_path)
    train(yaml_args)
