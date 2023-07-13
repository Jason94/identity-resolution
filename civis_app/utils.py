import os


def init_rs_env():
    """Match the environment variables Civis generates with the names Parsons expects."""
    os.environ["REDSHIFT_DB"] = os.environ["REDSHIFT_DATABASE"]
    os.environ["REDSHIFT_USERNAME"] = os.environ["REDSHIFT_CREDENTIAL_USERNAME"]
    os.environ["REDSHIFT_PASSWORD"] = os.environ["REDSHIFT_CREDENTIAL_PASSWORD"]
    os.environ["S3_TEMP_BUCKET"] = os.environ["S3_TEMP_BUCKET"]
