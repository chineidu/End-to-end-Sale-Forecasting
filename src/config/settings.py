import re
from urllib.parse import quote

from dotenv import load_dotenv
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def fix_url_credentials(url: str) -> str:
    """
    Fix URL by properly encoding special characters in credentials.

    Parameters
    ----------
    url : str
        The URL to fix.

    Returns
    -------
    fixed_url : str
        The fixed URL.
    """
    try:
        # More flexible pattern that accepts any scheme format
        # Captures: anything://username:password@host_and_rest
        pattern = r"^([^:]+://)([^:/?#]+):([^@]+)@(.+)$"
        match = re.match(pattern, url)

        if match:
            scheme, username, password, host_part = match.groups()
            # URL encode the username and password
            # safe='' means encode all special characters
            encoded_username = quote(username, safe="")
            encoded_password = quote(password, safe="")

            # Reconstruct the URL
            fixed_url = f"{scheme}{encoded_username}:{encoded_password}@{host_part}"

            # Extract scheme name for logging
            scheme_name = scheme.rstrip("://")  # noqa: B005
            print(f"Fixed {scheme_name!r} URL encoding for special characters")

            return fixed_url

        print("WARNING: No regex match found!")
        return url

    except Exception as e:
        print(f"Could not fix URL: {e}")
        return url


class Settings(BaseSettings):
    """
    Settings class for managing application configuration.
    """

    model_config = SettingsConfigDict(env_file=".env")

    # ======= MLFlow =======
    MLFLOW_HOST: str = "localhost"
    MLFLOW_PORT: int = 5001

    # ======= AWS/MinIO =======
    AWS_S3_HOST: str = "localhost"
    AWS_S3_PORT: int = 9000
    AWS_S3_BUCKET: str = "mlflow-artifacts"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: SecretStr = SecretStr("minioadmin")
    AWS_DEFAULT_REGION: str = "us-east-1"

    @field_validator("MLFLOW_PORT", "S3_PORT", mode="before")
    @classmethod
    def parse_port_fields(cls, v: str | int) -> int:
        """Parses port fields to ensure they are integers."""
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                raise ValueError(f"Invalid port value: {v}") from None

        if isinstance(v, int) and not (1 <= v <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {v}")

        return v

    @property
    def mlflow_tracking_uri(self) -> str:
        """
        Constructs the MLflow tracking URI.

        Returns
        -------
        str
            Complete MLflow tracking URI in the format:
            http://host:port
        """
        url: str = f"http://{self.MLFLOW_HOST}:{self.MLFLOW_PORT}"
        return fix_url_credentials(url)

    @property
    def mlflow_s3_endpoint_url(self) -> str:
        """
        Constructs the S3 endpoint URL for MLflow.

        Returns
        -------
        str
            Complete S3 endpoint URL in the format:
            http://host:port
        """
        url: str = f"http://{self.S3_HOST}:{self.S3_PORT}"
        return fix_url_credentials(url)


def refresh_settings() -> Settings:
    """Refresh environment variables and return new Settings instance.

    This function reloads environment variables from .env file and creates
    a new Settings instance with the updated values.

    Returns
    -------
    Settings
        A new Settings instance with refreshed environment variables
    """
    load_dotenv(override=True)
    return Settings()  # type: ignore


app_settings: Settings = refresh_settings()
