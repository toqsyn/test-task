from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):

    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    VIDEO_PATH: Path = BASE_DIR / "assets" / "test.mp4"
    JSON_PATH: Path = BASE_DIR / "restricted_zones.json"

    class Config:

        env_file = BASE_DIR / ".env"

settings = Settings()
