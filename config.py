from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    SECRET_KEY: str

    ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    DATABASE_URL: str

    OPENAI_API_KEY: str | None = None

    INTERNAL_API_KEY: str | None = None


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


settings = Settings()


SECRET_KEY = settings.SECRET_KEY

ALGORITHM = settings.ALGORITHM

ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

DATABASE_URL = settings.DATABASE_URL

OPENAI_API_KEY = settings.OPENAI_API_KEY

INTERNAL_API_KEY = settings.INTERNAL_API_KEY