from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    APP_ENV: str = "development"
    class Config:
        env_file = ".env"

settings = Settings()
