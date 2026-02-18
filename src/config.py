from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mcp_auth_token: str
    sync_user: str = "user"
    sync_password: str = "password"
    sync_endpoint: str = "http://anki-sync:8080"
    collection_path: str = "/data/collection.anki2"
    kanji_deck: str = "Japones KANJI"
    manga_deck: str = "Japones Vocab Mangas"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
