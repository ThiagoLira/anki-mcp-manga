from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str
    allowed_telegram_user_ids: str = ""
    openrouter_api_key: str
    openrouter_model: str = "anthropic/claude-sonnet-4"

    sync_user: str = "user"
    sync_password: str = "password"
    sync_endpoint: str = "http://anki-sync:8080"
    collection_path: str = "/data/collection.anki2"
    kanji_deck: str = "Japones KANJI"
    manga_deck: str = "Japones Vocab Mangas"

    enable_panel_detection: bool = True
    panel_model_device: str = "cuda"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def allowed_user_ids(self) -> list[int]:
        if not self.allowed_telegram_user_ids:
            return []
        return [int(x.strip()) for x in self.allowed_telegram_user_ids.split(",")]


settings = Settings()
