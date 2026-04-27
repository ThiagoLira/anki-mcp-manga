from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str
    allowed_telegram_user_ids: str = ""
    allowed_telegram_usernames: str = ""
    openrouter_api_key: str
    openrouter_model: str = "anthropic/claude-sonnet-4"

    # Local LLM (llama-server pod on the janus k3s agent node, reached over
    # Tailscale). Probed at agent init; if reachable, used in preference to
    # OpenRouter. The pod exposes containerPort 8080 via hostPort 9080 — port
    # 8080 on every node IP is claimed cluster-wide by the anki-sync-external
    # LoadBalancer Service, so we picked 9080 for llama. Override per-host via
    # LOCAL_LLM_URL env var.
    local_llm_url: str = "http://100.81.144.115:9080/v1"
    local_llm_model: str = "local"
    local_llm_probe_timeout_s: float = 1.0

    # Irodori-TTS HTTP server (VoiceDesign-only, runs on janus over Tailscale).
    # Primary TTS path; falls back to in-process Kokoro on any timeout / connect /
    # HTTP error. Set to empty string to disable Irodori entirely (Kokoro-only).
    irodori_tts_url: str = "http://100.81.144.115:8200"
    irodori_tts_timeout_s: float = 90.0

    sync_user: str = "user"
    sync_password: str = "password"
    sync_endpoint: str = "http://anki-sync:8080"
    collection_path: str = "/data/collection.anki2"
    kanji_deck: str = "Japones KANJI"
    manga_deck: str = "Japones Vocab Mangas"

    enable_panel_detection: bool = True
    panel_model_device: str = "cuda"
    panel_model_path: str = "models/panel_detector.onnx"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def allowed_user_ids(self) -> list[int]:
        if not self.allowed_telegram_user_ids:
            return []
        return [int(x.strip()) for x in self.allowed_telegram_user_ids.split(",")]

    @property
    def allowed_usernames(self) -> list[str]:
        if not self.allowed_telegram_usernames:
            return []
        return [x.strip().lstrip("@").lower() for x in self.allowed_telegram_usernames.split(",")]


settings = Settings()
