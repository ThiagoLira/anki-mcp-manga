from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    telegram_bot_token: str
    allowed_telegram_user_ids: str = ""
    allowed_telegram_usernames: str = ""
    openrouter_api_key: str
    openrouter_model: str = "anthropic/claude-sonnet-4"

    # Local LLM (llama-server-mtp-cuda pod on jobim/RTX 3090 Ti, reached
    # over Tailscale). Probed at agent init; if reachable, used in
    # preference to OpenRouter. Points at the MTP speculative-decoding
    # build (Qwen3.6-27B UD-Q4_K_XL, ~35 tok/s) rather than the slower
    # Vulkan janus pod (~15 tok/s on Q8_0). Both expose containerPort
    # 8080 via hostPort 9090 — port 8080 on every node IP is claimed
    # cluster-wide by the anki-sync-external LoadBalancer Service, and
    # 9080 was the original llama-server port; 9090 is the MTP variant.
    # Override per-host via LOCAL_LLM_URL env var.
    local_llm_url: str = "http://100.86.254.13:9090/v1"
    local_llm_model: str = "local"
    local_llm_probe_timeout_s: float = 1.0

    # Irodori-TTS HTTP server (CUDA pod on jobim/RTX 3090 Ti, reached over
    # Tailscale). Primary TTS path; falls back to the Kokoro-FastAPI service
    # on any timeout / connect / HTTP error. Set to empty string to disable
    # Irodori entirely (Kokoro-only). Was on janus/CPU until 2026-05-09 — if
    # you see the old 100.81.144.115:8200 in logs, the bot is running pre-
    # migration code.
    irodori_tts_url: str = "http://100.86.254.13:8200"
    irodori_tts_timeout_s: float = 90.0

    # Kokoro-FastAPI service (GPU pod on jobim, k8s LoadBalancer at 8880).
    # Fallback path when Irodori is unreachable. Replaces the previous
    # in-process kokoro-onnx + misaki[ja] dependency, dropping ~325 MB of
    # baked-in model weights and the espeak-ng/cmake build deps. Voice prefix
    # `jf_` = Japanese female; pick another voice (e.g. `jm_kumo`) by setting
    # KOKORO_TTS_VOICE. The teagolab-1 IP still works because k3s' klipper-lb
    # binds the LB port on every node.
    kokoro_tts_url: str = "http://100.102.150.83:8880"
    kokoro_tts_timeout_s: float = 30.0
    kokoro_tts_voice: str = "jf_alpha"

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
