from config import Config, reset_config


def test_config_reads_secret_file_env(monkeypatch, tmp_path):
    secret_file = tmp_path / "api_key.txt"
    secret_file.write_text("from-file-secret\n", encoding="utf-8")

    monkeypatch.delenv("API_API_KEY", raising=False)
    monkeypatch.setenv("API_API_KEY_FILE", str(secret_file))

    reset_config()
    config = Config()

    assert config.api_config.api_key == "from-file-secret"

    reset_config()
