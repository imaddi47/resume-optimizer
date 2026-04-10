import pytest
from unittest.mock import patch


@pytest.fixture
def fernet_key():
    """A valid Fernet key for testing."""
    from cryptography.fernet import Fernet
    return Fernet.generate_key().decode()


def test_encrypt_decrypt_roundtrip(fernet_key):
    with patch("app.services.crypto.get_settings") as mock:
        mock.return_value.ENCRYPTION_KEY = fernet_key
        from importlib import reload
        import app.services.crypto as mod
        reload(mod)
        from app.services.crypto import encrypt, decrypt

        plaintext = "sk-test-key-12345"
        encrypted = encrypt(plaintext)
        assert encrypted != plaintext
        assert decrypt(encrypted) == plaintext


def test_encrypt_produces_different_ciphertexts(fernet_key):
    with patch("app.services.crypto.get_settings") as mock:
        mock.return_value.ENCRYPTION_KEY = fernet_key
        from importlib import reload
        import app.services.crypto as mod
        reload(mod)
        from app.services.crypto import encrypt

        a = encrypt("same-key")
        b = encrypt("same-key")
        # Fernet includes a timestamp, so ciphertexts differ
        assert a != b


def test_decrypt_wrong_key_fails(fernet_key):
    from cryptography.fernet import Fernet

    with patch("app.services.crypto.get_settings") as mock:
        mock.return_value.ENCRYPTION_KEY = fernet_key
        from importlib import reload
        import app.services.crypto as mod
        reload(mod)
        from app.services.crypto import encrypt

        encrypted = encrypt("my-secret")

    other_key = Fernet.generate_key().decode()
    with patch("app.services.crypto.get_settings") as mock:
        mock.return_value.ENCRYPTION_KEY = other_key
        from importlib import reload
        import app.services.crypto as mod
        reload(mod)
        from app.services.crypto import decrypt

        with pytest.raises(Exception):
            decrypt(encrypted)
