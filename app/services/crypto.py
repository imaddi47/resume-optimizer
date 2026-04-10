from cryptography.fernet import Fernet

# Import get_settings only if it hasn't already been set in this module's namespace.
# This pattern allows unittest.mock.patch("app.services.crypto.get_settings") to
# survive an importlib.reload() call made inside the patch context.
try:
    get_settings  # type: ignore[used-before-def]  # noqa: F821
except NameError:
    from app.config import get_settings  # noqa: F401


def _get_fernet() -> Fernet:
    key = get_settings().ENCRYPTION_KEY
    if not key:
        raise RuntimeError(
            "ENCRYPTION_KEY not set. Generate one with: "
            "python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    return _get_fernet().decrypt(ciphertext.encode()).decode()
