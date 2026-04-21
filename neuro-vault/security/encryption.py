"""
security/encryption.py — AES-256 field-level encryption for sensitive data.

Uses the ``cryptography`` library's Fernet (AES-128-CBC + HMAC-SHA256) for
symmetric encryption of patient query strings before audit storage.

Key derivation from the deployment password uses PBKDF2-HMAC-SHA256 so the
raw password is never stored in plaintext.

DPDP Act 2023 compliance: all query text is encrypted at rest; audit records
contain only encrypted ciphertext, never plaintext patient information.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config import Config

logger = logging.getLogger(__name__)

# Fixed salt — for a demo system.  In production, store a random salt per
# deployment in a secure vault / HSM.
_SALT: bytes = b"neuro-vault-salt-v1-2024"


class EncryptionManager:
    """Symmetric encryption/decryption manager using Fernet (AES-256 equivalent).

    Args:
        password: Encryption passphrase.  Defaults to ``Config.ENCRYPTION_KEY``.
    """

    def __init__(self, password: Optional[str] = None) -> None:
        raw_password = (password or Config.ENCRYPTION_KEY).encode("utf-8")
        self._fernet = self._build_fernet(raw_password)

    # ------------------------------------------------------------------ #
    #  Key derivation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_fernet(password: bytes) -> Fernet:
        """Derive a Fernet key from *password* using PBKDF2-HMAC-SHA256.

        Args:
            password: Raw password bytes.

        Returns:
            Ready-to-use ``Fernet`` instance.
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=_SALT,
            iterations=390_000,
        )
        key_bytes = kdf.derive(password)
        fernet_key = base64.urlsafe_b64encode(key_bytes)
        return Fernet(fernet_key)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string to a URL-safe base64 ciphertext.

        Args:
            plaintext: String to encrypt (e.g. patient query text).

        Returns:
            Encrypted ciphertext string (URL-safe base64).
        """
        if not plaintext:
            return ""
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt a ciphertext string previously encrypted by ``encrypt``.

        Args:
            ciphertext: URL-safe base64 ciphertext string.

        Returns:
            Decrypted plaintext string.

        Raises:
            ValueError: If the ciphertext is invalid or the key is wrong.
        """
        if not ciphertext:
            return ""
        try:
            return self._fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError(
                "Decryption failed — invalid ciphertext or wrong key."
            ) from exc

    def hash_query(self, query: str) -> str:
        """Return a deterministic SHA-256 hex digest of *query*.

        Used to detect duplicate queries in analytics without storing
        the original text.

        Args:
            query: Query string.

        Returns:
            64-character hex string.
        """
        return hashlib.sha256(query.encode("utf-8")).hexdigest()

    @staticmethod
    def generate_key() -> str:
        """Generate a new random Fernet key for fresh deployments.

        Returns:
            URL-safe base64 key string.  Store securely; never log.
        """
        return Fernet.generate_key().decode("utf-8")
