"""Account registration and opaque browser sessions."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..config import Settings
from ..storage.db import session_scope
from ..storage.models import AuthSessionRow, UserAccountRow

SESSION_COOKIE = "mewhelp_session"
SESSION_TTL = timedelta(days=30)
PASSWORD_ROUNDS = 600_000
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@dataclass(frozen=True)
class Account:
    user_id: str
    email: str


class AccountError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ROUNDS)
    return f"pbkdf2_sha256${PASSWORD_ROUNDS}${salt.hex()}${digest.hex()}"


def _password_matches(password: str, stored: str) -> bool:
    try:
        algorithm, rounds_text, salt_hex, digest_hex = stored.split("$")
        rounds = int(rounds_text)
        if algorithm != "pbkdf2_sha256" or not 100_000 <= rounds <= 1_000_000:
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), rounds
        )
        return hmac.compare_digest(candidate.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


class AccountStore:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def normalize_email(email: str) -> str:
        normalized = email.strip().lower()
        if len(normalized) > 254 or not EMAIL_RE.fullmatch(normalized):
            raise AccountError("INVALID_EMAIL")
        return normalized

    @staticmethod
    def validate_password(password: str) -> None:
        if len(password) < 10:
            raise AccountError("PASSWORD_TOO_SHORT")
        if len(password) > 128:
            raise AccountError("PASSWORD_TOO_LONG")

    def register(self, email: str, password: str) -> tuple[Account, str]:
        normalized = self.normalize_email(email)
        self.validate_password(password)
        token = secrets.token_urlsafe(32)
        account = Account(user_id=str(uuid.uuid4()), email=normalized)
        try:
            with session_scope(settings=self.settings) as session:
                row = UserAccountRow(
                    user_id=account.user_id,
                    email=normalized,
                    password_hash=_password_hash(password),
                )
                session.add(row)
                session.flush()
                session.add(
                    AuthSessionRow(
                        token_hash=_token_digest(token),
                        user_id=account.user_id,
                        expires_at=datetime.now(timezone.utc) + SESSION_TTL,
                    )
                )
        except IntegrityError as exc:
            raise AccountError("EMAIL_EXISTS") from exc
        return account, token

    def login(self, email: str, password: str) -> tuple[Account, str]:
        normalized = self.normalize_email(email)
        token = secrets.token_urlsafe(32)
        with session_scope(settings=self.settings) as session:
            row = session.scalar(
                select(UserAccountRow).where(UserAccountRow.email == normalized)
            )
            if row is None or not _password_matches(password, row.password_hash):
                raise AccountError("INVALID_CREDENTIALS")
            session.add(
                AuthSessionRow(
                    token_hash=_token_digest(token),
                    user_id=row.user_id,
                    expires_at=datetime.now(timezone.utc) + SESSION_TTL,
                )
            )
            account = Account(user_id=row.user_id, email=row.email)
        return account, token

    def from_token(self, token: str | None) -> Account | None:
        if not token:
            return None
        now = datetime.now(timezone.utc)
        with session_scope(settings=self.settings) as session:
            row = session.scalar(
                select(UserAccountRow)
                .join(AuthSessionRow, AuthSessionRow.user_id == UserAccountRow.user_id)
                .where(
                    AuthSessionRow.token_hash == _token_digest(token),
                    AuthSessionRow.expires_at > now,
                )
            )
            if row is None:
                return None
            return Account(user_id=row.user_id, email=row.email)

    def logout(self, token: str | None) -> None:
        if not token:
            return
        digest = _token_digest(token)
        with session_scope(settings=self.settings) as session:
            row = session.get(AuthSessionRow, digest)
            if row is not None:
                session.delete(row)
