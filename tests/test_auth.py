from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from mewhelp.api import create_app
from mewhelp.api.auth import SESSION_COOKIE
from mewhelp.config import Settings
from mewhelp.storage import session_scope
from mewhelp.storage.models import UserAccountRow
from mewhelp.workflow.graph import build_deps


PASSWORD = "correct horse battery"


def _email() -> str:
    return f"account-{uuid.uuid4().hex}@example.org"


def _register(client: TestClient, email: str | None = None) -> dict:
    response = client.post(
        "/api/auth/register",
        json={"email": email or _email(), "password": PASSWORD},
    )
    assert response.status_code == 200, response.text
    return response.json()["user"]


def _sse_events(response_text: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in response_text.replace("\r\n", "\n").split("\n\n"):
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        if name and data:
            events.append((name, data))
    return events


def test_auth_registration_login_and_logout(tmp_path):
    settings = Settings(
        MEWHELP_DATABASE_URL=f"sqlite:///{(tmp_path / 'accounts.db').as_posix()}"
    )
    app = create_app(build_deps(settings=settings))
    email = _email()

    with TestClient(app) as client:
        registered = client.post(
            "/api/auth/register", json={"email": email.upper(), "password": PASSWORD}
        )
        assert registered.status_code == 200
        assert registered.json()["user"]["email"] == email
        cookie = client.cookies.get(SESSION_COOKIE)
        assert cookie
        set_cookie = registered.headers.get("set-cookie", "").lower()
        assert "httponly" in set_cookie and "samesite=lax" in set_cookie

        with session_scope(settings=app.state.accounts.settings) as session:
            account = session.get(UserAccountRow, registered.json()["user"]["id"])
            assert account is not None
            assert PASSWORD not in account.password_hash
            assert account.password_hash.startswith("pbkdf2_sha256$")

        duplicate = client.post(
            "/api/auth/register", json={"email": email, "password": PASSWORD}
        )
        assert duplicate.status_code == 409

        assert client.post(
            "/api/auth/login", json={"email": email, "password": "wrong password"}
        ).status_code == 401
        client.post("/api/auth/logout")
        assert client.get("/api/auth/me").json() == {"authenticated": False, "user": None}

        logged_in = client.post(
            "/api/auth/login", json={"email": email, "password": PASSWORD}
        )
        assert logged_in.status_code == 200
        assert logged_in.json()["user"]["id"] == registered.json()["user"]["id"]
        assert client.get("/api/auth/me").json()["authenticated"] is True

        client.post("/api/auth/logout")
        assert client.get("/api/auth/me").json() == {"authenticated": False, "user": None}


def test_private_endpoints_require_a_registered_account(tmp_path):
    settings = Settings(
        MEWHELP_DATABASE_URL=f"sqlite:///{(tmp_path / 'private.db').as_posix()}"
    )
    app = create_app(build_deps(settings=settings))

    with TestClient(app) as client:
        requests = [
            client.get("/api/sessions"),
            client.get("/api/memories"),
            client.get("/api/orders"),
            client.get("/api/knowledge/graph"),
            client.post("/api/chat", json={"message": "尺码怎么选"}),
            client.post("/api/feedback", json={"session_id": "s-unauth"}),
        ]
        assert [response.status_code for response in requests] == [401] * len(requests)
        assert client.get("/api/auth/me").json() == {"authenticated": False, "user": None}


def test_spoofed_user_id_and_cross_account_session_access_are_rejected(tmp_path):
    settings = Settings(
        MEWHELP_DATABASE_URL=f"sqlite:///{(tmp_path / 'isolation.db').as_posix()}"
    )
    app = create_app(build_deps(settings=settings))

    with TestClient(app) as owner:
        owner_user = _register(owner)
        owner.post(
            "/api/memories",
            json={"user_id": "forged-owner", "text": "只属于原账号的资料"},
        )
        assert owner.get("/api/memories").json()["count"] == 1
        assert owner.get("/api/memories", params={"user_id": "forged-owner"}).json()["count"] == 1

        session_id = f"s-private-{uuid.uuid4().hex}"
        original = owner.post(
            "/api/chat",
            json={
                "user_id": "forged-owner",
                "session_id": session_id,
                "message": "这是账号 A 的私密对话内容",
            },
        )
        assert original.status_code == 200
        assert owner.get(f"/api/sessions/{session_id}").json()["ok"] is True

        with TestClient(app) as other:
            _register(other)
            other.post(
                "/api/memories",
                json={"user_id": owner_user["id"], "text": "尝试写入账号 A 的资料"},
            )
            other_memories = other.get("/api/memories").json()["items"]
            assert len(other_memories) == 1
            assert all(item["text"] != "只属于原账号的资料" for item in other_memories)

            detail = other.get(f"/api/sessions/{session_id}").json()
            assert detail["ok"] is False
            assert other.get("/api/sessions").json()["sessions"] == []

            takeover = other.post(
                "/api/chat",
                json={
                    "user_id": owner_user["id"],
                    "session_id": session_id,
                    "message": "尝试覆盖账号 A 的对话",
                },
            )
            assert takeover.status_code == 200
            events = _sse_events(takeover.text)
            assert [event for event, _ in events] == ["error"]
            assert "这次对话不可用" in events[0][1]
            assert "私密对话内容" not in takeover.text

            assert other.patch(
                f"/api/sessions/{session_id}", json={"title": "跨账号改名"}
            ).json()["ok"] is False
            assert other.delete(f"/api/sessions/{session_id}").json()["ok"] is False
            assert other.post(
                "/api/feedback",
                json={"session_id": session_id, "request_id": "forged-request"},
            ).json()["ok"] is False

        restored = owner.get(f"/api/sessions/{session_id}").json()
        assert restored["ok"] is True
        assert restored["messages"][0]["content"] == "这是账号 A 的私密对话内容"
        assert len(restored["messages"]) == 2
