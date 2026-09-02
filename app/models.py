"""
User accounts and job ownership.

Jobs live in two places on purpose: the row here records who owns it and how it
ended, while the heavy output (clips, stems, result.json) stays on disk under
jobs/<id>/. The database stays small and the analysis code keeps writing plain
files.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(200))
    picture: Mapped[str | None] = mapped_column(String(1000))

    # Null for Google-only accounts; null google_sub for password-only accounts.
    # An account can have both, which is what links a Google sign-in to an
    # existing email account instead of duplicating the user.
    password_hash: Mapped[str | None] = mapped_column(String(200))
    google_sub: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_login_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    jobs: Mapped[list["Job"]] = relationship(back_populates="user",
                                             cascade="all, delete-orphan")

    @property
    def display_name(self) -> str:
        return self.name or self.email.split("@")[0]

    def public(self) -> dict:
        return {"id": self.id, "email": self.email, "name": self.display_name,
                "picture": self.picture,
                "has_password": bool(self.password_hash),
                "google_linked": bool(self.google_sub)}


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                         index=True)
    source_name: Mapped[str] = mapped_column(String(500), default="")
    state: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    tone_count: Mapped[int] = mapped_column(Integer, default=0)
    tempo_bpm: Mapped[int] = mapped_column(Integer, default=0)
    from_url: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True),
                                                    default=_now, index=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    user: Mapped[User] = relationship(back_populates="jobs")

    def public(self) -> dict:
        return {"job_id": self.id, "source_name": self.source_name,
                "state": self.state, "progress": self.progress,
                "error": self.error, "tone_count": self.tone_count,
                "tempo_bpm": self.tempo_bpm,
                "created_at": self.created_at.isoformat() if self.created_at else None,
                "finished_at": self.finished_at.isoformat() if self.finished_at else None}
