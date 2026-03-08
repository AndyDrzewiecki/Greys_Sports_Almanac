"""Shared functionality used by all Market Watch agents."""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import select

import config
from storage.database import AuditLog, KBUpdate, get_session, init_db

LOGGER = logging.getLogger(__name__)

_OLLAMA_TIMEOUT = 120  # seconds — 70b models can be slow on first token


class BaseAgent:
    """Base class that provides KB, Ollama LLM, and audit logging helpers."""

    def __init__(self, name: str, kb_path: str) -> None:
        """Initialize the agent, load KB text, and wire shared services."""

        self.name = name
        self.base_dir = Path(__file__).resolve().parents[1]
        self.kb_dir = self.base_dir / kb_path
        self.kb_file = self.kb_dir / "memory.md"
        init_db()
        self.memory = self.read_kb()
        self.session_factory = get_session

    def read_kb(self) -> str:
        """Read and return the full KB contents, or an empty string if missing."""

        try:
            return self.kb_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            LOGGER.warning("KB file missing for agent %s at %s", self.name, self.kb_file)
            return ""

    def write_kb_staging(self, note: str, category: str) -> None:
        """Append a timestamped note to the staging KB section and persist it to the database."""

        timestamp = datetime.now(timezone.utc).isoformat()
        with self.session_factory() as session:
            update = KBUpdate(
                agent_name=self.name,
                update_type=category,
                content=note,
                status="staging",
            )
            session.add(update)
            session.flush()
            entry = f"- [{timestamp}] [KB_NOTE_ID:{update.id}] [{category}] {note}"
            existing = self.read_kb()
            updated = self._append_to_section(existing, "## Self-Improvement Notes (Staging)", entry)
            self.kb_dir.mkdir(parents=True, exist_ok=True)
            self.kb_file.write_text(updated, encoding="utf-8")
            self.memory = updated

    def promote_kb_note(self, note_id: int) -> None:
        """Move a staging KB note to the active KB section and mark it approved in the database."""

        with self.session_factory() as session:
            update = session.scalar(select(KBUpdate).where(KBUpdate.id == note_id))
            if update is None:
                raise ValueError(f"KB note id {note_id} not found")

            kb_text = self.read_kb()
            note_token = f"[KB_NOTE_ID:{note_id}]"
            active_entry = (
                f"- [{datetime.now(timezone.utc).isoformat()}] {note_token} "
                f"[{update.update_type}] {update.content}"
            )
            kb_text = self._remove_line_containing(kb_text, note_token)
            kb_text = self._append_to_section(kb_text, "## Self-Improvement Notes (Active)", active_entry)
            self.kb_file.write_text(kb_text, encoding="utf-8")
            update.status = "approved"
            update.reviewed_at = datetime.now(timezone.utc)
            self.memory = kb_text

    def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        include_kb: bool = True,
        model_tier: str = "reasoning",
    ) -> str:
        """Call the local Ollama instance, log the audit event, and return text output.

        Routes to the primary URL (ZBook 70b) for model_tier='reasoning' and to the
        secondary URL (EliteBook 8b) for 'extraction' and 'fast' tiers. Falls back to
        the other URL when the primary is unreachable.

        Args:
            system_prompt: Instructions for the LLM.
            user_message: User-facing content (data payload, question, etc.).
            include_kb: When True, the agent's KB is prepended to the system prompt.
            model_tier: One of 'reasoning', 'extraction', 'fast'. Controls which model
                        and which machine is used.
        """

        effective_system_prompt = system_prompt
        if include_kb and self.memory:
            effective_system_prompt = f"{system_prompt}\n\nKnowledge Base Context:\n{self.memory}"

        model = config.OLLAMA_MODELS.get(model_tier, config.OLLAMA_MODELS["reasoning"])
        primary_url = config.OLLAMA_TIER_ROUTING.get(model_tier, config.OLLAMA_PRIMARY_URL)
        fallback_url = (
            config.OLLAMA_SECONDARY_URL
            if primary_url == config.OLLAMA_PRIMARY_URL
            else config.OLLAMA_PRIMARY_URL
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
        }

        response_text = self._post_ollama(primary_url, payload, fallback_url)
        self.log_audit(
            event_type="llm_call",
            input_summary=user_message[:500],
            output_summary=response_text[:1000],
        )
        return response_text

    def call_claude(
        self,
        system_prompt: str,
        user_message: str,
        include_kb: bool = True,
        model_tier: str = "reasoning",
    ) -> str:
        """Backward-compatible alias for call_llm.

        All agents originally called call_claude; this alias keeps them working
        without modification while routing through Ollama.
        """

        return self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            include_kb=include_kb,
            model_tier=model_tier,
        )

    def log_audit(
        self,
        event_type: str,
        input_summary: str,
        output_summary: str,
        error: str | None = None,
        data_quality_flags: str | None = None,
    ) -> None:
        """Write an event to the audit log table."""

        with self.session_factory() as session:
            session.add(
                AuditLog(
                    agent_name=self.name,
                    event_type=event_type,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    data_quality_flags=data_quality_flags,
                    error=error,
                )
            )

    def validate_json_output(self, raw: str, required_keys: list[str]) -> dict[str, Any]:
        """Parse LLM JSON output, strip markdown fences, and validate required keys."""

        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON output: {exc}") from exc

        missing = [key for key in required_keys if key not in parsed]
        if missing:
            raise ValueError(f"JSON output missing required keys: {missing}")
        return parsed

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _post_ollama(
        self,
        url: str,
        payload: dict[str, Any],
        fallback_url: str,
        retries: int = 2,
    ) -> str:
        """POST to the Ollama /api/chat endpoint with retry and URL fallback."""

        endpoint = f"{url.rstrip('/')}/api/chat"
        for attempt in range(retries + 1):
            try:
                resp = requests.post(endpoint, json=payload, timeout=_OLLAMA_TIMEOUT)
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"].strip()
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == 0 and fallback_url != url:
                    LOGGER.warning(
                        "Primary Ollama at %s unreachable (%s), trying fallback %s",
                        url, exc, fallback_url,
                    )
                    endpoint = f"{fallback_url.rstrip('/')}/api/chat"
                elif attempt < retries:
                    wait = 2 ** attempt
                    LOGGER.warning("Ollama request failed (attempt %d), retrying in %ds", attempt + 1, wait)
                    time.sleep(wait)
                else:
                    self.log_audit(
                        event_type="llm_call_failed",
                        input_summary=str(payload.get("messages", [{}])[-1].get("content", ""))[:500],
                        output_summary="",
                        error=str(exc),
                    )
                    raise RuntimeError(
                        f"Ollama unreachable at {url} and {fallback_url}: {exc}"
                    ) from exc
            except Exception as exc:
                self.log_audit(
                    event_type="llm_call_failed",
                    input_summary=str(payload.get("messages", [{}])[-1].get("content", ""))[:500],
                    output_summary="",
                    error=str(exc),
                )
                raise
        return ""  # unreachable — satisfies type checker

    def _append_to_section(self, content: str, section_header: str, entry: str) -> str:
        """Append a line to a markdown section, creating the section if needed."""

        if not content:
            content = f"{section_header}\n"
        if section_header not in content:
            content = f"{content.rstrip()}\n\n{section_header}\n"

        lines = content.splitlines()
        for index, line in enumerate(lines):
            if line.strip() == section_header:
                insert_at = index + 1
                while insert_at < len(lines) and not lines[insert_at].startswith("## "):
                    insert_at += 1
                lines.insert(insert_at, entry)
                return "\n".join(lines).rstrip() + "\n"
        return f"{content.rstrip()}\n{entry}\n"

    def _remove_line_containing(self, content: str, needle: str) -> str:
        """Remove the first line containing a specific token from the KB markdown."""

        lines = content.splitlines()
        filtered = []
        removed = False
        for line in lines:
            if not removed and needle in line:
                removed = True
                continue
            filtered.append(line)
        return "\n".join(filtered).rstrip() + "\n"
