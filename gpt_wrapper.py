"""Simple custom GPT wrapper for Implement Consulting Group consultants.

This module provides a command-line interface where consultants can supply
workflow context (for example an email drafting scenario). The tool calls the
OpenAI GPT-4 model, injects the structured insights into a standard template,
and logs all usage to a local SQLite database for lightweight analytics.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Dict

try:
    # The OpenAI SDK exposes the "OpenAI" client entry point in recent releases.
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - defensive dependency guard.
    raise SystemExit(
        "The 'openai' package is required. Install it with 'pip install openai'."
    ) from exc


def _default_template() -> "WorkflowTemplate":
    """Return the default template used across workflows."""

    return WorkflowTemplate(
        name="Implement Consulting Group Consultant Summary",
        template=(
            "Implement Consulting Group — {workflow_type}\n"
            "Context Provided:\n{context}\n\n"
            "Key Insights:\n{key_insights}\n\n"
            "Recommended Consultant Actions:\n{recommended_actions}\n\n"
            "Risks & Dependencies:\n{risks}\n\n"
            "Stakeholder & Communication Tips:\n{communication_tips}\n"
        ),
    )


@dataclass
class WorkflowTemplate:
    """Represent a named template that can be populated with GPT output."""

    name: str
    template: str

    def render(self, values: Dict[str, str]) -> str:
        """Populate the template with safe defaults for missing fields."""

        defaults = {
            "workflow_type": "(unspecified)",
            "context": "(no context captured)",
            "key_insights": "No insights returned.",
            "recommended_actions": "No actions returned.",
            "risks": "No risks highlighted.",
            "communication_tips": "No communication tips returned.",
        }
        defaults.update({k: v for k, v in values.items() if v})
        return self.template.format(**defaults)


class UsageLogger:
    """Persist usage data to a lightweight SQLite database."""

    def __init__(self, database_path: str = "usage_logs.db") -> None:
        self.database_path = Path(database_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the database schema if it does not already exist."""

        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    workflow_type TEXT NOT NULL,
                    context TEXT NOT NULL,
                    suggestions TEXT NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER
                )
                """
            )

    def log(
        self,
        *,
        workflow_type: str,
        context: str,
        suggestions: Dict[str, Any],
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        """Insert a new usage log entry into the database."""

        with sqlite3.connect(self.database_path) as conn:
            conn.execute(
                """
                INSERT INTO usage_log (
                    created_at,
                    workflow_type,
                    context,
                    suggestions,
                    prompt_tokens,
                    completion_tokens
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(timespec="seconds"),
                    workflow_type,
                    context,
                    json.dumps(suggestions, ensure_ascii=False),
                    prompt_tokens,
                    completion_tokens,
                ),
            )


class ConsultantGPTWrapper:
    """Encapsulate OpenAI interactions and template population."""

    def __init__(
        self,
        *,
        model: str,
        template: WorkflowTemplate | None = None,
        logger: UsageLogger | None = None,
    ) -> None:
        self.model = model
        self.template = template or _default_template()
        self.logger = logger or UsageLogger()
        self.client = OpenAI()

    def generate_suggestions(
        self, *, workflow_type: str, context: str
    ) -> Dict[str, Any]:
        """Call the OpenAI API and return structured consultant guidance."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior management consultant at Implement Consulting "
                            "Group. Provide practical, structured advice in concise bullet "
                            "points. Respond as compact JSON with keys: key_insights, "
                            "recommended_actions, risks, communication_tips."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Workflow type: {workflow_type}\n"
                            f"Context summary: {context}\n"
                            "Return actionable suggestions tailored to Implement Consulting "
                            "Group's consulting style."
                        ),
                    },
                ],
            )
        except Exception as api_error:  # pragma: no cover - network failure scenario.
            raise RuntimeError(
                "Failed to call the OpenAI API. Verify your network connection and API key."
            ) from api_error

        message = response.choices[0].message
        content = message.content or ""

        structured = self._parse_response_content(content)
        prompt_tokens = getattr(response.usage, "prompt_tokens", None)
        completion_tokens = getattr(response.usage, "completion_tokens", None)

        # Persist the usage details for lightweight analytics.
        try:
            self.logger.log(
                workflow_type=workflow_type,
                context=context,
                suggestions=structured,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        except sqlite3.Error as db_error:  # pragma: no cover - logging failure is non-fatal.
            # The application should continue even if logging fails; we surface a warning.
            print(f"[warning] Failed to log usage: {db_error}")

        return structured

    def populate_template(
        self, *, workflow_type: str, context: str, suggestions: Dict[str, Any]
    ) -> str:
        """Convert structured suggestions into a formatted consultant brief."""

        formatted_values = {
            "workflow_type": workflow_type,
            "context": context,
            "key_insights": self._stringify_section(suggestions.get("key_insights")),
            "recommended_actions": self._stringify_section(
                suggestions.get("recommended_actions")
            ),
            "risks": self._stringify_section(suggestions.get("risks")),
            "communication_tips": self._stringify_section(
                suggestions.get("communication_tips")
            ),
        }
        return self.template.render(formatted_values)

    @staticmethod
    def _parse_response_content(content: str) -> Dict[str, Any]:
        """Convert the language model response to structured Python data."""

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fall back to returning the raw string if JSON parsing fails.
            return {
                "key_insights": content.strip(),
                "recommended_actions": "",
                "risks": "",
                "communication_tips": "",
            }

    @staticmethod
    def _stringify_section(value: Any) -> str:
        """Convert different data types into human-readable bullet lists."""

        if value is None:
            return "(no information provided)"

        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned if cleaned else "(no information provided)"

        if isinstance(value, dict):
            items = [f"- {key}: {ConsultantGPTWrapper._stringify_section(val)}" for key, val in value.items()]
            return "\n".join(items)

        if isinstance(value, Iterable):
            items = [f"- {ConsultantGPTWrapper._stringify_section(item)}" for item in value]
            return "\n".join(items)

        return str(value)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for the wrapper."""

    parser = argparse.ArgumentParser(
        description=(
            "Call GPT-4 with Implement Consulting Group specific prompts and log usage."
        )
    )
    parser.add_argument(
        "--workflow-type",
        "-w",
        default="General Advisory",
        help="Short label describing the consulting workflow (e.g. 'Email Draft').",
    )
    parser.add_argument(
        "--context",
        "-c",
        default=None,
        help="Detailed context for the workflow. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4",
        help="OpenAI model identifier to use. Defaults to 'gpt-4'.",
    )
    parser.add_argument(
        "--database",
        "-d",
        default="usage_logs.db",
        help="Path to the SQLite database used for usage logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the command-line interface."""

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY environment variable is not set. Please export your API key."
        )

    args = parse_arguments()

    context = args.context
    if not context:
        try:
            context = input("Enter workflow context details: \n")
        except KeyboardInterrupt:
            raise SystemExit("\nAborted by user before providing context.") from None

    wrapper = ConsultantGPTWrapper(
        model=args.model, template=_default_template(), logger=UsageLogger(args.database)
    )

    suggestions = wrapper.generate_suggestions(
        workflow_type=args.workflow_type, context=context
    )
    output = wrapper.populate_template(
        workflow_type=args.workflow_type, context=context, suggestions=suggestions
    )

    print("\n=== Consultant Brief ===\n")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - defensive guard for CLI usage.
        print(f"[error] {error}")
        raise
