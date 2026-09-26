"""The hot write path (HLD 7).

Validate, redact, truncate, append, return. No embedding, no LLM call, no
graph write. If consolidation is down, ingestion keeps succeeding and the
backlog drains later: memory degrading to "nothing was learned today" is
acceptable, memory blocking an agent mid-task is not.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from .config import IngestConfig
from .models import Episode, ValidationError, new_id
from .redact import RedactionStats, redact
from .store import SQLiteStore

__all__ = ["Ingestor", "IngestResult"]


@dataclass(slots=True)
class IngestResult:
    episode_id: str
    accepted: bool
    deduplicated: bool = False
    sampled_out: bool = False
    redactions: int = 0
    truncated_fields: int = 0


class Ingestor:
    def __init__(
        self,
        store: SQLiteStore,
        config: IngestConfig | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.store = store
        self.config = config or IngestConfig()
        self._rng = rng or random.Random()

    # -- backpressure ------------------------------------------------------

    def _keep_under_backpressure(self, episode: Episode) -> bool:
        """HLD 7. Above the high-water mark, keep every failure and a
        decreasing fraction of successes.

        Whatever this drops is still counted in the unsampled counter table,
        so base rates stay honest even as the buffer skews.
        """
        depth = self.store.buffer_depth()
        hw = self.config.buffer_high_water
        if depth < hw:
            return True

        status = episode.outcome.status if episode.outcome else None
        if status in ("failure", "partial"):
            return True  # failures are never dropped

        # Keep probability falls off as the buffer runs past the mark.
        overflow = (depth - hw) / max(hw, 1)
        keep_p = 1.0 / (1.0 + overflow)
        return self._rng.random() < keep_p

    # -- sanitisation ------------------------------------------------------

    def _sanitize(self, episode: Episode) -> tuple[Episode, int, int]:
        stats = RedactionStats()
        truncated = 0
        cfg = self.config

        if len(episode.steps) > cfg.max_steps_per_episode:
            # Keep the head and the tail: the tail holds the failure, the head
            # holds the setup. The middle of a 900-step loop is noise.
            keep = cfg.max_steps_per_episode
            head = keep // 2
            episode.steps = episode.steps[:head] + episode.steps[-(keep - head):]
            truncated += 1

        for step in episode.steps:
            if cfg.redact:
                step.input = redact(step.input, stats)
                step.output_summary = redact(step.output_summary, stats)
                if step.error:
                    step.error = redact(step.error, stats)

            limit = cfg.max_output_summary_bytes
            if len(step.output_summary.encode("utf-8")) > limit:
                step.output_summary = (
                    step.output_summary.encode("utf-8")[:limit].decode("utf-8", "ignore")
                    + " …[truncated]"
                )
                truncated += 1
            if len(step.input.encode("utf-8")) > limit:
                step.input = (
                    step.input.encode("utf-8")[:limit].decode("utf-8", "ignore")
                    + " …[truncated]"
                )
                truncated += 1

        if cfg.redact:
            episode.task.description = redact(episode.task.description, stats)
            if episode.outcome:
                episode.outcome.detail = redact(episode.outcome.detail, stats)
                episode.outcome.evidence = redact(episode.outcome.evidence, stats)

        return episode, stats.total, truncated

    # -- public API --------------------------------------------------------

    def remember(self, episode: Episode) -> IngestResult:
        episode.validate()  # reject, don't coerce

        if not episode.client_key:
            # Derive a stable key so an agent that retries the same episode
            # without supplying one still deduplicates.
            digest = hashlib.blake2b(
                episode.to_json().encode("utf-8"), digest_size=16
            ).hexdigest()
            episode.client_key = f"auto:{digest}"

        episode, n_redactions, n_truncated = self._sanitize(episode)
        if not episode.episode_id:
            episode.episode_id = new_id("ep")

        # Counters are unsampled: they are bumped for every episode, including
        # ones the buffer drops (HLD 7).
        status = episode.outcome.status if episode.outcome else "unreported"
        self.store.bump_counter(
            episode.scope_id,
            episode.task.type or "unknown",
            status,
            episode.duration_ms,
            len(episode.steps),
        )

        if not self._keep_under_backpressure(episode):
            return IngestResult(
                episode_id=episode.episode_id,
                accepted=False,
                sampled_out=True,
                redactions=n_redactions,
                truncated_fields=n_truncated,
            )

        buffered_id, was_new = self.store.buffer_episode(
            episode.scope_id,
            episode.to_json(),
            client_key=episode.client_key,
            task_type=episode.task.type or "unknown",
        )

        # An episode carrying an inline outcome also files an outcome report,
        # so attribution does not have to wait for consolidation.
        if was_new and episode.outcome and episode.task.recall_id:
            self.store.add_outcome(
                episode.scope_id,
                episode.outcome.status,
                episode.outcome.signal_tier,
                recall_id=episode.task.recall_id,
                episode_id=buffered_id,
                idem_key=f"{episode.client_key}:inline",
                detail=episode.outcome.detail or episode.outcome.evidence,
            )

        return IngestResult(
            episode_id=buffered_id,
            accepted=True,
            deduplicated=not was_new,
            redactions=n_redactions,
            truncated_fields=n_truncated,
        )

    def report_outcome(
        self,
        scope_id: str,
        recall_id: str,
        outcome: str,
        signal_tier: int,
        *,
        evidence: str = "",
        idem_key: str | None = None,
    ) -> tuple[str, bool]:
        if outcome not in ("success", "failure", "partial", "abandoned"):
            raise ValidationError(f"unknown outcome {outcome!r}")
        if signal_tier not in (1, 2, 3, 4):
            raise ValidationError(f"signal_tier must be 1-4, got {signal_tier}")
        return self.store.add_outcome(
            scope_id,
            outcome,
            signal_tier,
            recall_id=recall_id,
            idem_key=idem_key,
            detail=redact(evidence) if self.config.redact else evidence,
        )
