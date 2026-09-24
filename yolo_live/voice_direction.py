"""Persistent appearance-based target selection and camera alignment."""

from dataclasses import dataclass, field
import math

from .direction import AlignmentController, overlap, position

import logging

LOG = logging.getLogger("yolo_live.tracking")


def unit_vector(values):
    if values is None:
        return None

    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None

    if not vector or not all(math.isfinite(value) for value in vector):
        return None

    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-9:
        return None

    return tuple(value / length for value in vector)


def similarity(a, b):
    if a is None or b is None or len(a) != len(b):
        return -1.0
    return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def nearby(a, b):
    if a is None or b is None:
        return False

    distance = math.hypot(
        (a[0] + a[2] - b[0] - b[2]) / 2,
        (a[1] + a[3] - b[1] - b[3]) / 2,
    )
    return overlap(a, b) > 0.05 or distance < 0.20


@dataclass
class TargetController(AlignmentController):
    horizontal: str | None = None

    # Initial values: calibrate with your actual camera and objects.
    tracking_threshold: float = 0.60
    return_threshold: float = 0.60
    minimum_margin: float = 0.06
    anchor_floor: float = 0.55
    confirmation_frames: int = 3
    confirmation_gap: float = 2.0

    reference: tuple | None = None
    views: list = field(default_factory=list)
    missing: bool = True

    pending_vector: tuple | None = None
    pending_box: tuple | None = None
    pending_count: int = 0
    pending_time: float | None = None

    stable_frames: int = 0

    last_diag_key: str | None = None
    last_diag_at: float = float("-inf")

    def log_diagnostic(self, now, key, message, *args):
        # Avoid printing the same issue for every processed frame.
        if key != self.last_diag_key or now - self.last_diag_at >= 1.0:
            LOG.info(message, *args)
            self.last_diag_key = key
            self.last_diag_at = now

    def clear_pending(self):
        self.pending_vector = None
        self.pending_box = None
        self.pending_count = 0
        self.pending_time = None

    def waiting(self, matches, text=None, state="searching"):
        result = self._lost()
        result.update(
            text=text or (
                f"Looking for your {self.target}…"
                if self.reference is not None
                else f"Looking for a {self.target}…"
            ),
            matches=matches,
            tracking_state=state,
            has_reference=self.reference is not None,
        )
        return result

    def uncertain(self, matches, text=None):
        self.missing = True
        self.stable_frames = 0
        self.clear_pending()
        return self.waiting(matches, text)

    def confirm_candidate(self, vector, box, now):
        consistent = (
            self.pending_time is not None
            and now - self.pending_time <= self.confirmation_gap
            and similarity(vector, self.pending_vector) >= 0.80
            and nearby(box, self.pending_box)
        )

        self.pending_count = self.pending_count + 1 if consistent else 1
        self.pending_vector = vector
        self.pending_box = tuple(box)
        self.pending_time = now

        return self.pending_count >= self.confirmation_frames

    def accept(self, index, detection, vector, now, matches, score):
        # Identity selection is already done above. Reuse the original
        # controller only for smoothing and camera-centering instructions.
        self.previous_box = None
        result = super().update([detection], now)
        result["target_index"] = index
        result["matches"] = matches
        result["tracking_state"] = "tracking"
        result["has_reference"] = True
        result["appearance_similarity"] = round(score, 4)

        self.missing = False
        self.stable_frames += 1
        self.clear_pending()

        # Keep the original reference permanently for this target session.
        # Add only strongly supported views to limit gradual identity drift.
        if (
            self.stable_frames >= 5
            and similarity(vector, self.reference) >= 0.80
            and score >= 0.92
            and max(similarity(vector, view) for view in self.views) < 0.98
        ):
            self.views.append(vector)
            if len(self.views) > 6:
                del self.views[1]  # Preserve the original view at index zero.

        return result

    def update(self, detections, now, appearances=None):
        appearances = appearances or {}

        candidates = [
            (index, detection)
            for index, detection in enumerate(detections)
            if detection["label"] == self.target
        ]
        matches = len(candidates)

        # Initial selection: do not arbitrarily choose among several objects.
        if self.reference is None:
            eligible = candidates

            if self.horizontal is not None:
                eligible = [
                    (index, detection)
                    for index, detection in eligible
                    if position(detection["box"]).split("-")[-1]
                    == self.horizontal
                ]

            if not eligible:
                self.clear_pending()
                return self.waiting(matches)

            if len(eligible) != 1:
                self.clear_pending()
                return self.waiting(
                    matches,
                    f"Multiple {self.target} objects. Specify a side.",
                    state="ambiguous",
                )

            index, detection = eligible[0]
            vector = unit_vector(appearances.get(index))
            if vector is None:
                self.clear_pending()
                return self.waiting(
                    matches,
                    f"Waiting for a clearer view of the {self.target}…",
                )

            if not self.confirm_candidate(vector, detection["box"], now):
                return self.waiting(
                    matches,
                    f"Confirming the {self.target}…",
                    state="confirming",
                )

            self.reference = vector
            self.views = [vector]
            return self.accept(
                index, detection, vector, now, matches, 1.0,
            )

        # No timeout clears the visual reference.
        if not candidates:
            self.log_diagnostic(
                now,
                "no_candidate",
                "Tracking: no detection with target label %r; reference=%s",
                self.target,
                self.reference is not None,
            )
            return self.uncertain(matches)

        scored = []
        for index, detection in candidates:
            vector = unit_vector(appearances.get(index))
            if vector is None:
                continue

            score = max(similarity(vector, view) for view in self.views)
            anchor_score = similarity(vector, self.reference)
            scored.append((score, anchor_score, index, detection, vector))

        if not scored:
            return self.uncertain(matches)

        scored.sort(key=lambda item: item[0], reverse=True)
        score, anchor_score, index, detection, vector = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        margin = score - second_score

        continuous = (
            not self.missing
            and self.last_seen is not None
            and now - self.last_seen <= self.forget_after
            and nearby(detection["box"], self.previous_box)
        )

        threshold = (
            self.tracking_threshold if continuous else self.return_threshold
        )

        failures = []
        if score < threshold:
            failures.append("similarity below threshold")
        if anchor_score < self.anchor_floor:
            failures.append("anchor similarity too low")
        if margin < self.minimum_margin:
            failures.append("competing detections too similar")

        if failures:
            self.log_diagnostic(
                now,
                "rejected:" + ",".join(failures),
                "Tracking rejected %s: confidence=%.3f similarity=%.3f "
                "required=%.3f anchor=%.3f required_anchor=%.3f "
                "margin=%.3f required_margin=%.3f continuous=%s",
                detection["label"],
                detection["confidence"],
                score,
                threshold,
                anchor_score,
                self.anchor_floor,
                margin,
                self.minimum_margin,
                continuous,
            )
            return self.uncertain(matches)

        if not continuous:
            self.missing = True
            self.stable_frames = 0

            if not self.confirm_candidate(vector, detection["box"], now):
                return self.waiting(
                    matches,
                    f"Checking a possible match for your {self.target}…",
                    state="confirming",
                )

        return self.accept(
            index, detection, vector, now, matches, score,
        )