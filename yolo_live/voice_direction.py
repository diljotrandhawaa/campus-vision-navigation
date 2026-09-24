"""Conservative target selection around the existing alignment controller."""

from dataclasses import dataclass
import math

from .direction import AlignmentController, overlap, position


@dataclass
class TargetController(AlignmentController):
    horizontal: str | None = None
    blocked: bool = False

    def unavailable(self, text, matches=0, ambiguous=False):
        result = self._lost()
        result.update(
            state="ambiguous" if ambiguous else "lost",
            text=text,
            matches=matches,
        )
        return result

    def lose_lock(self):
        self.blocked = True
        return self.unavailable(
            f"{self.target.capitalize()} lost. Reacquire the target."
        )

    def update(self, detections, now):
        if self.blocked:
            return self.lose_lock()

        candidates = [
            (index, detection)
            for index, detection in enumerate(detections)
            if detection["label"] == self.target
        ]
        matches = len(candidates)

        if self.previous_box is None:
            if self.horizontal is not None:
                candidates = [
                    (index, detection)
                    for index, detection in candidates
                    if position(detection["box"]).split("-")[-1] == self.horizontal
                ]

            if not candidates:
                side = f" on the {self.horizontal}" if self.horizontal else ""
                return self.unavailable(
                    f"{self.target.capitalize()} not visible{side}", matches,
                )

            if len(candidates) != 1:
                return self.unavailable(
                    f"Multiple {self.target} objects. Specify a side.",
                    matches,
                    ambiguous=True,
                )

            index, chosen = candidates[0]
            result = super().update([chosen], now)
            result["target_index"] = index
            result["matches"] = matches
            return result

        if self.last_seen is None or now - self.last_seen > self.forget_after:
            return self.lose_lock()

        old = self.previous_box
        old_x = (old[0] + old[2]) / 2
        old_y = (old[1] + old[3]) / 2
        nearby = []

        for index, detection in candidates:
            box = detection["box"]
            distance = math.hypot(
                (box[0] + box[2]) / 2 - old_x,
                (box[1] + box[3]) / 2 - old_y,
            )
            if overlap(old, box) > 0.05 or distance < 0.18:
                nearby.append(index)

        if len(nearby) != 1:
            return self.lose_lock()

        # Once acquired, do not reapply the side filter as the camera turns.
        return super().update(detections, now)