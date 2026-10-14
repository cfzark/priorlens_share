"""Pathwise selectors, with unmatched distinct from numerical failure."""
from dataclasses import dataclass
import numpy as np
from .core import DefinitionError, NumericalFailure


@dataclass(frozen=True)
class BidirectionalSelector:
    tie: str = "positive"
    delta: float | None = None
    secondary: str = "gap"

    def __call__(self, rows):
        if self.tie not in {"positive", "negative"}:
            raise DefinitionError("Tie rule must be positive or negative")
        if not rows or any(not np.isfinite(r["loss"]) for r in rows):
            raise NumericalFailure("Incomplete/nonfinite directional objective array")
        if self.delta is not None and not 0 <= self.delta <= 1:
            raise DefinitionError("delta must lie in [0,1]")
        candidates = rows if self.delta is None else [r for r in rows if r["adequacy"] <= self.delta]
        if not candidates:
            return {"status": "unmatched", "signed_m": None, "selected_direction": None, "boundary_selected": False}
        if self.secondary not in {"gap", "min_m"}:
            raise DefinitionError("Adequacy secondary rule must be gap or min_m")
        def key(r):
            objective = r["loss"] if self.delta is None else r["gap"] if self.secondary == "gap" else r["m"]
            return objective, 0 if r["direction"] == self.tie else 1, r["m"]
        best = min(candidates, key=key)
        return {"status": "matched", "signed_m": best["m"] * (1 if best["direction"] == "positive" else -1),
                "selected_direction": best["direction"], "selected_loss": best["loss"],
                "selected_adequacy": best.get("adequacy"), "boundary_selected": best["m"] == max(r["m"] for r in rows)}
