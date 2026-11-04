"""Fixed observed focal-school anchor; only auxiliary schools may change."""
from dataclasses import dataclass

import numpy as np

from .core import DefinitionError, digest


PROTOCOL = "fixed_observed_anchor_v1"


@dataclass
class FocalAnchor:
    model: object
    data: dict
    unit_id: str

    def __post_init__(self):
        if self.model.name != "eight_schools" or self.data["unit_ids"].count(self.unit_id) != 1:
            raise DefinitionError("Focal protocol requires one original Eight Schools identity")
        self.index = self.data["unit_ids"].index(self.unit_id)
        self.anchor = self.model.subset(self.data, [self.index])
        self.pool = np.array([i for i in range(len(self.data["y"])) if i != self.index], dtype=int)

    def attach(self, auxiliary):
        result = self.model.append(self.anchor, auxiliary)
        self.check(result)
        return result

    def check(self, candidate):
        if candidate["unit_ids"].count(self.unit_id) != 1:
            raise DefinitionError("Focal observation must occur exactly once")
        j = candidate["unit_ids"].index(self.unit_id)
        if any(candidate[k][j] != self.anchor[k][0] for k in ("y", "s")):
            raise DefinitionError("Focal y/s must remain fixed within the experiment")
        if candidate.get("original_unit_ids") != self.data.get("original_unit_ids"):
            raise DefinitionError("Original school-index mapping changed")

    def audit(self):
        return {"protocol": PROTOCOL, "focal_unit_id": self.unit_id,
                "anchor_y": self.anchor["y"][0], "anchor_s": self.anchor["s"][0],
                "anchor_hash": digest(self.anchor), "observed_auxiliary_count": len(self.pool),
                "unit": "auxiliary_school", "total_size_rule": "1 + auxiliary_count"}


def validate_focal(model, data, target, method):
    if not target.focal_unit_id:
        return None
    if (target.dimension != 1 or not target.coordinates[0].variable.startswith("theta[")
            or method.get("focal_protocol", PROTOCOL) != PROTOCOL):
        raise DefinitionError("Use a theta[j] marginal and fixed_observed_anchor_v1")
    model.validate_target(target, data)
    return FocalAnchor(model, data, target.focal_unit_id)
