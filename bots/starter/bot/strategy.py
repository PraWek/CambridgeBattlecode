from __future__ import annotations

from cambc import Controller

from bot.constants import (
    PHASE_BOOTSTRAP,
    PHASE_EXPAND_AXIONITE,
    PHASE_EXPAND_TITANIUM,
    PHASE_STABILIZE,
    TITANIUM_LINE_READY_HARVESTERS,
    TITANIUM_LINE_READY_SCALE,
)


def is_titanium_line_ready(ct: Controller, titanium_harvesters: int) -> bool:
    return titanium_harvesters >= TITANIUM_LINE_READY_HARVESTERS or (
        titanium_harvesters >= 1 and ct.get_scale_percent() >= TITANIUM_LINE_READY_SCALE
    )


def choose_phase(ct: Controller, titanium_harvesters: int, axionite_harvesters: int) -> int:
    if titanium_harvesters == 0:
        return PHASE_BOOTSTRAP
    if not is_titanium_line_ready(ct, titanium_harvesters):
        return PHASE_EXPAND_TITANIUM
    if axionite_harvesters == 0:
        return PHASE_EXPAND_AXIONITE
    return PHASE_STABILIZE
