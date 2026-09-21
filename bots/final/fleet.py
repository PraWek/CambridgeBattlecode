"""Track core-spawned workers without counting turrets as economic builders."""


def unit_is_alive(controller, entity_id: int) -> bool:
    """Probe one known ID; fog of war is not evidence of a casualty.

    The engine checks existence before vision: dead IDs raise ``Unknown id``;
    a living unit outside vision raises ``Position out of vision range``.
    Keep unknown error variants conservative rather than replacing a worker
    merely because its health could not be read.
    """
    try:
        return controller.get_hp(entity_id) > 0
    except Exception as error:
        return str(error) != "Unknown id"
