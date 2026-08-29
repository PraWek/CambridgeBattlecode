def desired_builder_count(
        current_round: int,
        initial_builders: int,
        maximum_builders: int,
        expansion_start_round: int,
        expansion_interval_rounds: int,
) -> int:
    """Return the builder budget without coupling it to future combat roles.

    The first wave gives one scout to each sector.  Later builders are added
    gradually, so they can reuse discovered roads and construct independent
    conveyor trunks without multiplying the expensive opening scan.
    """
    if current_round < expansion_start_round:
        return initial_builders
    additions = 1 + (
        current_round - expansion_start_round
    ) // expansion_interval_rounds
    return min(maximum_builders, initial_builders + additions)
