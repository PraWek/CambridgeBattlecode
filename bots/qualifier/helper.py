"""
HELPER USED FOR CODE GEN

not meant to be used internally or for any logic
"""

from cambc import EntityType, Environment

from botlib.constants import (
    HASHMAP_ENTITY_TYPE,
    HASHMAP_ENV,
    ENTITY_TYPE_CACHE,
)


def format_bits(n: int, width: int = None, group: int = 4) -> str:
    # Determine width automatically if not provided
    if width is None:
        width = max(1, n.bit_length())

    # Round width up to a multiple of group size
    if width % group != 0:
        width += group - (width % group)

    # Format as zero-padded binary
    bits = format(n, f"0{width}b")

    # Insert underscores every `group` bits
    grouped = "_".join(bits[i : i + group] for i in range(0, len(bits), group))

    return f"0b{grouped}"


def typelist_to_bits(types: list[EntityType]):
    bits = 0
    for type in types:
        bits |= 1 << HASHMAP_ENTITY_TYPE[type]
    return bits


def typelist_to_env_bits(types: list[Environment]):
    bits = 0
    for type in types:
        bits |= 1 << HASHMAP_ENV[type]
    return bits


def debug_heap(minheap):
    for i in range(minheap.heap_size):
        print(minheap.heap[i])
    print("-")


def bit_pattern_to_type_list(bits: int):
    types = []
    for i in range(bits.bit_length()):
        bit = bits & (0b1 << i)
        if bit:
            types.append(ENTITY_TYPE_CACHE[i].value)
    return types


print(bit_pattern_to_type_list(0b0001_0111_1000_0100))

print(
    format_bits(
        typelist_to_bits(
            [
                # EntityType.BUILDER_BOT,  # 1
                EntityType.CORE,  # 2
                # EntityType.GUNNER,  # 3
                # EntityType.SENTINEL,  # 4
                # EntityType.BREACH,  # 5
                # EntityType.LAUNCHER,  # 6
                EntityType.CONVEYOR,  # 7
                EntityType.SPLITTER,  # 8
                EntityType.ARMOURED_CONVEYOR,  # 9
                EntityType.BRIDGE,  # 10
                # EntityType.HARVESTER,  # 11
                EntityType.FOUNDRY,  # 12
                # EntityType.ROAD,  # 13
                # EntityType.BARRIER,  # 14
                # EntityType.MARKER,  # 15
            ]
        )
    )
)

CONVEYOR_PATTERN = 0b1

# print(format_bits(typelist_to_env_bits([Environment.ORE_AXIONITE, Environment.ORE_TITANIUM])))

# Precompute masks to prevent wrapping
# text = "[\n"
# for w in range(20, 51, 1):
#     text += "[\n"
#     for h in range(20, 51, 1):
#         not_left_edge = 0
#         for y in range(h):
#             # A '1' at every position except the first column of every row
#             row_mask = ((1 << w) - 1) << (y * w)
#             first_col_bit = 1 << (y * w)
#             first_col_bit |= 1 << (y * w + 1)
#             not_left_edge |= row_mask ^ first_col_bit

#         not_right_edge = 0
#         for y in range(h):
#             # A '1' at every position except the last column of every row
#             row_mask = ((1 << w) - 1) << (y * w)
#             last_col_bit = 1 << (y * w + w - 1)
#             last_col_bit |= 1 << (y * w + w - 2)
#             not_right_edge |= row_mask ^ last_col_bit

#         text += f"(\n0x{not_left_edge:x},\n0x{not_right_edge:x}\n),\n"
#     text += "],\n"
# text += "\n]"

# with open("./bitboard_edges.txt", "w") as f:
#     f.write(text)


def generate_ordered_deltas(r):
    """
    Generates a tuple of (dx, dy) deltas ordered from the
    innermost ring (r=1) to the outermost ring (r=r).
    """
    ordered_deltas = []

    # Loop through each ring level starting from 1
    for current_r in range(1, r + 1):
        ring = []
        # Check the square boundary defined by current_r
        for dx in range(-current_r, current_r + 1):
            for dy in range(-current_r, current_r + 1):
                # A point is on the "crust" of the ring if at least
                # one of its coordinates is equal to the current radius
                if max(abs(dx), abs(dy)) == current_r:
                    ring.append((dx, dy))

        # Optional: Sort the individual ring (e.g., clockwise or by angle)
        # ring.sort(key=lambda p: ...)

        ordered_deltas.extend(ring)

    return tuple(ordered_deltas)


# Example usage:
r_max = 4
result = generate_ordered_deltas(r_max)

print(f"Ordered Deltas up to r={r_max}:")
print(result)
