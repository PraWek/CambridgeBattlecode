from botlib import BitBoardBFS

from botlib import (
    BITBOARD_GRID,
    MAP_INFO,
    BITBOARD_TARGETS,
    BITBOARD_HAZARDS,
    POSITION_CACHE,
    UNIT_INFO,
    best_enemy_core_idx,
)

from botlib.constants import (
    BITBOARD_EDGES,
    BITBOARD_EDGES_2,
    DIRECTION_DELTAS,
    CARDINAL_DIRECTION_DELTAS,
    BRIDGE_DIRECTION_DELTAS,
)


class BotBitboardSolver:
    MAX_COST = 20
    NUM_FRONTIERS = 1

    __slots__ = (
        "bots",
        "buildings",
        "pathable",
        "default",
        "gunner",
        "sentinel",
        "fallback_move",
        "treat_buildings_as_wall",
    )

    def __init__(self):
        self.bots = 0
        self.buildings = 0
        self.gunner = 0
        self.sentinel = 0
        self.pathable = 0
        self.default = 0

        # Signals if a given move from `get_next_tile` was a fallback or not
        self.fallback_move = False
        self.treat_buildings_as_wall = False

    def update(self, bitboard: BitBoardBFS):
        still_bots = (BITBOARD_GRID.prev_enemy_bots & BITBOARD_GRID.enemy[1]) | (
            BITBOARD_GRID.prev_ally_bots & BITBOARD_GRID.ally[1]
        )
        bots = (
            BITBOARD_GRID.prev_enemy_bots
            | BITBOARD_GRID.prev_ally_bots
            | BITBOARD_GRID.ally[1]
            | BITBOARD_GRID.enemy[1]
        ) & (~still_bots)

        enemy_core = BITBOARD_TARGETS[best_enemy_core_idx()]
        enemy_core |= (enemy_core << 1) | (enemy_core >> 1)
        enemy_core |= (enemy_core << MAP_INFO.width) | (enemy_core >> MAP_INFO.width)

        buildings = (
            BITBOARD_GRID.ally[3]
            | BITBOARD_GRID.ally[4]
            | BITBOARD_GRID.ally[5]
            | BITBOARD_GRID.ally[6]
            | BITBOARD_GRID.ally[11]
            | BITBOARD_GRID.ally[12]
            | BITBOARD_GRID.ally[14]
            | BITBOARD_GRID.enemy[2]
            | BITBOARD_GRID.enemy[3]
            | BITBOARD_GRID.enemy[4]
            | BITBOARD_GRID.enemy[5]
            | BITBOARD_GRID.enemy[6]
            | BITBOARD_GRID.enemy[11]
            | BITBOARD_GRID.enemy[12]
            | BITBOARD_GRID.enemy[14]
            | enemy_core
        )

        bitboard.walls |= (
            BITBOARD_GRID.env[1]
            | BITBOARD_GRID.env_mirrors[0][1]
            | still_bots
            | BITBOARD_HAZARDS[6]
        )

        if self.treat_buildings_as_wall:
            bitboard.walls |= buildings

        self.bots = bots
        self.gunner = BITBOARD_HAZARDS[3]
        self.sentinel = BITBOARD_HAZARDS[4]
        self.buildings = buildings
        meta = (
            self.bots | self.buildings | self.gunner | self.sentinel | still_bots | bitboard.walls
        )
        self.pathable = (
            BITBOARD_GRID.ally[2]
            | BITBOARD_GRID.ally[7]
            | BITBOARD_GRID.ally[8]
            | BITBOARD_GRID.ally[9]
            | BITBOARD_GRID.ally[10]
            | BITBOARD_GRID.ally[13]
            | BITBOARD_GRID.ally[15]
            | BITBOARD_GRID.enemy[7]
            | BITBOARD_GRID.enemy[8]
            | BITBOARD_GRID.enemy[9]
            | BITBOARD_GRID.enemy[10]
            | BITBOARD_GRID.enemy[13]
            | BITBOARD_GRID.enemy[15]
        ) & (~meta)
        self.default = ~(meta | self.pathable)

    def solve(
        self,
        bitboard: BitBoardBFS,
        root_idx: int,
        stop_mask: int,
        stop_mask_any: int,
        stop_cost: int | None,
        stop_idx: int | None,
        inflate: int,
        time_limit: int,
        max_depth: int,
    ):
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        get_cpu_time_elapsed = bitboard.get_cpu_time_elapsed

        not_left_edge, not_right_edge = BITBOARD_EDGES[map_width - 20][map_height - 20]

        frontier_archive = bitboard._frontier_archives[0]
        archive = bitboard._archive
        scheduler = bitboard.schedulers[0]
        full_mask = bitboard.full_mask

        modulo = bitboard.max_cost - 1

        stop_mask |= 0 if stop_idx is None else BITBOARD_TARGETS[stop_idx]
        stop_mask_any &= ~bitboard.walls
        root_bit = BITBOARD_TARGETS[root_idx]
        visited = bitboard.walls | root_bit

        frontier = root_bit
        frontier_archive[0] = frontier
        archive[0] = visited
        bitboard._archive_size = 1
        t = 1

        depth = 0
        start = get_cpu_time_elapsed()
        while True:
            horiz = (
                frontier | ((frontier << 1) & not_left_edge) | ((frontier >> 1) & not_right_edge)
            )
            expanded_frontier = horiz | (horiz >> map_width) | ((horiz << map_width) & full_mask)

            # pathable
            next_frontier = (
                (expanded_frontier & self.pathable) | scheduler[(idx := (t % modulo))]
            ) & ~visited

            scheduler[idx] = 0  # NOTE Must clear here for future use below

            # default
            scheduler[(t + 1) % modulo] |= expanded_frontier & self.default
            # sentinel
            scheduler[(t + 4) % modulo] |= expanded_frontier & self.sentinel
            # gunner
            scheduler[(t + 6) % modulo] |= expanded_frontier & self.gunner
            # bots
            scheduler[(t + 10) % modulo] |= expanded_frontier & self.bots
            # buildings
            scheduler[idx] |= expanded_frontier & self.buildings

            if not next_frontier and not any(scheduler):
                break

            visited |= next_frontier
            if len(archive) == t:
                archive.append(visited)
            else:
                archive[t] = visited
            if len(frontier_archive) == t:
                frontier_archive.append(next_frontier)
            else:
                frontier_archive[t] = next_frontier
            t += 1

            should_stop = False
            if stop_cost is not None and t > stop_cost:
                should_stop = True
            if visited & stop_mask_any:
                should_stop = True
            if stop_mask != 0 and (visited & stop_mask) == stop_mask:
                should_stop = True

            if should_stop:
                inflate -= 1
                if inflate < 0:
                    break

            frontier = next_frontier

            depth += 1
            if get_cpu_time_elapsed() - start > time_limit or (max_depth > 0 and depth > max_depth):
                yield None

                depth = 0
                start = get_cpu_time_elapsed()

        bitboard._archive_size = t
        yield visited & (~bitboard.walls)

    def get_next_tile(self, bitboard: BitBoardBFS, tile_idx: int, t: int, use_fallback=True):
        x, y = POSITION_CACHE[tile_idx]
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        frontier_archive = bitboard.frontier_archives[0]
        invalid_tiles = bitboard.walls | self.bots | self.buildings
        valid_tiles = ~invalid_tiles

        # NOTE order matters - you preference the tiles further back in the timeline, so thats why we check
        #      in reverse order since those that travel further back the timeline, reach the goal faster
        for i in [20, 11, 7, 5, 2, 1]:
            if t < i:
                continue

            prev = frontier_archive[t - i] & valid_tiles
            for dx, dy in DIRECTION_DELTAS:
                neighbour_x = x + dx
                neighbour_y = y + dy

                if (
                    neighbour_x < 0
                    or neighbour_y < 0
                    or neighbour_x >= map_width
                    or neighbour_y >= map_height
                ):
                    continue

                if BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x] & prev:
                    self.fallback_move = False
                    return neighbour_idx

        if not use_fallback:
            return None

        # If no valid next tile is found in the archive, look for a valid tile that progresses you to the goal
        print("pathing fallback")
        target_idx = -1
        min_dist = 0

        for dx, dy in DIRECTION_DELTAS:
            neighbour_x = x + dx
            neighbour_y = y + dy

            if (
                neighbour_x < 0
                or neighbour_y < 0
                or neighbour_x >= map_width
                or neighbour_y >= map_height
            ):
                continue

            if (
                BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x]
                & invalid_tiles
            ):
                continue

            neighbour_t = bitboard.dist(neighbour_idx)

            if neighbour_t is not None and (target_idx == -1 or neighbour_t < min_dist):
                target_idx = neighbour_idx
                min_dist = neighbour_t

        self.fallback_move = True
        return target_idx if target_idx != -1 else None

    def get_prev_tile(self, bitboard: BitBoardBFS, tile_idx: int, t: int):
        x, y = POSITION_CACHE[tile_idx]
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        archive_size = bitboard.archive_size
        frontier_archive = bitboard.frontier_archives[0]
        invalid_tiles = bitboard.walls | self.bots | self.buildings
        valid_tiles = ~invalid_tiles

        # NOTE order matters - you preference the tiles further back in the timeline, so thats why we check
        #      in order since those that travel the least into the future, move away from the goal faster
        for i in [1, 3, 11, 20]:
            if t + i >= archive_size:
                continue

            prev = frontier_archive[t + i] & valid_tiles
            for dx, dy in DIRECTION_DELTAS:
                neighbour_x = x + dx
                neighbour_y = y + dy

                if (
                    neighbour_x < 0
                    or neighbour_y < 0
                    or neighbour_x >= map_width
                    or neighbour_y >= map_height
                ):
                    continue

                if BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x] & prev:
                    self.fallback_move = False
                    return neighbour_idx

        # If no valid next tile is found in the archive, look for a valid tile that progresses you to the goal
        print("pathing fallback")
        target_idx = -1
        min_dist = 0

        for dx, dy in DIRECTION_DELTAS:
            neighbour_x = x + dx
            neighbour_y = y + dy

            if (
                neighbour_x < 0
                or neighbour_y < 0
                or neighbour_x >= map_width
                or neighbour_y >= map_height
            ):
                continue

            if (
                BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x]
                & invalid_tiles
            ):
                continue

            neighbour_t = bitboard.dist(neighbour_idx)

            if neighbour_t is not None and (target_idx == -1 or neighbour_t < min_dist):
                target_idx = neighbour_idx
                min_dist = neighbour_t

        self.fallback_move = True
        return target_idx if target_idx != -1 else None


class ResourceBitboardSolver:
    MAX_COST = 40
    NUM_FRONTIERS = 1

    __slots__ = (
        "allow_merging",
        "check_flow",
        "bots_ore",
        "merge_lines",
        "treat_existing_lines_as_wall",
        "treat_ally_core_as_conveyor",
        "inflate_titanium_harvesters",
        "avoid_lines",
        "default",
        "placed_lines",
        "source_mask",
        "hazards",
        "avoid_enemy_lines",
    )

    def __init__(self):
        # User can specify whether lines merge or not
        self.allow_merging = True
        # When allow_merging is False, this option toggles
        # true line avoidance because normally it just makes lines have high cost
        # to prevent unreachable
        self.treat_existing_lines_as_wall = False
        # User can specify whether lines check flow
        self.check_flow = True
        # Treat the ally core as a conveyor or as a building
        self.treat_ally_core_as_conveyor = True
        # User can specify what tiles we have placed so we do not merge resource lines with themselves
        self.placed_lines = 0
        # Inflate titanium harvesters
        self.inflate_titanium_harvesters = False
        self.avoid_enemy_lines = False

        self.bots_ore = 0
        self.merge_lines = 0
        self.avoid_lines = 0
        self.default = 0
        self.source_mask = 0

    def mark_placed_lines(self, tile_idx: int):
        self.placed_lines |= BITBOARD_TARGETS[tile_idx]
        return self.placed_lines

    def update(self, bitboard: BitBoardBFS):
        self.source_mask = ~(
            BITBOARD_GRID.ally[11]
            | BITBOARD_GRID.ally[12]
            | BITBOARD_GRID.enemy[11]
            | BITBOARD_GRID.enemy[12]
        )

        enemy_core = BITBOARD_TARGETS[best_enemy_core_idx()]
        enemy_core |= (enemy_core << 1) | (enemy_core >> 1)
        enemy_core |= (enemy_core << MAP_INFO.width) | (enemy_core >> MAP_INFO.width)

        buildings = (
            BITBOARD_GRID.ally[3]
            | BITBOARD_GRID.ally[4]
            | BITBOARD_GRID.ally[5]
            | BITBOARD_GRID.ally[6]
            | BITBOARD_GRID.ally[11]
            | BITBOARD_GRID.ally[12]
            | BITBOARD_GRID.ally[14]
            | BITBOARD_GRID.enemy[2]
            | BITBOARD_GRID.enemy[3]
            | BITBOARD_GRID.enemy[4]
            | BITBOARD_GRID.enemy[5]
            | BITBOARD_GRID.enemy[6]
            | BITBOARD_GRID.enemy[11]
            | BITBOARD_GRID.enemy[12]
            | BITBOARD_GRID.enemy[14]
            | enemy_core
        )
        if not self.treat_ally_core_as_conveyor:
            buildings |= BITBOARD_GRID.ally[2]
        if self.inflate_titanium_harvesters:
            map_width = MAP_INFO.width
            map_height = MAP_INFO.height
            not_left_edge, not_right_edge = BITBOARD_EDGES[map_width - 20][map_height - 20]
            # titanium_harvesters = (
            #     BITBOARD_GRID.ally[11] | BITBOARD_GRID.enemy[11]
            # ) & BITBOARD_GRID.env[2]
            titanium_harvesters = BITBOARD_GRID.env[
                2
            ]  # TODO verify if we should inflate just titanium or ores
            buildings |= (
                titanium_harvesters
                | ((titanium_harvesters << 1) & not_left_edge)
                | ((titanium_harvesters >> 1) & not_right_edge)
                | (titanium_harvesters >> map_width)
                | ((titanium_harvesters << map_width) & bitboard.full_mask)
            )

        bitboard.walls |= (
            BITBOARD_GRID.env[1]
            | BITBOARD_GRID.env_mirrors[0][1]
            | buildings
            | (BITBOARD_HAZARDS[3] | BITBOARD_HAZARDS[4] | BITBOARD_HAZARDS[5])
        )

        bots = BITBOARD_GRID.ally[1] | BITBOARD_GRID.enemy[1]
        ore = (
            BITBOARD_GRID.env[2]
            | BITBOARD_GRID.env[3]
            | BITBOARD_GRID.env_mirrors[0][2]
            | BITBOARD_GRID.env_mirrors[0][3]
        )
        self.bots_ore = (bots | ore) & (~bitboard.reachable)
        ally_lines = (
            BITBOARD_GRID.ally[7]
            | BITBOARD_GRID.ally[8]
            | BITBOARD_GRID.ally[9]
            | BITBOARD_GRID.ally[10]
        )
        enemy_lines = 0
        if self.avoid_enemy_lines:
            enemy_lines = (
                BITBOARD_GRID.enemy[7]
                | BITBOARD_GRID.enemy[8]
                | BITBOARD_GRID.enemy[9]
                | BITBOARD_GRID.enemy[10]
            )
        else:
            ally_lines |= (
                BITBOARD_GRID.enemy[7]
                | BITBOARD_GRID.enemy[8]
                | BITBOARD_GRID.enemy[9]
                | BITBOARD_GRID.enemy[10]
            )
        if self.treat_existing_lines_as_wall:
            bitboard.walls |= ally_lines

        lines_with_high_flow = ally_lines & (BITBOARD_GRID.flow[0][4] | BITBOARD_GRID.flow[2][4])
        axionite_lines = ally_lines & (
            # Currently ignore all axionite flow / stall
            BITBOARD_GRID.flow[1][0]
            | BITBOARD_GRID.flow[1][1]
            | BITBOARD_GRID.flow[1][2]
            | BITBOARD_GRID.flow[1][3]
            | BITBOARD_GRID.flow[1][4]
            | BITBOARD_GRID.stall[1][0]
            | BITBOARD_GRID.stall[1][1]
            | BITBOARD_GRID.stall[1][2]
            | BITBOARD_GRID.stall[1][3]
            | BITBOARD_GRID.stall[1][4]
        )
        if self.check_flow:
            stalled_lines = ally_lines & (
                BITBOARD_GRID.stall[0][1]
                | BITBOARD_GRID.stall[0][2]
                | BITBOARD_GRID.stall[0][3]
                | BITBOARD_GRID.stall[0][4]
                | BITBOARD_GRID.stall[2][1]
                | BITBOARD_GRID.stall[2][2]
                | BITBOARD_GRID.stall[2][3]
                | BITBOARD_GRID.stall[2][4]
            )
        else:
            stalled_lines = 0

        if self.allow_merging:
            meta_lines = lines_with_high_flow | stalled_lines | self.placed_lines | axionite_lines
            self.merge_lines = ally_lines & (~meta_lines)
            self.avoid_lines = enemy_lines | (ally_lines & meta_lines)
        elif not self.treat_existing_lines_as_wall:
            # we want to merge with dead lines to revive them
            very_low_flow = ally_lines & (
                BITBOARD_GRID.flow[0][0]
                & BITBOARD_GRID.flow[1][0]
                & BITBOARD_GRID.flow[2][0]
                & BITBOARD_GRID.stall[0][0]
                & BITBOARD_GRID.stall[1][0]
                & BITBOARD_GRID.stall[2][0]
            )
            self.merge_lines = very_low_flow
            self.avoid_lines = (ally_lines & (~very_low_flow)) | enemy_lines
        self.default = ~(self.bots_ore | ally_lines | enemy_lines)

    def solve(
        self,
        bitboard: BitBoardBFS,
        root_idx: int,
        stop_mask: int,
        stop_mask_any: int,
        stop_cost: int | None,
        stop_idx: int | None,
        inflate: int,
        time_limit: int,
        max_depth: int,
    ):
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        not_left_edge, not_right_edge = BITBOARD_EDGES[map_width - 20][map_height - 20]
        not_left_edge_2, not_right_edge_2 = BITBOARD_EDGES_2[map_width - 20][map_height - 20]

        get_cpu_time_elapsed = bitboard.get_cpu_time_elapsed

        frontier_archive = bitboard._frontier_archives[0]
        scheduler = bitboard.schedulers[0]
        archive = bitboard._archive
        full_mask = bitboard.full_mask

        modulo = bitboard.max_cost - 1

        stop_mask |= 0 if stop_idx is None else BITBOARD_TARGETS[stop_idx]
        stop_mask_any &= ~bitboard.walls
        root_bit = BITBOARD_TARGETS[root_idx]
        visited = bitboard.walls | root_bit

        frontier = root_bit
        frontier_archive[0] = frontier
        archive[0] = visited
        bitboard._archive_size = 1
        t = 1

        depth = 0
        start = get_cpu_time_elapsed()
        while True:
            conveyor_expansion = (
                ((frontier << 1) & not_left_edge)
                | ((frontier >> 1) & not_right_edge)
                | (frontier >> map_width)
                | ((frontier << map_width) & full_mask)
            )

            bridge_expansion = (
                ((conveyor_expansion << 1) & not_left_edge)
                | ((conveyor_expansion >> 1) & not_right_edge)
                | (conveyor_expansion << map_width)
                | (conveyor_expansion >> map_width)
            )
            bridge_expansion |= (
                ((bridge_expansion << 1) & not_left_edge)
                | ((bridge_expansion >> 1) & not_right_edge)
                | (bridge_expansion << map_width)
                | (bridge_expansion >> map_width)
            )
            vert_temp = (frontier << map_width * 2) | (frontier >> map_width * 2)
            bridge_expansion |= ((vert_temp << 2) & not_left_edge_2) | (
                (vert_temp >> 2) & not_right_edge_2
            )
            bridge_expansion &= full_mask
            # you cant bridge onto a source, prevents resource path sources using immediate bridge move
            # which isn't possible
            bridge_expansion &= self.source_mask

            next_frontier = (
                scheduler[(idx := (t % modulo))] | (conveyor_expansion & self.merge_lines)
            ) & ~visited

            # NOTE Must clear here for future use below
            scheduler[idx] = 0

            # default
            scheduler[(t + 1) % modulo] |= conveyor_expansion & self.default
            # bridge to existing lines
            scheduler[(t + 8) % modulo] |= bridge_expansion & self.merge_lines
            # bridge
            scheduler[(t + 12) % modulo] |= bridge_expansion & self.default
            # merge with avoid line
            scheduler[(t + 20) % modulo] |= conveyor_expansion & self.avoid_lines
            # bridge to avoid line
            scheduler[(t + 28) % modulo] |= bridge_expansion & self.avoid_lines
            # ore & bots
            scheduler[idx] |= (bridge_expansion | conveyor_expansion) & self.bots_ore

            if not next_frontier and not any(scheduler):
                break

            visited |= next_frontier
            if len(archive) == t:
                archive.append(0)
            archive[t] = visited
            if len(frontier_archive) == t:
                frontier_archive.append(0)
            frontier_archive[t] = next_frontier
            t += 1

            should_stop = False
            if stop_cost is not None and t > stop_cost:
                should_stop = True
            if visited & stop_mask_any:
                should_stop = True
            if stop_mask != 0 and (visited & stop_mask) == stop_mask:
                should_stop = True

            if should_stop:
                inflate -= 1
                if inflate < 0:
                    break

            frontier = next_frontier

            depth += 1
            if get_cpu_time_elapsed() - start > time_limit or (max_depth > 0 and depth > max_depth):
                yield None

                depth = 0
                start = get_cpu_time_elapsed()

        bitboard._archive_size = t
        yield visited & (~bitboard.walls)

    def get_next_tile(self, bitboard: BitBoardBFS, tile_idx: int, t: int):
        x, y = POSITION_CACHE[tile_idx]
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        frontier_archive = bitboard.frontier_archives[0]
        neg_walls = ~bitboard.walls

        # TODO return a "next best tile" when best tile is blocked due to bot / building

        # NOTE order matters - you preference the tiles further back in the timeline, so thats why we check
        #      in reverse order since those that travel further back the timeline, reach the goal faster
        # 1 = cardinal merge with free resource line
        # 2 = default cardinal move
        # 21 = cardinal merge with a stalled or overflowed resource line
        # 40 = cardinal merge with a location that has an ore or a bot on it
        for i in [40, 21, 2, 1]:
            if t < i:
                continue

            prev = frontier_archive[t - i] & neg_walls
            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                neighbour_x = x + dx
                neighbour_y = y + dy

                if (
                    neighbour_x < 0
                    or neighbour_y < 0
                    or neighbour_x >= map_width
                    or neighbour_y >= map_height
                ):
                    continue

                if BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x] & prev:
                    return neighbour_idx

        # NOTE order matters - you preference the tiles further back in the timeline, so thats why we check
        #      in reverse order since those that travel further back the timeline, reach the goal faster
        # 9 = bridge merge with free resource line
        # 13 = default bridge move
        # 29 = bridge merge with a stalled or overflowed resource line
        # 40 = bridge merge with a location that has an ore or a bot on it
        for i in [40, 29, 13, 9]:
            if t < i:
                continue

            prev = frontier_archive[t - i] & neg_walls
            for dx, dy in BRIDGE_DIRECTION_DELTAS:
                neighbour_x = x + dx
                neighbour_y = y + dy

                if (
                    neighbour_x < 0
                    or neighbour_y < 0
                    or neighbour_x >= map_width
                    or neighbour_y >= map_height
                ):
                    continue

                if BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x] & prev:
                    return neighbour_idx

        return None

    def get_prev_tile(self, bitboard: BitBoardBFS, tile_idx: int, t: int):
        x, y = POSITION_CACHE[tile_idx]
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        frontier_archive = bitboard.frontier_archives[0]
        neg_walls = ~bitboard.walls
        archive_size = bitboard.archive_size

        # TODO return a "next best tile" when best tile is blocked due to bot / building

        # NOTE order matters - you preference the tiles further back in the timeline, so thats why we check
        #      in reverse order since those that travel further back the timeline, reach the goal faster
        # 1 = cardinal merge with free resource line
        # 2 = default cardinal move
        # 21 = cardinal merge with a stalled or overflowed resource line
        # 40 = cardinal merge with a location that has an ore or a bot on it
        for i in [40, 21, 2, 1]:
            if t + i >= archive_size:
                continue

            prev = frontier_archive[t + i] & neg_walls
            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                neighbour_x = x + dx
                neighbour_y = y + dy

                if (
                    neighbour_x < 0
                    or neighbour_y < 0
                    or neighbour_x >= map_width
                    or neighbour_y >= map_height
                ):
                    continue

                if BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x] & prev:
                    return neighbour_idx

        # NOTE order matters - you preference the tiles further back in the timeline, so thats why we check
        #      in reverse order since those that travel further back the timeline, reach the goal faster
        # 9 = bridge merge with free resource line
        # 13 = default bridge move
        # 29 = bridge merge with a stalled or overflowed resource line
        # 40 = bridge merge with a location that has an ore or a bot on it
        for i in [40, 29, 13, 9]:
            if t + i >= archive_size:
                continue

            prev = frontier_archive[t + i] & neg_walls
            for dx, dy in BRIDGE_DIRECTION_DELTAS:
                neighbour_x = x + dx
                neighbour_y = y + dy

                if (
                    neighbour_x < 0
                    or neighbour_y < 0
                    or neighbour_x >= map_width
                    or neighbour_y >= map_height
                ):
                    continue

                if BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x] & prev:
                    return neighbour_idx

        return None


class BotUniformBitboardSolver:
    MAX_COST = 1
    NUM_FRONTIERS = 1

    def __init__(self):
        self.use_environment = True
        self.use_buildings = True
        self.use_bots = False
        self.use_enemy_core = True

    def update(self, bitboard: BitBoardBFS):
        if self.use_environment:
            bitboard.walls |= BITBOARD_GRID.env[1] | BITBOARD_GRID.env_mirrors[0][1]

        if self.use_bots:
            bitboard.walls |= BITBOARD_GRID.ally[1] | BITBOARD_GRID.enemy[1]

        if self.use_buildings:
            bitboard.walls |= (
                BITBOARD_GRID.ally[3]
                | BITBOARD_GRID.ally[4]
                | BITBOARD_GRID.ally[5]
                | BITBOARD_GRID.ally[6]
                | BITBOARD_GRID.ally[11]
                | BITBOARD_GRID.ally[12]
                | BITBOARD_GRID.ally[14]
                | BITBOARD_GRID.enemy[3]
                | BITBOARD_GRID.enemy[4]
                | BITBOARD_GRID.enemy[5]
                | BITBOARD_GRID.enemy[6]
                | BITBOARD_GRID.enemy[11]
                | BITBOARD_GRID.enemy[12]
                | BITBOARD_GRID.enemy[14]
            )

        if self.use_enemy_core:
            enemy_core = BITBOARD_TARGETS[best_enemy_core_idx()]
            enemy_core |= (enemy_core << 1) | (enemy_core >> 1)
            enemy_core |= (enemy_core << MAP_INFO.width) | (enemy_core >> MAP_INFO.width)

            bitboard.walls |= BITBOARD_GRID.enemy[2] | enemy_core

    def solve(
        self,
        bitboard: BitBoardBFS,
        root_idx: int,
        stop_mask: int,
        stop_mask_any: int,
        stop_cost: int | None,
        stop_idx: int | None,
        inflate: int,
        time_limit: int,
        max_depth: int,
    ):
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        get_cpu_time_elapsed = bitboard.get_cpu_time_elapsed

        not_left_edge, not_right_edge = BITBOARD_EDGES[map_width - 20][map_height - 20]

        frontier_archive = bitboard._frontier_archives[0]
        archive = bitboard._archive
        full_mask = bitboard.full_mask

        stop_mask |= 0 if stop_idx is None else BITBOARD_TARGETS[stop_idx]
        stop_mask_any &= ~bitboard.walls
        root_bit = BITBOARD_TARGETS[root_idx]
        visited = bitboard.walls | root_bit

        frontier = root_bit
        frontier_archive[0] = frontier
        archive[0] = visited
        bitboard._archive_size = 1
        t = 1

        depth = 0
        start = get_cpu_time_elapsed()
        while True:
            horiz = (
                frontier | ((frontier << 1) & not_left_edge) | ((frontier >> 1) & not_right_edge)
            )
            next_frontier = (
                horiz | (horiz >> map_width) | ((horiz << map_width) & full_mask)
            ) & ~visited

            if not next_frontier:
                break

            visited |= next_frontier
            if len(archive) == t:
                archive.append(visited)
            else:
                archive[t] = visited
            if len(frontier_archive) == t:
                frontier_archive.append(next_frontier)
            else:
                frontier_archive[t] = next_frontier
            t += 1

            should_stop = False
            if stop_cost is not None and t > stop_cost:
                should_stop = True
            if visited & stop_mask_any:
                should_stop = True
            if stop_mask != 0 and (visited & stop_mask) == stop_mask:
                should_stop = True

            if should_stop:
                inflate -= 1
                if inflate < 0:
                    break

            frontier = next_frontier

            depth += 1
            if get_cpu_time_elapsed() - start > time_limit or (max_depth > 0 and depth > max_depth):
                yield None

                depth = 0
                start = get_cpu_time_elapsed()

        bitboard._archive_size = t
        yield visited & (~bitboard.walls)

    def get_next_tile(self, bitboard: BitBoardBFS, tile_idx: int, t: int):
        x, y = POSITION_CACHE[tile_idx]
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        frontier_archive = bitboard.frontier_archives[0]
        invalid_tiles = bitboard.walls
        valid_tiles = ~invalid_tiles

        prev = frontier_archive[t - 1] & valid_tiles
        for dx, dy in DIRECTION_DELTAS:
            neighbour_x = x + dx
            neighbour_y = y + dy

            if (
                neighbour_x < 0
                or neighbour_y < 0
                or neighbour_x >= map_width
                or neighbour_y >= map_height
            ):
                continue

            if BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x] & prev:
                return neighbour_idx

        # If no valid next tile is found in the archive, look for a valid tile that progresses you to the goal
        target_idx = -1
        min_dist = t

        for dx, dy in DIRECTION_DELTAS:
            neighbour_x = x + dx
            neighbour_y = y + dy

            if (
                neighbour_x < 0
                or neighbour_y < 0
                or neighbour_x >= map_width
                or neighbour_y >= map_height
            ):
                continue

            if (
                BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x]
                & invalid_tiles
            ):
                continue

            neighbour_t = bitboard.dist(neighbour_idx)

            if neighbour_t is not None and neighbour_t < min_dist:
                target_idx = neighbour_idx
                min_dist = neighbour_t

        return target_idx if target_idx != -1 else None

    def get_prev_tile(self, bitboard: BitBoardBFS, tile_idx: int, t: int):
        x, y = POSITION_CACHE[tile_idx]
        map_width = MAP_INFO.width
        map_height = MAP_INFO.height

        archive_size = bitboard.archive_size
        frontier_archive = bitboard.frontier_archives[0]
        invalid_tiles = bitboard.walls
        valid_tiles = ~invalid_tiles

        if t + 1 < archive_size:
            prev = frontier_archive[t + 1] & valid_tiles
            for dx, dy in DIRECTION_DELTAS:
                neighbour_x = x + dx
                neighbour_y = y + dy

                if (
                    neighbour_x < 0
                    or neighbour_y < 0
                    or neighbour_x >= map_width
                    or neighbour_y >= map_height
                ):
                    continue

                if BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x] & prev:
                    return neighbour_idx

        # If no valid next tile is found in the archive, look for a valid tile that progresses you to the goal
        target_idx = -1
        min_dist = t

        for dx, dy in DIRECTION_DELTAS:
            neighbour_x = x + dx
            neighbour_y = y + dy

            if (
                neighbour_x < 0
                or neighbour_y < 0
                or neighbour_x >= map_width
                or neighbour_y >= map_height
            ):
                continue

            if (
                BITBOARD_TARGETS[neighbour_idx := neighbour_y * map_width + neighbour_x]
                & invalid_tiles
            ):
                continue

            neighbour_t = bitboard.dist(neighbour_idx)

            if neighbour_t is not None and neighbour_t < min_dist:
                target_idx = neighbour_idx
                min_dist = neighbour_t

        return target_idx if target_idx != -1 else None
