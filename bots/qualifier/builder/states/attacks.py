from __future__ import annotations
from cambc import Controller, Direction, EntityType, Position
from typing import Callable
from botlib import State
from .. import STATE

from botlib import MAP_INFO, UNIT_INFO
from botlib import (
    BITBOARD_ENEMY,
    RESOURCE_FLOW,
    RESOURCE_TYPE,
    CARDINAL_DIRECTION_DELTAS,
    BITBOARD_ALLY,
    BITBOARD_ENV,
    ALLY_BUILDINGS,
    ENEMY_BUILDINGS,
    bind_bail,
    in_bounds,
)
from botlib import (
    best_enemy_core_idx,
    vision_iter,
    not_in_bounds,
    xy_to_idx,
    idx_to_pos,
    get_building_hp,
    get_building_id,
    format_bits,
    get_building_direction,
    get_building_maxhp,
    get_building_type,
    get_bridge_target_idx,
)

from botlib.constants import POSITION_CACHE, DIRECTION_CACHE

from ..utility.movement import move_to
from ..utility.building import (
    direction_to,
    build_if_can,
    can_afford,
    destroy,
    build_if_can_afford,
)

from ..data import BOT_PATHING

# ------------- STATES -------------

from . import attacks_parasite
from . import gather_resource

from .fundamentals import path_to, move_to, break_tile, heal_target, BREAK_TILE_COOLDOWN


def stp_bail(ct: Controller, self: smart_turret_place.state):
    if self.ah_idx is None:
        return False

    x, y = POSITION_CACHE[self.ah_idx]
    # populate tiles lists
    empty_idxs = []
    ally_road_idxs = []
    ally_def_conveyor_idxs = []
    ally_turret_idxs = []
    enemy_turret_idxs = []
    enemy_soft_idxs = []

    # TODO double check
    if ct.is_in_vision(idx_to_pos(self.ah_idx)):
        if (BITBOARD_ALLY[self.ah_idx] & 0b0000_1000_0000_0000) == 0 and (
            BITBOARD_ENEMY[self.ah_idx] & 0b0000_1000_0000_0000
        ) == 0:
            print("harvester gone!, from bail function")
            STATE.switch_to(ct, self.exit_state)
            return

    for dx, dy in CARDINAL_DIRECTION_DELTAS:
        nx = x + dx
        ny = y + dy
        if not_in_bounds(nx, ny):
            continue
        neighbour_idx = ny * MAP_INFO.width + nx
        pos = idx_to_pos(neighbour_idx)

        print(
            f"{idx_to_pos(neighbour_idx)} -> {format_bits(BITBOARD_ALLY[neighbour_idx])} {format_bits(BITBOARD_ENEMY[neighbour_idx])}"
        )

        # TODO enable
        if BITBOARD_ENEMY[neighbour_idx] != 0 and STATE.tick < BREAK_TILE_COOLDOWN[neighbour_idx]:
            continue

        # if empty, or you are standing on it, and there is nothing underneath
        if (
            (BITBOARD_ENV[neighbour_idx] & 0b1101) != 0
            # not wall
            and (
                BITBOARD_ALLY[neighbour_idx]
                == 0
                # and no ally
            )
            and BITBOARD_ENEMY[neighbour_idx] == 0
            # and no enemy tile
        ):
            empty_idxs.append(neighbour_idx)
        elif (BITBOARD_ALLY[neighbour_idx] & 0b0010_0000_0000_0000) != 0:
            # ['road']#
            ally_road_idxs.append(neighbour_idx)
        elif (BITBOARD_ALLY[neighbour_idx] & 0b0000_0010_1000_0000) != 0:
            # armoured conveyor, conveyor
            # if conveyor is pointing INTO the harvester aka DEFENSIVE add it to destructible list
            if get_building_direction(ALLY_BUILDINGS[neighbour_idx]) == direction_to(
                pos, idx_to_pos(self.ah_idx)
            ):
                ally_def_conveyor_idxs.append(neighbour_idx)
        elif (BITBOARD_ALLY[neighbour_idx] & 0b0000_0000_0111_1000) != 0:
            # launcher, breach, sentinel, gunner
            ally_turret_idxs.append(neighbour_idx)
        elif (BITBOARD_ENEMY[neighbour_idx] & 0b0000_0000_0111_1000) != 0:
            # launcher, breach, sentinel, gunner
            enemy_turret_idxs.append(neighbour_idx)
        elif (BITBOARD_ENEMY[neighbour_idx] & 0b1010_0101_1000_0000) != 0:
            # if some other enemy
            # TODO, check, currently excludes bots
            enemy_soft_idxs.append(neighbour_idx)

    # prioritize healing any damaged turrets:
    for idx in ally_turret_idxs:
        if ct.is_in_vision(idx_to_pos(idx)):
            target_hp = get_building_hp(ALLY_BUILDINGS[idx])
            max_hp = get_building_maxhp(ALLY_BUILDINGS[idx])

            if target_hp < max_hp:
                STATE.switch_to(
                    ct,
                    heal_target,
                    heal_target.state(idx, self),
                )
                return True

    print(f"empty idxs {empty_idxs}")
    # find the best turret idx and try to place
    best_turret_idx = -1

    for idx in enemy_soft_idxs:
        best_turret_idx = idx
        break
    for idx in ally_def_conveyor_idxs:
        best_turret_idx = idx
        break
    for idx in ally_road_idxs:
        best_turret_idx = idx
        break
    for idx in empty_idxs:
        best_turret_idx = idx
        break

    if best_turret_idx != self.turret_idx:
        self.turret_idx = best_turret_idx
        STATE.switch_to(ct, self)
        return True  # this bails

    return False  # does not bail


# TODO should be a "path_to" state with a bail out
#      or manually run path_to.run() internally
class attack_generic:
    """
    Generic attack state that walks to enemy core and performs various attacks
    on the way
    """

    # ------------- STATE -------------

    class state:
        STATE_ID: int = -1

    @classmethod
    def enter(self, ct: Controller):
        pass

    @classmethod
    def run(self, ct: Controller):

        titanium_flow = RESOURCE_FLOW[0]
        for idx in vision_iter():
            if (BITBOARD_ENEMY[idx] & 0b1000_0000_0000) != 0 and (BITBOARD_ENV[idx] & 0b0100) != 0:
                # Harvester
                # TODO yeah idk bruh
                x, y = POSITION_CACHE[idx]

                # TODO looping multiple times chop chop, do smth else maybe, probably needs to be a function

                # if all cardinals are full, leave

                spot_available = False

                for dx, dy in CARDINAL_DIRECTION_DELTAS:
                    nx = x + dx
                    ny = y + dy
                    if not_in_bounds(nx, ny):
                        continue
                    neighbour_idx = ny * MAP_INFO.width + nx
                    if (
                        (BITBOARD_ALLY[neighbour_idx] & 0b0101_1010_0111_1110) == 0
                        and (BITBOARD_ENEMY[neighbour_idx] & 0b0101_1010_0111_1110) == 0
                        and (BITBOARD_ENV[neighbour_idx] & 0b0010) == 0
                    ):
                        # ['builder_bot', 'core', 'gunner', 'sentinel', 'breach', 'launcher', 'armoured_conveyor', 'harvester', 'foundry', 'barrier']
                        # wall
                        # if no impassibles, there is a valid spot so break away
                        spot_available = True
                        break

                if spot_available:
                    STATE.switch_to(
                        ct,
                        attack_harvester.state(idx, attack_generic),
                    )
                    return

            # Look for titanium lines with flow and break
            # if RESOURCE_TYPE[idx] & 0b001 or titanium_flow[idx].get_flow() > 0:
            #     if (BITBOARD_ENEMY[idx] & 0b0101_1000_0000) != 0:
            #         # Enemy conveyor, splitter or bridge
            #         STATE.switch_to(ct, break_tile.state(idx, attack_generic))
            #         return

        move_to(ct, best_enemy_core_idx())

    @classmethod
    def exit(self, ct: Controller):
        pass


STATE.register(attack_generic)

# TODO CHECK THE STATE SWITCH LOGIC OF BELOW CLASSES


# put in harvester idx to attack it


class attack_harvester:
    # ------------- STATE -------------

    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            harvester_idx,
            exit_state,
            bail: Callable[[Controller, attack_harvester.state], bool] | None = None,
        ):
            self.harvester_idx = harvester_idx
            self.exit_state = exit_state
            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        # bail function
        if self.bail is not None and self.bail(ct, self):
            return

        # TODO double check
        # if you ocan see it, and the bitboard says there is no harvester, bail
        if ct.is_in_vision(idx_to_pos(self.harvester_idx)):
            if (BITBOARD_ALLY[self.harvester_idx] & 0b0000_1000_0000_0000) == 0 and (
                BITBOARD_ENEMY[self.harvester_idx] & 0b0000_1000_0000_0000
            ) == 0:
                print("harvester gone!")
                STATE.switch_to(ct, self.exit_state)
                return

        x, y = POSITION_CACHE[self.harvester_idx]

        # TODO looping multiple times chop chop, do smth else maybe, probably needs to be a function

        # if all cardinals are full, leave

        spot_available = False

        # check if there is an available spot
        for dx, dy in CARDINAL_DIRECTION_DELTAS:
            nx = x + dx
            ny = y + dy
            if not_in_bounds(nx, ny):
                continue
            neighbour_idx = ny * MAP_INFO.width + nx
            if BREAK_TILE_COOLDOWN[neighbour_idx] > STATE.tick:
                # skip tiles that were recently healed (failed to break)
                continue
            if (
                (
                    (BITBOARD_ALLY[neighbour_idx] & 0b0101_1010_0111_1110) == 0
                    and (BITBOARD_ENEMY[neighbour_idx] & 0b0101_1010_0111_1110) == 0
                )
                or (
                    (BITBOARD_ALLY[neighbour_idx] & 0b0000_0000_0000_0010) != 0
                    and (BITBOARD_ENEMY[neighbour_idx] & 0b0000_0010_0000_0000) == 0
                    and UNIT_INFO.position_idx == neighbour_idx
                )
                or not ct.is_in_vision(idx_to_pos(neighbour_idx))
            ):
                # ['builder_bot', 'core', 'gunner', 'sentinel', 'breach', 'launcher', 'armoured_conveyor', 'harvester', 'foundry', 'barrier']
                # if no impassibles, there is a valid spot so break away

                # added special case, you are the bot on the position, and there is no armoured conveyor
                spot_available = True
                break

        if not spot_available:
            print("no spot, in attack harvester, bailing")
            STATE.switch_to(ct, self.exit_state)
            return

        # populate tiles lists
        empty_idxs = []
        ally_road_idxs = []
        ally_def_conveyor_idxs = []
        ally_turret_idxs = []
        enemy_turret_idxs = []
        enemy_soft_idxs = []

        for dx, dy in CARDINAL_DIRECTION_DELTAS:
            nx = x + dx
            ny = y + dy
            if not_in_bounds(nx, ny):
                continue
            neighbour_idx = ny * MAP_INFO.width + nx

            if BREAK_TILE_COOLDOWN[neighbour_idx] > STATE.tick:
                continue

            pos = idx_to_pos(neighbour_idx)

            print(
                f"{idx_to_pos(neighbour_idx)} -> {format_bits(BITBOARD_ALLY[neighbour_idx])} {format_bits(BITBOARD_ENEMY[neighbour_idx])}"
            )

            if (
                BITBOARD_ENEMY[neighbour_idx] != 0
                and STATE.tick < BREAK_TILE_COOLDOWN[neighbour_idx]
            ):
                continue

            # if empty, or you are standing on it, and there is nothing underneath
            if (
                (BITBOARD_ENV[neighbour_idx] & 0b1101) != 0
                # not wall
                and (
                    BITBOARD_ALLY[neighbour_idx]
                    == 0
                    # and no ally
                )
                and BITBOARD_ENEMY[neighbour_idx] == 0
                # and no enemy tile
            ):
                empty_idxs.append(neighbour_idx)
            elif (BITBOARD_ALLY[neighbour_idx] & 0b0010_0000_0000_0000) != 0:
                # ['road']#
                ally_road_idxs.append(neighbour_idx)
            elif (BITBOARD_ALLY[neighbour_idx] & 0b0000_0010_1000_0000) != 0:
                # armoured conveyor, conveyor
                # if conveyor is pointing INTO the harvester aka DEFENSIVE add it to destructible list
                # TODO check, changed from awu function to ours
                if get_building_direction(ALLY_BUILDINGS[neighbour_idx]) == direction_to(
                    pos, idx_to_pos(self.harvester_idx)
                ):
                    ally_def_conveyor_idxs.append(neighbour_idx)
            elif (BITBOARD_ALLY[neighbour_idx] & 0b0000_0000_0111_1000) != 0:
                # launcher, breach, sentinel, gunner
                ally_turret_idxs.append(neighbour_idx)
            elif (BITBOARD_ENEMY[neighbour_idx] & 0b0000_0000_0111_1000) != 0:
                # launcher, breach, sentinel, gunner
                enemy_turret_idxs.append(neighbour_idx)
            elif (BITBOARD_ENEMY[neighbour_idx] & 0b1010_0101_1000_0000) != 0 and (
                BITBOARD_ENEMY[neighbour_idx] & 0b0010
            ) == 0:
                # if some other enemy
                # NOTE, if enemy bot on the soft tile, dont mark
                enemy_soft_idxs.append(neighbour_idx)

        # prioritize healing any damaged turrets:

        for idx in ally_turret_idxs:
            # if you cant see it, dont heal it !
            if not ct.is_in_vision(idx_to_pos(idx)):
                continue
            target_hp = get_building_hp(ALLY_BUILDINGS[idx])
            max_hp = get_building_maxhp(ALLY_BUILDINGS[idx])

            if target_hp < max_hp:
                STATE.switch_to(
                    ct,
                    heal_target,
                    heal_target.state(idx, self),
                )
                return

        print(f"empty idxs {empty_idxs}")
        # find the best turret idx and try to place
        best_turret_idx = -1

        for idx in enemy_soft_idxs:
            best_turret_idx = idx
            break
        for idx in ally_def_conveyor_idxs:
            best_turret_idx = idx
            break
        for idx in ally_road_idxs:
            best_turret_idx = idx
            break
        for idx in empty_idxs:
            best_turret_idx = idx
            break

        if best_turret_idx != -1:
            print(f"attempt to place at {idx_to_pos(best_turret_idx)}")
            STATE.switch_to(
                ct,
                smart_turret_place.state(
                    best_turret_idx, self, ah_idx=self.harvester_idx, bail=stp_bail
                ),
            )
            return

        STATE.switch_to(ct, self.exit_state)
        return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(attack_harvester)


class smart_turret_place:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            turret_idx: int,
            exit_state,
            ah_idx: int = None,
            bail: Callable[[Controller, smart_turret_place.state], bool] | None = None,
        ):
            self.turret_idx = turret_idx
            self.exit_state = exit_state
            self.ah_idx = ah_idx

            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        print("targeting:")
        print(idx_to_pos(self.turret_idx))

        # bail function
        if self.bail is not None and self.bail(ct, self):
            print("triggered bail function!")
            return

        if STATE.tick < BREAK_TILE_COOLDOWN[self.turret_idx]:
            # skip tiles that were recently healed (failed to break)
            STATE.switch_to(ct, self.exit_state)
            return

        # if enemy indestructible, or wall, bail
        if (BITBOARD_ENEMY[self.turret_idx] & 0b0101_1010_0111_1110) != 0 or (
            BITBOARD_ENV[self.turret_idx] & 0b0010 != 0
        ):
            # TODO added builder bot ['core', 'gunner', 'sentinel', 'breach', 'launcher', 'armoured_conveyor', 'harvester', 'foundry', 'barrier']
            # wall
            STATE.switch_to(ct, self.exit_state)
            return

        # if ally important building, bail TODO, what buildings should i overwrite
        if (BITBOARD_ALLY[self.turret_idx] & 0b0001_1000_0111_1000) != 0:
            # ['gunner', 'sentinel', 'breach', 'launcher', 'harvester', 'foundry']
            STATE.switch_to(ct, self.exit_state)
            return

        # if out of range, path to
        # TODO, trying to fix corner bug, check
        if (
            UNIT_INFO.position.distance_squared(idx_to_pos(self.turret_idx)) > 2
            and self.turret_idx != -1
        ):
            STATE.switch_to(
                ct,
                path_to.state(self.turret_idx, 2, self, bail=bind_bail(self.bail, self)),
            )
            return
        # TODO what if you cannot reach tile
        # we have returned, double check we are in range, and not on the tile
        # if UNIT_INFO.position.distance_squared(idx_to_pos(self.turret_idx)) > 2:
        #     # assume unreachable
        #     STATE.switch_to(ct, self.exit_state)
        #     return

        # if enemy tile on target turret location, try and break it
        if (BITBOARD_ENEMY[self.turret_idx] & 0b0010_0101_1000_0000) != 0:
            # ['conveyor', 'splitter', 'bridge', 'road']
            print(f"there is something here, {idx_to_pos(self.turret_idx)}")
            STATE.switch_to(
                ct,
                break_tile.state(self.turret_idx, self, bail=bind_bail(self.bail, self)),
            )
            return

        # # if on the tile, walk off one square
        # if UNIT_INFO.position.distance_squared(idx_to_pos(self.turret_idx)) == 0:
        #     move_to(ct, UNIT_INFO.ally_core_idx)
        #     return

        # Detect neighbouring harvesters

        harvester_idx_list = []

        x, y = POSITION_CACHE[self.turret_idx]

        for dx, dy in CARDINAL_DIRECTION_DELTAS:
            nx = x + dx
            ny = y + dy
            if not_in_bounds(nx, ny):
                continue
            neighbour_idx = ny * MAP_INFO.width + nx
            if (BITBOARD_ENEMY[neighbour_idx] & 0b0001_1000_0000_0000) != 0 or (
                BITBOARD_ALLY[neighbour_idx] & 0b0001_1000_0000_0000
            ) != 0:
                # if enemy or ally HARVESTER or FOUNDRY, feeding the turret position
                harvester_idx_list.append(neighbour_idx)

        # iterate through neigbouring harvesters and look for a candidate turret type and direction

        building_type: EntityType | None = None
        building_direction = None

        found = False

        for idx in harvester_idx_list:
            x, y = POSITION_CACHE[idx]

            for dx, dy in CARDINAL_DIRECTION_DELTAS:
                nx = x + dx
                ny = y + dy
                # target enemy turrets
                target_bitcode = 0b0011_1000
                # GUNNER, SENTINEL, BREACH

                if not_in_bounds(nx, ny):
                    continue

                if dy == 1:
                    if (
                        in_bounds(x + 1, y) and BITBOARD_ENEMY[xy_to_idx(x + 1, y)] & target_bitcode
                    ) != 0:
                        building_type = EntityType.GUNNER
                        building_direction = Direction.NORTHEAST
                        found = True
                        break
                    if (
                        in_bounds(x - 1, y) and BITBOARD_ENEMY[xy_to_idx(x - 1, y)] & target_bitcode
                    ) != 0:
                        building_type = EntityType.GUNNER
                        building_direction = Direction.NORTHWEST
                        found = True
                        break
                elif dy == -1:
                    if (
                        in_bounds(x + 1, y) and BITBOARD_ENEMY[xy_to_idx(x + 1, y)] & target_bitcode
                    ) != 0:
                        building_type = EntityType.GUNNER
                        building_direction = Direction.NORTHEAST
                        found = True
                        break
                    if (
                        in_bounds(x - 1, y) and BITBOARD_ENEMY[xy_to_idx(x - 1, y)] & target_bitcode
                    ) != 0:
                        building_type = EntityType.GUNNER
                        building_direction = Direction.NORTHWEST
                        found = True
                        break
                elif dx == -1:
                    if (
                        in_bounds(x + 1, y) and BITBOARD_ENEMY[xy_to_idx(x + 1, y)] & target_bitcode
                    ) != 0:
                        building_type = EntityType.GUNNER
                        building_direction = Direction.NORTHEAST
                        found = True
                        break
                    if (
                        in_bounds(x - 1, y)
                        and BITBOARD_ENEMY[xy_to_idx(x - 1, y)] & target_bitcode != 0
                    ):
                        building_type = EntityType.GUNNER
                        building_direction = Direction.NORTHWEST
                        found = True
                        break
                elif dx == -1:
                    if (
                        in_bounds(x + 1, y) and BITBOARD_ENEMY[xy_to_idx(x + 1, y)] & target_bitcode
                    ) != 0:
                        building_type = EntityType.GUNNER
                        building_direction = Direction.NORTHEAST
                        found = True
                        break
                    if (
                        in_bounds(x - 1, y) and BITBOARD_ENEMY[xy_to_idx(x - 1, y)] & target_bitcode
                    ) != 0:
                        building_type = EntityType.GUNNER
                        building_direction = Direction.NORTHWEST
                        found = True
                        break
            if found:
                break

        if not found:
            # if no relevant gunner, place a diagonal sentinel facing towards enemy core
            # TODO switch back to sentinel
            building_type = EntityType.SENTINEL
            building_direction = direction_to(
                self.turret_idx, best_enemy_core_idx(), diagonals_only=True
            )

        print("type, dir")
        print(building_type, building_direction)

        if building_type is not None and can_afford(ct, building_type):
            # if on target turret location, and no tile, move off and place
            # if on the tile, and empty, walk off one square
            if UNIT_INFO.position.distance_squared(idx_to_pos(self.turret_idx)) == 0 and (
                BITBOARD_ENEMY[self.turret_idx] == 0
            ):
                move_to(ct, self.turret_idx, forwards=False)
            print("can afford!")
            destroy(ct, self.turret_idx)
            build_if_can(ct, building_type, self.turret_idx, building_direction)

        STATE.switch_to(ct, self.exit_state)
        return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(smart_turret_place)


class snipe:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_idx: int,
            ammo_idx: int,
            exit_state,
            ammo_type: EntityType = None,
            target_id: int = -1,
            ammo_dir: Direction = None,
            ammo_bridge_idx: int = -1,
            bail: Callable[[Controller, snipe_replace.state], bool] | None = None,
        ):
            self.target_idx = target_idx
            self.target_id = target_id
            self.ammo_idx = ammo_idx
            self.ammo_type = ammo_type
            self.ammo_dir = ammo_dir
            self.exit_state = exit_state
            self.ammo_bridge_idx = ammo_bridge_idx

            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        if self.target_id == -1:
            self.target_id = get_building_id(ENEMY_BUILDINGS[self.target_idx])

        if self.ammo_type is None:
            self.ammo_type = get_building_type(ALLY_BUILDINGS[self.ammo_idx])

        if self.ammo_dir is None:
            self.ammo_dir = get_building_direction(ALLY_BUILDINGS[self.ammo_idx])

        if self.ammo_dir == -1:
            self.ammo_dir = get_bridge_target_idx(ALLY_BUILDINGS[self.ammo_idx])

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        # bail function
        if self.bail is not None and self.bail(ct, self):
            return

        ammo_pos = idx_to_pos(self.ammo_idx)
        target_pos = idx_to_pos(self.target_idx)

        # if out of range, path to
        # TODO bail on unreachable
        # IMPORTANT TODO TODO you need to ensure you can see the target as well
        if UNIT_INFO.position.distance_squared(ammo_pos) > 2:
            STATE.switch_to(
                ct,
                path_to.state(self.ammo_idx, 2, self),
            )
            return

        # TODO, use path to not move to, just want to do a single step off
        if UNIT_INFO.position == ammo_pos:
            move_to(ct, UNIT_INFO.ally_core_idx)

        turret_direction = direction_to(ammo_pos, target_pos)

        built = False
        if ((BITBOARD_ALLY[self.ammo_idx] & 0b0001_0000) != 0) and get_building_direction(
            ALLY_BUILDINGS[self.ammo_idx] == turret_direction
        ):
            # if there is a sentinel pointing the right way, exit
            built = True
        else:
            destroy(ct, ammo_pos)
            build_if_can_afford(ct, EntityType.SENTINEL, ammo_pos, turret_direction)

        # NOTE, only works on enemy buildings
        if built and self.target_id != get_building_id(ENEMY_BUILDINGS[self.target_idx]):
            # TODO, check this works with bridges
            destroy(ct, ammo_pos)
            if self.ammo_type == EntityType.BRIDGE:
                build_if_can_afford(ct, self.ammo_type, ammo_pos, idx_to_pos(self.ammo_bridge_idx))
            else:
                build_if_can_afford(ct, self.ammo_type, ammo_pos, self.ammo_dir)

        # TODO double check the tile is replaced properly
        # target is gone, then exit
        if self.target_id != get_building_id(ENEMY_BUILDINGS[self.target_idx]):
            print("destroyed and replaced, exiting")
            STATE.switch_to(ct, self.exit_state)
            return
        else:
            print("job not done")

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(snipe)


class snipe_replace:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_idx: int,
            ammo_idx: int,
            exit_state,
            ammo_type: EntityType = None,
            target_id: int = -1,
            ammo_dir: Direction = None,
            ammo_bridge_idx: int = -1,
            bail: Callable[[Controller, snipe_replace.state], bool] | None = None,
        ):
            self.target_idx = target_idx
            self.target_id = target_id
            self.ammo_idx = ammo_idx
            self.ammo_type = ammo_type
            self.ammo_dir = ammo_dir
            self.exit_state = exit_state
            self.ammo_bridge_idx = ammo_bridge_idx

            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        if self.target_id == -1:
            self.target_id = get_building_id(ENEMY_BUILDINGS[self.target_idx])

        if self.ammo_type is None:
            self.ammo_type = get_building_type(ALLY_BUILDINGS[self.ammo_idx])

        if self.ammo_dir is None:
            self.ammo_dir = get_building_direction(ALLY_BUILDINGS[self.ammo_idx])

        if self.ammo_dir == -1:
            self.ammo_dir = get_bridge_target_idx(ALLY_BUILDINGS[self.ammo_idx])

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        # bail function
        if self.bail is not None and self.bail(ct, self):
            return

        ammo_pos = idx_to_pos(self.ammo_idx)
        target_pos = idx_to_pos(self.target_idx)

        # if out of range, path to
        # TODO bail on unreachable
        # IMPORTANT TODO TODO you need to ensure you can see the target as well
        if UNIT_INFO.position.distance_squared(ammo_pos) > 2:
            STATE.switch_to(
                ct,
                path_to.state(self.ammo_idx, 2, self),
            )
            return

        # TODO, use path to not move to, just want to do a single step off
        if UNIT_INFO.position == ammo_pos:
            move_to(ct, UNIT_INFO.ally_core_idx)

        turret_direction = direction_to(ammo_pos, target_pos)

        built = False
        if ((BITBOARD_ALLY[self.ammo_idx] & 0b0001_0000) != 0) and get_building_direction(
            ALLY_BUILDINGS[self.ammo_idx] == turret_direction
        ):
            # if there is a sentinel pointing the right way, exit
            built = True
        else:
            destroy(ct, ammo_pos)
            build_if_can_afford(ct, EntityType.SENTINEL, ammo_pos, turret_direction)

        # NOTE, only works on enemy buildings
        if built and self.target_id != get_building_id(ENEMY_BUILDINGS[self.target_idx]):
            # TODO, check this works with bridges
            destroy(ct, ammo_pos)
            if self.ammo_type == EntityType.BRIDGE:
                build_if_can_afford(ct, self.ammo_type, ammo_pos, idx_to_pos(self.ammo_bridge_idx))
            else:
                build_if_can_afford(ct, self.ammo_type, ammo_pos, self.ammo_dir)

        # TODO double check the tile is replaced properly
        # target is gone, then exit
        if self.target_id != get_building_id(ENEMY_BUILDINGS[self.target_idx]):
            print("destroyed and replaced, exiting")
            STATE.switch_to(ct, self.exit_state)
            return
        else:
            print("job not done")

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(snipe_replace)
