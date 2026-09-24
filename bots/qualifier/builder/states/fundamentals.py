from __future__ import annotations

from typing import Callable

from cambc import Controller, Direction, EntityType, Position

from botlib import State, BitBoardBFS
from .. import STATE

from botlib import MAP_INFO, UNIT_INFO
from botlib import (
    BITBOARD_ENEMY,
    BITBOARD_ALLY,
    BITBOARD_ENV,
    get_building_id,
    ENEMY_BUILDINGS,
    ALLY_BUILDINGS,
    BITBOARD_TARGETS,
    HAZARDS,
    get_building_hp,
    idx_to_pos,
    get_building_maxhp,
)
from botlib import in_bounds, xy_to_idx, bind_bail, MAX_MAP_SIZE

from botlib.constants import POSITION_CACHE, DIRECTION_CACHE, DIRECTION_DELTAS

from ..utility.movement import move_to, safe_move
from ..utility.building import destroy

from ..data import BOT_PATHING, DISTANCE_FIELD, VISION_DELTAS

import random


# TODO ability to path through ally buildings if unreachable, use a recursive state to acheive this
class path_to:
    # ------------- STATE -------------

    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_idx: int,
            target_dist_squared: int,
            exit_state,
            bail: Callable[[Controller, path_to.state], bool] | None = None,
            give_up: int = 10,
            check_reachability=True,
        ):
            self.exit_state = exit_state

            self.target_idx = target_idx
            self.target_dist_squared = target_dist_squared
            self.bail = bail

            self.fatigue = 0
            self.give_up = give_up

            self.path_to_succeeded = False
            self.visited = 0

            self.last_valid_bot_solve: int | None = None
            self.last_valid_distance_solve: int | None = None
            self.check_reachability = check_reachability

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        self.last_valid_bot_solve = None
        self.last_valid_distance_solve = None

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        if self.bail is not None and self.bail(ct, self):
            return

        if self.give_up <= 0:
            print(f"gave up - didnt make it")
            STATE.switch_to(ct, self.exit_state)
            return

        print(f"pathing to: {POSITION_CACHE[self.target_idx]}")

        tile_pos = POSITION_CACHE[self.target_idx]

        if self.target_dist_squared == 0 and ct.is_in_vision(tile_pos) and self.check_reachability:
            if (
                (BITBOARD_ENV[self.target_idx] & 0b10) != 0
                or (BITBOARD_ENEMY[self.target_idx] & 0b0101_1000_0111_1100) != 0
                or (BITBOARD_ALLY[self.target_idx] & 0b0101_1000_0111_1000) != 0
            ):
                print("cannot path onto a solid tile (target_dist was 0)")
                # Cannot move onto tile, exit
                STATE.switch_to(ct, self.exit_state)
                return

        if UNIT_INFO.position.distance_squared(tile_pos) <= self.target_dist_squared:
            self.path_to_succeeded = True
            STATE.switch_to(ct, self.exit_state)
            return

        print(f"fatigue {self.fatigue}")

        if self.fatigue > 0:
            self.fatigue -= 1
            self.give_up -= 1
            if random.random() < 0.5:
                move_to(ct, self.target_idx, forwards=False)
            return

        BOT_PATHING.solve(self.target_idx, stop_idx=UNIT_INFO.position_idx)
        if BOT_PATHING.ready():
            if not BOT_PATHING.has_visited(UNIT_INFO.position_idx):
                if cls.double_check_unreachable(BOT_PATHING):
                    print("unreachable by bot")
                    self.visited = BOT_PATHING.visited
                    STATE.switch_to(ct, self.exit_state)
                    return
            else:
                self.last_valid_bot_solve = None

        if self.check_reachability:
            DISTANCE_FIELD.solve(self.target_idx, stop_idx=UNIT_INFO.position_idx)
            if DISTANCE_FIELD.ready():
                if not DISTANCE_FIELD.has_visited(UNIT_INFO.position_idx):
                    if cls.double_check_unreachable(DISTANCE_FIELD):
                        print("unreachable by bot")
                        self.visited = DISTANCE_FIELD.visited
                        STATE.switch_to(ct, self.exit_state)
                        return
                else:
                    self.last_valid_distance_solve = None

        move_to(ct, self.target_idx, solve_already_done=True)
        if BOT_PATHING.solver.fallback_move:
            self.give_up -= 1
            self.fatigue = random.randint(0, 3)

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state

    # ------------- METHODS -------------

    @classmethod
    def double_check_unreachable(cls, bitboard: BitBoardBFS):
        # double check unreachable by waiting until next solve

        self = cls.current_state
        if bitboard == DISTANCE_FIELD:
            if self.last_valid_distance_solve is None:
                self.last_valid_distance_solve = DISTANCE_FIELD.solve_version
                return False
            elif self.last_valid_distance_solve < DISTANCE_FIELD.solve_version:
                return True
        else:
            if self.last_valid_bot_solve is None:
                self.last_valid_bot_solve = BOT_PATHING.solve_version
                return False
            elif self.last_valid_bot_solve < BOT_PATHING.solve_version:
                return True

        return False


STATE.register(path_to)

BREAK_TILE_COOLDOWN = [0] * MAX_MAP_SIZE


class break_tile:
    # ------------- STATE -------------

    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            target_idx: int,
            exit_state,
            bail: Callable[[Controller, break_tile.state], bool] | None = None,
            target_id: int = -1,
        ):
            self.exit_state = exit_state

            self.target_idx = target_idx
            self.target_building_id = target_id
            self.ally = False
            self.bail = bail

            self.destroy_successful = False
            self.prev_hp = None

            self.path_task: path_to.state = None

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        if self.target_building_id == -1:
            if ALLY_BUILDINGS[self.target_idx] != 0:
                self.target_building_id = get_building_id(ALLY_BUILDINGS[self.target_idx])
                self.ally = True
            else:
                self.target_building_id = get_building_id(ENEMY_BUILDINGS[self.target_idx])
                self.ally = False

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        if self.bail is not None and self.bail(ct, self):
            return

        if self.path_task is not None and not self.path_task.path_to_succeeded:
            print("blocked path")
            BREAK_TILE_COOLDOWN[self.target_idx] = STATE.tick + 30
            STATE.switch_to(ct, self.exit_state)
            return

        print(
            f"breaking tile: {POSITION_CACHE[self.target_idx]} expecting tile id = {self.target_building_id}"
        )

        if (building_id := get_building_id(ALLY_BUILDINGS[self.target_idx])) is None:
            building_id = get_building_id(ENEMY_BUILDINGS[self.target_idx])

        if building_id is None or building_id != self.target_building_id:
            # tile was destroyed or not what we expected
            print(
                f"tile was destroyed by something else or was not what we expected it to be: {building_id}"
            )
            STATE.switch_to(ct, self.exit_state)
            return

        tile_pos = POSITION_CACHE[self.target_idx]
        if ct.is_in_vision(tile_pos) and not self.ally:
            if (BITBOARD_ENEMY[self.target_idx] & 0b0101_1000_0111_1100) != 0:
                print("cannot destroy a solid tile")
                # Undestroyable tile, exit
                STATE.switch_to(ct, self.exit_state)
                return

        if not self.ally:
            if UNIT_INFO.position.distance_squared(tile_pos) == 0:
                # Attack the tile
                if ct.can_fire(tile_pos):
                    ct.fire(tile_pos)

                    hp = get_building_hp(ENEMY_BUILDINGS[self.target_idx])
                    if self.prev_hp != None and hp >= self.prev_hp:
                        print("can't kill")
                        STATE.switch_to(ct, self.exit_state)
                        BREAK_TILE_COOLDOWN[self.target_idx] = STATE.tick + 30
                        return
                    self.prev_hp = hp

                if ct.get_tile_building_id(tile_pos) is None:
                    print("destroyed tile")
                    self.destroy_successful = True
                    STATE.switch_to(ct, self.exit_state)
                    return
            else:
                # If we are not on the tile move to it
                self.path_task = path_to.state(self.target_idx, 0, self)
                STATE.switch_to(ct, self.path_task)
                return
        else:
            if UNIT_INFO.position.distance_squared(tile_pos) <= 2:
                if destroy(ct, tile_pos):
                    print("destroyed tile")
                    self.destroy_successful = True
                    STATE.switch_to(ct, self.exit_state)
                    return
            else:
                # If we are not on the tile move to it
                self.path_task = path_to.state(self.target_idx, 2, self)
                STATE.switch_to(ct, self.path_task)
                return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state

    # ------------- METHODS -------------

    @classmethod
    def example_method(self):
        pass


STATE.register(break_tile)


class heal_target:
    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            heal_idx: int,
            exit_state,
            target_id: int = -1,
            fatigue: int = 2,
            bail: Callable[[Controller, heal_target.state], bool] | None = None,
        ):
            self.heal_idx = heal_idx
            self.exit_state = exit_state
            self.target_id = target_id

            self.fatigue = fatigue

            self.bail = bail

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

        if self.target_id == -1:
            self.target_id = get_building_id(ALLY_BUILDINGS[self.heal_idx])

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        # in the bail function, you should check whether the target is still there, or do replacement and such
        # this functions assumes you can reach the target, and will heal until full and exit
        if self.bail is not None and self.bail(ct, self):
            return

        target_building = ALLY_BUILDINGS[self.heal_idx]

        if target_building == 0 or self.target_id != get_building_id(target_building):
            print(
                f"building was destroyed or id changed {self.target_id} -> {get_building_id(target_building)}"
            )
            STATE.switch_to(ct, self.exit_state)
            return

        heal_pos = idx_to_pos(self.heal_idx)

        # if out of range, path to
        if UNIT_INFO.position.distance_squared(heal_pos) > 2:
            STATE.switch_to(
                ct,
                path_to.state(self.heal_idx, 2, self),
            )
            return

        # should be in range so, drop the fatigue
        self.fatigue -= 1

        # check hp, then try to heal
        target_hp = get_building_hp(ALLY_BUILDINGS[self.heal_idx])
        max_hp = get_building_maxhp(ALLY_BUILDINGS[self.heal_idx])

        # only heal if you can get full value
        if max_hp - target_hp >= 4:
            if ct.can_heal(heal_pos):
                print(f"healing: {heal_pos}")
                ct.heal(heal_pos)
                self.fatigue += 1
                return
        # TODO, check this whole fatigue thing, idk if its fixed, havent seen issues tho
        if self.fatigue <= 0:
            if max_hp - target_hp > 0 and ct.can_heal(heal_pos):
                print(f"healing: {heal_pos}")
                ct.heal(heal_pos)
                STATE.switch_to(ct, self.exit_state)
                return
            if max_hp == target_hp:
                STATE.switch_to(ct, self.exit_state)
                return
        else:
            pass

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(heal_target)


class fog_max:
    class unveil_item:
        __slots__ = "score", "version", "idx"

        def __init__(self, idx: int):
            self.idx = idx
            self.score = 0
            self.version = 0

    class state:
        STATE_ID: int = -1

        def __init__(
            self,
            exit_state,
            bail: Callable[[Controller, fog_max.state], bool] | None = None,
            mask: int = 0,
            seed_idx: int | None = None,
        ):
            self.bail = bail
            self.exit_state = exit_state

            self.path_task = None

            self.unveil_version = 1
            self.all_unveil_scores_are_zero = True
            self.largest_unveil_idx = 0
            self.largest_unveil_score = 0
            self.unveil_scores: list[fog_max.unveil_item] = [
                fog_max.unveil_item(idx) for idx in range(8)
            ]

            self.mask = mask
            self.vision = 0
            self.unreachable = 0

            self.seed_idx = seed_idx

        def check_move(self, x: int, y: int):
            if not in_bounds(x, y):
                return False
            idx = xy_to_idx(x, y)
            if (
                (BITBOARD_ALLY[idx] & 0b0111_1000_0111_1010) != 0
                or (BITBOARD_ENEMY[idx] & 0b0111_1000_0111_1110) != 0
                or HAZARDS[6][idx] > 0
            ):
                return False
            return (BITBOARD_ENV[idx] & 0b1101) != 0

        def check_fog_deltas(
            self,
            x: int,
            y: int,
            unveil_item: fog_max.unveil_item,
            vision_deltas: list[tuple[int, int]],
        ):
            if unveil_item.version != self.unveil_version:
                unveil_item.version = self.unveil_version
                unveil_item.score = 0

            for dx, dy in vision_deltas:
                if (
                    in_bounds(x + dx, y + dy)
                    and (self.vision & BITBOARD_TARGETS[xy_to_idx(x + dx, y + dy)]) == 0
                ):
                    self.all_unveil_scores_are_zero = False
                    unveil_item.score += 1

            if self.largest_unveil_idx == -1 or unveil_item.score > self.largest_unveil_score:
                self.largest_unveil_idx = unveil_item.idx
                self.largest_unveil_score = unveil_item.score

    current_state: state = None

    @classmethod
    def enter(cls, ct: Controller, state: state):
        cls.current_state = state
        self = cls.current_state

    @classmethod
    def run(cls, ct: Controller):
        self = cls.current_state

        if self.bail is not None and self.bail(ct, self):
            print("fog max bail!")
            return

        if self.seed_idx is not None:
            print(f"seeded: {POSITION_CACHE[self.seed_idx]}")
            self.path_task = path_to.state(self.seed_idx, 9, self, bail=bind_bail(self.bail, self))
            self.seed_idx = None
            STATE.switch_to(ct, self.path_task)
            return

        if self.path_task is not None and not self.path_task.path_to_succeeded:
            self.unreachable |= self.path_task.visited
            self.path_task = None

        self.vision |= UNIT_INFO.vision_mask

        self.vision |= self.mask

        x, y = UNIT_INFO.position

        self.unveil_version += 1
        self.all_unveil_scores_are_zero = True
        self.largest_unveil_idx = -1
        self.largest_unveil_score = 0

        for i in range(8):
            dx, dy = DIRECTION_DELTAS[i]
            if self.check_move(x + dx, y + dy):
                self.check_fog_deltas(x, y, self.unveil_scores[i], VISION_DELTAS[i])

        if not self.all_unveil_scores_are_zero:
            print("fog")
            DISTANCE_FIELD.invalidate_last_solve()
            safe_move(ct, DIRECTION_CACHE[self.largest_unveil_idx])
        else:
            print("no fog")

            # Find nearest off vision
            fog_mask = ~self.vision
            DISTANCE_FIELD.unreachable = self.unreachable
            DISTANCE_FIELD.solver.use_bots = True
            DISTANCE_FIELD.solve(UNIT_INFO.position_idx, stop_mask_any=fog_mask)
            DISTANCE_FIELD.solver.use_bots = False
            DISTANCE_FIELD.unreachable = 0
            if DISTANCE_FIELD.ready():
                last_frontier = (
                    DISTANCE_FIELD.frontier_archives[0][DISTANCE_FIELD.archive_size - 1] & fog_mask
                )

                if last_frontier == 0:
                    print("reset vision")
                    # We have seen everything, reset
                    self.vision = 0

                    # seed a random position on mask
                    bits: int = ~self.mask
                    if bits != 0:
                        k = random.randint(0, bits.bit_count() - 1)
                        print(f"{k}")
                        for _ in range(k):
                            bits &= bits - 1
                        target_idx = (bits & -bits).bit_length() - 1
                    else:
                        target_idx = random.randint(0, MAP_INFO.size - 1)
                else:
                    target_idx = ((last_frontier & (-last_frontier)) - 1).bit_count()

                self.path_task = path_to.state(target_idx, 9, self, bail=bind_bail(self.bail, self))
                STATE.switch_to(ct, self.path_task)
                return

    @classmethod
    def exit(cls, ct: Controller):
        self = cls.current_state


STATE.register(fog_max)
