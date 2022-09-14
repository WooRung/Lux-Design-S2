from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from luxai2022.factory import Factory
from luxai2022.map.position import Position
from luxai2022.map_generator.generator import GameMap
from luxai2022.unit import Unit


class Board:
    def __init__(self, seed=None, map_size=None) -> None:
        self.height = 64
        self.width = 64
        if map_size is not None:
            self.height = map_size
            self.width = map_size
        self.seed = seed
        # self.map: GameMap = random_map()
        map_type = None #args.get("map_type", None)
        symmetry = None # args.get("symmetry", None)
        # TODO fix Craters RNG
        rng = np.random.RandomState(seed=seed)
        self.map = GameMap.random_map(seed=seed, symmetry="horizontal", map_type="Cave", width=self.width, height=self.height)
        self.factories_per_team = rng.randint(2, 6)

        # remove bottom once generator is fully ready
        self.map.rubble = self.map.rubble.astype(int)
        self.map.rubble[self.map.rubble < 10] = 0
        self.map.ice[self.map.ice != 0] = 1
        # self.map.ore[self.map.ore > 50] = 1
        # self.map.ore[self.map.ore <= 50] = 0
        self.lichen = np.zeros((self.height, self.width))
        # ownership of lichen by factory id, a simple mask
        # -1 = no ownership
        self.lichen_strains = -np.ones((self.height, self.width), dtype=int)
        # self.units_map: np.ndarray = np.zeros((self.height, self.width))
        self.units_map: Dict[str, List[Unit]] = defaultdict(list)
        # TODO consider unit occupancy map, may speed up computation by storing more info in a form indexable in numpy arrays

        # maps center of factory to the factory
        self.factory_map: Dict[str, 'Factory'] = dict()
        # != -1 if a factory tile is on the location. Equals factory number id
        self.factory_occupancy_map = -np.ones((self.height, self.width), dtype=int)


        # Valid spawn locations
        player_0_spawn_info = self.get_valid_spawns(0)
        player_1_spawn_info = self.get_valid_spawns(1)
        self.spawns = {"player_0": player_0_spawn_info[0],
                       "player_1": player_1_spawn_info[0]}
        self.spawn_masks = {"player_0": player_0_spawn_info[1],
                       "player_1": player_1_spawn_info[1]}
    def pos_hash(self, pos: Position):
        return f"{pos.x},{pos.y}"
    def get_units_at(self, pos: Position):
        pos_hash = self.pos_hash(pos)
        if pos_hash in self.units_map:
            return self.units_map[pos_hash]
        return None
    def get_factory_at(self, pos: Position):
        pos_hash = self.pos_hash(pos)
        if pos_hash in self.factory_map:
            return self.factory_map[pos_hash]
        return None
    def get_valid_spawns(self, team_id):
        xx, yy = np.mgrid[:self.width, :self.height]
        if self.map.symmetry == "vertical":
            if team_id == 0:
                spawns_mask = xx < (self.width - 1) / 2
            else:
                spawns_mask = xx > (self.width - 1) / 2
        if self.map.symmetry == "horizontal":
            if team_id == 0:
                spawns_mask = yy < (self.height - 1) / 2

            else:
                spawns_mask = yy > (self.height - 1) / 2

        if self.map.symmetry == "rotational":
            if team_id == 0:
                spawns_mask = xx < (self.width - 1) / 2

            else:
                spawns_mask = xx > (self.width - 1) / 2

        if self.map.symmetry == "/":
            if team_id == 0:
                spawns_mask = xx - yy < 0

            else:
                spawns_mask = xx - yy > 0

        if self.map.symmetry == "\\":
            if team_id == 0:
                spawns_mask = xx + yy < (self.width + self.height) / 2 - 1

            else:
                spawns_mask = xx + yy > (self.width + self.height) / 2 - 1

        x, y = np.where(spawns_mask)

        spawns = np.array([*zip(x, y)])
        spawns = spawns[(spawns[:, 0] > 0) & (spawns[:, 1] > 0)]
        spawns = spawns[(spawns[:, 0] < self.width - 1) & (spawns[:, 1] < self.height - 1)]
        return spawns, spawns_mask

    @property
    def rubble(self) -> np.ndarray:
        return self.map.rubble
    @property
    def ice(self) -> np.ndarray:
        return self.map.ice
    @property
    def ore(self):
        return self.map.ore
    def state_dict(self):
        return dict(
            rubble=self.rubble.copy(),
            ore=self.ore.copy(),
            ice=self.ice.copy(),
            lichen=self.lichen.copy(),
            lichen_strains=self.lichen_strains.copy(),
            spawns=self.spawns.copy(),
            factories_per_team=self.factories_per_team,
        )
