import numpy as np
from gym import spaces
from luxai_s2.factory import FactoryStateDict
from luxai_s2.map.board import BoardStateDict
from luxai_s2.state import ObservationStateDict
from luxai_s2.state.state import SparseBoardStateDict
from luxai_s2.team import TeamStateDict
from luxai_s2.unit import UnitStateDict



from luxai_s2.wrappers import Controller as IController
from typing import Any, Dict, Literal

from numpy import typing as npt

# from luxai_s2 import actions
from luxai_s2 import actions as lux_actions
from luxai_s2.actions import Action as LuxAction

# from juxai_s2.jux import actions

# from juxai_s2.jux.config import EnvConfig # tuple
from kits.rl.sb3.lux.config import EnvConfig  # dict

# Typing
AgentStrType = Literal['player_0', 'player_1', 'player_2', 'player_3']
ObservationAgentStateType = Dict[AgentStrType, ObservationStateDict]


class BaseController(IController):
    def __init__(self, env_cfg: EnvConfig, action_space: spaces.Space) -> None:
        self.env_cfg = env_cfg
        super().__init__(action_space)

    # def __init__(self, env_cfg: EnvConfig) -> None:
    #
    #     self.env_cfg = env_cfg

    def action_to_lux_action(self, agent: str, obs: ObservationAgentStateType, action: npt.NDArray):
        raise NotImplementedError()

    def action_masks(self, agent: str, obs: ObservationAgentStateType):
        raise NotImplementedError()


class BaseRLController(BaseController):
    def __init__(self, env_cfg: EnvConfig, action_space: spaces.Space) -> None:
        super().__init__(env_cfg, action_space)


class RobotActionMixin:
    __version__ = "0.0.1"

    # Amount(inf)와 repeat과 n을 고정해서  해보자
    def __init__(self, env_cfg: EnvConfig) -> None:
        self.move_act_dims = 4  # [0, move_dir, 0, 0, repeat, n] ->  방향은 4방향 (1,2,3,4) * repeat(2 T/F) * N(inf)
        self.transfer_act_dims = 5 * 5  # [1, transfer_dir, R, A, repeat, n] -> transfer_dir(5) * R(5) * A(inf) * repeat(2) * N(inf)
        self.pickup_act_dims = 5  # [2, 0, R, A, repeat, n] ->  R(5) * A(inf) * repeat(2) * N(inf)
        self.dig_act_dims = 1  # [3, 0, 0, 0, repeat, n] -> 1 * repeat(2) * N(inf)
        self.destruct_act_dims = 1  # [4, 0, 0, 0, repeat, n] -> 1 * repeat(2) * N(inf)
        self.recharge_act_dims = 1  # [5, 0, 0, A, repeat, n] -> 1 * A(inf) * repeat(2) * N(inf)

        self.no_op_dims = 1  # [Unit별로 no opt 하도록]

        # [_high 변수] masking을 위한 index일 듯
        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.destruct_dim_high = self.dig_dim_high + self.destruct_act_dims
        self.recharge_dim_high = self.destruct_dim_high + self.recharge_act_dims
        self.no_op_dim_high = self.recharge_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high

        self.action_space = spaces.Discrete(self.total_act_dims)

        action_space = spaces.Discrete(self.total_act_dims)

        self.env_cfg = None
        super().__init__(env_cfg, action_space)

        if self.env_cfg is None:
            self.env_cfg = env_cfg

    def _is_move_action(self, space_id: int):
        return space_id < self.move_dim_high

    def _get_move_action(self, space_id: int):
        # move direction is id + 1 since we don't allow move center here
        move_dir = space_id + 1

        # TODO: repeat 변경
        # TODO: n값 변경
        repeat = 0
        n_value = 1
        return np.array([0, move_dir, 0, 0, repeat, n_value])

    def _is_transfer_action(self, space_id: int):
        return space_id < self.transfer_dim_high

    def _get_transfer_action(self, space_id: int):
        space_id = space_id - self.move_dim_high
        transfer_dir, resource_type = space_id / 5, space_id % 5
        # transfer_dir = space_id % 5

        # TODO: Amount 변경
        # TODO: repeat 변경
        # TODO: n값 변경
        amount = self.env_cfg.max_transfer_amount
        repeat = 0
        n_value = 1
        return np.array([1, transfer_dir, resource_type, amount, repeat, n_value])

    def _is_pickup_action(self, space_id: int):
        return space_id < self.pickup_dim_high

    def _get_pickup_action(self, space_id: int):
        # TODO: Amount 변경
        # TODO: repeat 변경
        # TODO: n값 변경
        amount = self.env_cfg.max_transfer_amount
        repeat = 0
        n_value = 1
        return np.array([2, 0, 4, amount, repeat, n_value])

    def _is_dig_action(self, space_id: int):
        return space_id < self.dig_dim_high

    def _get_dig_action(self, space_id: int):
        # TODO: repeat 변경
        # TODO: n값 변경
        repeat = 0
        n_value = 1
        return np.array([3, 0, 0, 0, repeat, n_value])

    def _is_destruct_action(self, space_id: int):
        return space_id < self.destruct_dim_high

    def _get_destruct_action(self, space_id: int):
        # TODO: repeat 변경
        # TODO: n값 변경
        repeat = 0
        n_value = 1
        return np.array([4, 0, 0, 0, repeat, n_value])

    def _is_recharge_action(self, space_id: int):
        return space_id < self.recharge_dim_high

    def _get_recharge_action(self, space_id: int):
        # TODO: Amount 변경
        # TODO: repeat 변경
        # TODO: n값 변경

        amount = self.env_cfg.ROBOTS['BATTERY_CAPACITY']
        repeat = 0
        n_value = 1
        return np.array([5, 0, 0, amount, repeat, n_value])


class RLRobotController(RobotActionMixin, BaseRLController):
    def __init__(self, env_cfg: EnvConfig) -> None:
        # self.env_cfg
        super().__init__(env_cfg)

    def action_to_lux_action(self, agent: AgentStrType,
                             obs: ObservationAgentStateType, action: npt.NDArray):
        shared_obs: ObservationStateDict = obs[agent]

        pass

    # UnitStateDict
    # FactoryStateDict
    # BoardStateDict
    # TeamStateDict
    # SparseBoardStateDict

    def action_masks(self, agent: AgentStrType, obs: ObservationAgentStateType):
        """
        특정 agent에 대한 action_mask

        observations 를 보고, ActionSpace 대한 Action Mask
        [Action 별 고려 대상]
        1. move1
        2. transfer
        3. 등등등

        [environment 별 고려 대상]
        1. UnitStateDict: Unit's State:
        2. FactoryStateDict: Factory's State
        3. BoardStateDict: Board's State (board의 자원별 점유 상태)
            {rubble, ore, ice, lichen, lichen_strains, valid_spawns_mask, valid_spawns_mask, factories_per_team}
        4. TeamStateDict: Team의 상태
            {faction: str, water: int, metal: int, factories_to_place: int, factory_strains: List[int],
            place_first: bool, bid: int}
        """

        """
        robot의 action 제한
        1. move: 타일의 끝에 있으면 이동 X
        2. Transfer: 
            2.1. 해당 자원이 없으면 X
            2.2. 해당 방향에 아군 로봇 또는 factory가 없으면 X
        3. Pickup:
            3.1. Factory타일위에 있지 않으면 X
        4. Dig:
            4.1. 해당 타일에 다음 것들이 없으면X [RawResource(Ice, Ore), Rubble, Lichen]
        5. SelfDestruct:
            5.1. 어떤 위치에서든 가능.
        6. Recharge:
            6.1. 전력이 full이면 X
        """
        shared_obs: ObservationStateDict = obs[agent]
        shared_obs['units']
        shared_obs['teams']
        shared_obs['factories']
        shared_obs['board']

        units: Dict[str, Dict[str, UnitStateDict]]
        teams: Dict[str, TeamStateDict]
        factories: Dict[str, Dict[str, FactoryStateDict]]
        board: BoardStateDict
        real_env_steps: int
        global_id: int

        pass


class RLFactoryController(BaseRLController):
    """
    early_step
    # TODO: controller에서 bidding과 factory 위치 선정을 하지 말자

    3. Factory Unit Handling
    """

    def __init__(self, env_cfg: EnvConfig) -> None:
        action_space = spaces.Discrete(10)  # TODO: 확인하기
        super().__init__(env_cfg, action_space)

    def action_to_lux_action(self, agent: AgentStrType, obs: ObservationAgentStateType, action: npt.NDArray):
        shared_obs: ObservationStateDict = obs[agent]
        # shared_obs
        pass

    def action_masks(self, agent: AgentStrType, obs: ObservationAgentStateType):
        pass




# class RobotController(BaseController):
#     def _is_move_action(self, id):
#         pass
#
#     pass
#
#
# class MainController(IController):
#     pass


# Controller = SimpleController
class FactoryController(IController):
    DO_NOTHING = -1
    BUILD_LIGHT = 0
    BUILD_HEAVY = 1
    WATER = 2


class SimpleUnitDiscreteController(IController):

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def action_to_lux_action(
            self, agent: str, obs: Dict[str, Any], action: npt.NDArray
    ):
        shared_obs = obs["player_0"]
        lux_action = dict()
        units = shared_obs["units"][agent]
        for unit_id in units.keys():
            unit = units[unit_id]
            choice = action
            action_queue = []
            no_op = False
            if self._is_move_action(choice):
                action_queue = [self._get_move_action(choice)]
            elif self._is_transfer_action(choice):
                action_queue = [self._get_transfer_action(choice)]
            elif self._is_pickup_action(choice):
                action_queue = [self._get_pickup_action(choice)]
            elif self._is_dig_action(choice):
                action_queue = [self._get_dig_action(choice)]
            else:
                # action is a no_op, so we don't update the action queue
                no_op = True

            # simple trick to help agents conserve power is to avoid updating the action queue
            # if the agent was previously trying to do that particular action already
            if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                if same_actions:
                    no_op = True
            if not no_op:
                lux_action[unit_id] = action_queue

            break

        factories = shared_obs["factories"][agent]
        if len(units) == 0:
            for unit_id in factories.keys():
                lux_action[unit_id] = 1  # build a single heavy

        return lux_action

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        shared_obs = obs[agent]
        factory_occupancy_map = (
                np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                factory_occupancy_map[
                f_pos[0] - 1: f_pos[0] + 2, f_pos[1] - 1: f_pos[1] + 2
                ] = f_data["strain_id"]

        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)
            # movement is always valid
            action_mask[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )
                # check if theres a factory tile there
                if (
                        transfer_pos[0] < 0
                        or transfer_pos[1] < 0
                        or transfer_pos[0] >= len(factory_occupancy_map)
                        or transfer_pos[1] >= len(factory_occupancy_map[0])
                ):
                    continue
                factory_there = factory_occupancy_map[transfer_pos[0], transfer_pos[1]]
                if factory_there in shared_obs["teams"][agent]["factory_strains"]:
                    action_mask[
                        self.transfer_dim_high - self.transfer_act_dims + i
                        ] = True

            factory_there = factory_occupancy_map[pos[0], pos[1]]
            on_top_of_factory = (
                    factory_there in shared_obs["teams"][agent]["factory_strains"]
            )

            # dig is valid only if on top of tile with rubble or resources or lichen
            board_sum = (
                    shared_obs["board"]["ice"][pos[0], pos[1]]
                    + shared_obs["board"]["ore"][pos[0], pos[1]]
                    + shared_obs["board"]["rubble"][pos[0], pos[1]]
                    + shared_obs["board"]["lichen"][pos[0], pos[1]]
            )
            if board_sum > 0 and not on_top_of_factory:
                action_mask[
                self.dig_dim_high - self.dig_act_dims: self.dig_dim_high
                ] = True

            # pickup is valid only if on top of factory tile
            if on_top_of_factory:
                action_mask[
                self.pickup_dim_high - self.pickup_act_dims: self.pickup_dim_high
                ] = True
                action_mask[
                self.dig_dim_high - self.dig_act_dims: self.dig_dim_high
                ] = False

            # no-op is always valid
            action_mask[-1] = True
            break
        return action_mask

    def __init__(self, env_cfg) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.env_cfg = env_cfg
        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)


class RLFactoryController(BaseRLController):
    """
    ealry_step
    1. bidding
    2. factory 위치 선정
    3. Factory Unit Handling
    """

    def __init__(self, env_cfg: EnvConfig) -> None:
        action_space = spaces.Discrete(10)  # TODO: 확인하기
        super().__init__(env_cfg, action_space)

    def action_to_lux_action(self, agent: str, obs: Dict[str, Any], action: npt.NDArray):
        pass

    def action_masks(self, agent: str, obs: Dict[str, Any]):
        pass
