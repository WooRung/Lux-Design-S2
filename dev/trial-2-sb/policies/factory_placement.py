from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import FactoryPlacementActionType

from luxai_s2.utils.heuristics.factory_placement import random_factory_placement, place_near_random_ice

__all__ = ('random_factory_placement', 'place_near_random_ice', 'rl_factory_placement_policy')


def rl_factory_placement_policy(player, obs: ObservationStateDict) -> FactoryPlacementActionType:
    # TODO: RL로 해결하기
    pass
