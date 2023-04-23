from typing import Callable, Dict

from luxai_s2 import LuxAI_S2
from luxai_s2.state import ObservationStateDict
from luxai_s2.unit import BidActionType, FactoryPlacementActionType

from luxai_s2.wrappers import Controller, SB3Wrapper as ISB3Wrapper
from numpy import typing as npt


class SB3Wrapper(ISB3Wrapper):
    def __init__(self,
                 env: LuxAI_S2, bid_policy: Callable[[str, ObservationStateDict], Dict[str, BidActionType]] = None,
                 factory_placement_policy: Callable[[str, ObservationStateDict], Dict[str, FactoryPlacementActionType]] = None,
                 controller: Controller = None
                 ) -> None:
        super().__init__(env, bid_policy, factory_placement_policy, controller)

    def step(self, action: Dict[str, npt.NDArray]):
        return super().step(action)

    def reset(self, **kwargs):
        return super().reset(**kwargs)


