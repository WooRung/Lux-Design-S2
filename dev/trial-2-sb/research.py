from luxai_s2.state import ObservationStateDict, DeltaObservationStateDict

"""
분석을 위한 utils
only for human.
"""
def obs_by_feature(obs:ObservationStateDict):
  result = {}
  for obs_k in obs['player_0'].keys():
    result[obs_k] ={
      'player_0': obs['player_0'][obs_k],
      'player_1': obs['player_1'][obs_k]
    }
  return result