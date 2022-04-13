from ad.airdialogue import Scene
from bots.base_bot import BaseBot

def selfplay(agent1: BaseBot, agent2: BaseBot, scene: Scene, max_turns: int, verbose: bool=True):
    curr_event = None
    agents = [agent1, agent2]
    turn = 0
    if verbose:
        print('==============================')
        print('starting convsersation:')
        print('conversation data:', scene.data)
        print('==============================')
    while (curr_event is None or not curr_event.is_final_action()) and turn < max_turns:
        curr_event = agents[turn % 2].respond(curr_event, scene)
        if verbose:
            print(f'{str(curr_event.agent)}: {str(curr_event.event)}')
        turn += 1
    reward = curr_event.em_reward()
    if verbose:
        print('==============================')
        print('conversation reward:', reward)
        print('expected action:', scene.expected_action)
        print('==============================')
    return reward, curr_event