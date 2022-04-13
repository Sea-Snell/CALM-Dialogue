from ad.ad_types import *


months_map = {'Jan': [1], 'Feb': [2], 'Mar': [3], 'Apr': [4], 'May': [5],
              'June': [6], 'July': [7, 'Jul'], 'Aug': [8], 'Sept': [9, 'Sep'],
              'Oct': [10], 'Nov': [11], 'Dec': [12]}
goals_key_maps = {'book': ['return_month', 'return_day', 'max_price', 'departure_airport',
                           'airline_preference', 'max_connections', 'departure_day',
                           'return_time', 'departure_month', 'name', 'return_airport',
                           'class', 'departure_time', 'goal'],
                  'change': ['return_month', 'return_day', 'max_price', 'departure_airport',
                             'airline_preference', 'max_connections', 'departure_day',
                             'return_time', 'departure_month', 'name', 'return_airport',
                             'class', 'departure_time', 'goal'],
                  'no_reservation': ['name', 'goal'],
                  'no_flight': ['name', 'goal'],
                  'cancel': ['name', 'goal'],
                  'error': ['name', 'goal'],
                }
default_process_f = lambda v: [str(v)]
month_process_f = lambda v: [str(v)] + list(map(str, months_map[v]))
process_value_functions = {
    'return_month': month_process_f,
    'return_day': default_process_f,
    'max_price': lambda v: [str(v), 'price', 'fare', 'cost'],
    'departure_airport': default_process_f,
    'max_connections': lambda v: [str(v), 'connect', 'direct'],
    'departure_day': default_process_f,
    'departure_month': month_process_f,
    'name': default_process_f,
    'return_airport': default_process_f,
    'class': default_process_f,
    'airline_preference': lambda v: [str(v), 'normal', 'standard'],
    'departure_time': default_process_f,
    'return_time': default_process_f,
    'goal': default_process_f,
}

def _find_in_dialogue(dialogue_str, key_strs):
    for key_str in key_strs:
        if key_str.lower() in dialogue_str:
            return True
    return False

def heauristic_filter(item, filter_with_goal=False):
    constraints = item.customer_scenario.intention
    expectation = item.expected_action
    taken_acition = item.events[-1].event.kind_str()
    dialogue_str = '\n'.join(map(lambda x: str(x.event), item.events)).lower()
    status = 'error' if expectation['status'] != taken_acition else expectation['status']
    for k in goals_key_maps[status]:
        v = constraints[k]
        if v == 'None' or (k == 'max_price' and v == 5000):
            continue
        if (not filter_with_goal) and k == 'goal':
            continue
        key_strs = process_value_functions[k](v)
        in_dialogue = _find_in_dialogue(dialogue_str, key_strs)
        if not in_dialogue:
            return True
    return False
