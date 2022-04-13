def exact_match_reward(taken_action, expected_action, check_name=True):
    status_match = taken_action['status'].lower() == expected_action['status'].lower()
    flight_match = taken_action['flight'] == expected_action['flight'] or (taken_action['flight'] != [] and taken_action['flight'][0] in expected_action['flight'])
    name_match = True
    if check_name:
        name_match = taken_action['name'].lower() == expected_action['name'].lower()
    return {'reward': float(status_match and flight_match and name_match), 'name': float(name_match),
            'status': float(status_match), 'flight': float(flight_match)}

def zero_reward():
    return {'reward': 0.0, 'name': 0.0, 
            'status': 0.0, 'flight': 0.0}