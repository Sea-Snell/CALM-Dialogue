
def check_class(kb_entry, constraints):
    return kb_entry['class'] == constraints['class']

def check_month(kb_entry, constraints, kind):
    return kb_entry[kind+'_month'] == constraints[kind+'_month']

def check_day(kb_entry, constraints, kind):
    return kb_entry[kind+'_day'] == constraints[kind+'_day']

def check_airport(kb_entry, constraints, kind):
    return kb_entry[kind + '_airport'] == constraints[kind + '_airport']

def check_airline(kb_entry, constraints):
    if constraints['airline_preference'] == 'normal-cost':
        return kb_entry['airline'].lower() in ['aa', 'ua', 'hawaiian', 'delta']
    return True

def check_price(kb_entry, constraints):
    return kb_entry['price'] <= constraints['max_price']

def check_max_connection(kb_entry, constraints):
    return kb_entry['num_connections'] <= constraints['max_connections']

def check_time(kb_entry, constraints, kind):
    if constraints[kind+'_time'] == 'morning':
        return kb_entry[kind+'_time_num'] >= 3 and kb_entry[kind+'_time_num'] <= 11
    if constraints[kind+'_time'] == 'afternoon':
        return kb_entry[kind+'_time_num'] >= 12 and kb_entry[kind+'_time_num'] <= 19
    if constraints[kind+'_time'] == 'evening':
        return kb_entry[kind+'_time_num'] >= 20 or kb_entry[kind+'_time_num'] <= 2
    return True

constraint_checkers = {
                        'class': check_class,
                        'departure_month': lambda kb, const: check_month(kb, const, 'departure'),
                        'return_month': lambda kb, const: check_month(kb, const, 'return'),
                        'departure_day': lambda kb, const: check_day(kb, const, 'departure'),
                        'return_day': lambda kb, const: check_day(kb, const, 'return'),
                        'departure_airport': lambda kb, const: check_airport(kb, const, 'departure'),
                        'return_airport': lambda kb, const: check_airport(kb, const, 'return'),
                        'departure_time': lambda kb, const: check_time(kb, const, 'departure'),
                        'return_time': lambda kb, const: check_time(kb, const, 'return'),
                        'max_price': check_price,
                        'airline_preference': check_airline,
                        'max_connections': check_max_connection,
                        'airline_preference': check_airline,
                       }

def get_true_action(kb, reservation, constraints):
    status, flight = infer_action_given_constraints(kb,
                                                    reservation,
                                                    constraints)
    flight = filter_flights_by_price(flight, kb)
    predicted_action = {'status': status, 'flight': flight, 'name': constraints['name']}
    return predicted_action

def infer_action_given_constraints(kb, reservation, constraints):
    if constraints['goal'] == 'cancel':
        if reservation == 0:
            return 'no_reservation', []
        else:
            return 'cancel', []
    if constraints['goal'] == 'change':
        if reservation == 0:
            return 'no_reservation', []
        else:
            found_flights = run_table_inference(kb, constraints)
            if len(found_flights) == 0:
                return 'no_flight', []
            return 'change', found_flights
    if constraints['goal'] == 'book':
        found_flights = run_table_inference(kb, constraints)
        if len(found_flights) == 0:
            return 'no_flight', []
        return 'book', found_flights
    raise NotImplementedError

def run_table_inference(kb, constraints):
    valid_flights = []
    for item in kb:
        item_matches = True
        for k, v in constraints.items():
            if v == 'None':
                continue
            if k == 'goal' or k == 'name':
                continue
            if not constraint_checkers[k](item, constraints):
                item_matches = False
                break
        if item_matches:
            valid_flights.append(item['flight_number'])
    return valid_flights

def filter_flights_by_price(valid_flights, kb):
    if len(valid_flights) == 0:
        return []
    flight_to_kb = {item['flight_number']: item for item in kb}
    prices = []
    for flight in valid_flights:
        prices.append(flight_to_kb[flight]['price'])
    min_price = min(prices)
    final_flights = []
    for i in range(len(valid_flights)):
        if prices[i] == min_price:
            final_flights.append(valid_flights[i])
    return final_flights
