"""Pesquisa informada: heurísticas, sucessores e Greedy Best-First.
A* pode reutilizar as heurísticas e o espaço de estados.
"""

import heapq
from datetime import timedelta
from itertools import combinations

from src.model import (
    PICKUP, UNLOAD, LOAD, DELIVER,
    NEW, PICKED, STORED, LOADED, DELIVERED,
    apply_stop, simulate, check_solution,
)

def pending_operations(problem, state):
    count = 0.0
    for g in problem.pickup_groups:
        new = sum(1 for i in g.ids if state.status[i] == NEW)
        picked = sum(1 for i in g.ids if state.status[i] == PICKED)
        count += (2 * new + picked) / len(g.ids)
    for ids in problem.orders.values():
        s = state.status[ids[0]]
        count += 2 if s < LOADED else (1 if s == LOADED else 0)
    return count

def remaining_km(problem, state):
    W = problem.warehouse_location
    total = 0.0
    carrying = set()

    for g in problem.pickup_groups:
        if any(state.status[i] == NEW for i in g.ids):
            nearest = min(problem.get_distance(state.place[v], g.location)
                          for v in state.place)
            total += nearest + problem.get_distance(g.location, W)
        for i in g.ids:
            if state.status[i] == PICKED:
                carrying.add(state.owner[i])

    for v in carrying:
        total += problem.get_distance(state.place[v], W)

    for ids in problem.orders.values():
        first = ids[0]
        place = problem.request[first].delivery_location
        if state.status[first] == LOADED:
            total += problem.get_distance(state.place[state.owner[first]], place)
        elif state.status[first] != DELIVERED:
            total += problem.get_distance(W, place)

    return total

def heuristic01(problem, state):
    return pending_operations(problem, state)

def heuristic02(problem, state):
    return (pending_operations(problem, state), remaining_km(problem, state))

def final_cost(problem, state):
    W = problem.warehouse_location
    return state.distance + sum(problem.get_distance(p, W) for p in state.place.values())

def admissible_km(problem, state):
    d = problem.lower_distance
    W = problem.warehouse_location
    vehicles = list(state.place)
    home = {v: d(state.place[v], W) for v in vehicles}
    base = sum(home.values())
    best = base

    def cheapest(path_cost):
        return min(base - home[v] + path_cost(v) for v in vehicles)

    for place in {r.pickup_location for r in problem.requests
                  if state.status[r.id] == NEW}:
        best = max(best, cheapest(lambda v: d(state.place[v], place) + d(place, W)))

    for ids in problem.orders.values():
        first = ids[0]
        status = state.status[first]
        if status == DELIVERED:
            continue
        place = problem.request[first].delivery_location
        if status == LOADED:
            v0 = state.owner[first]
            best = max(best, base - home[v0] + d(state.place[v0], place) + d(place, W))
        else:
            best = max(best, cheapest(
                lambda v: d(state.place[v], W) + d(W, place) + d(place, W)))

    return best

def is_goal(problem, state):
    return all(s == DELIVERED for s in state.status.values())

def pickup_options(problem, state, vehicle_id, full_limit=8):
    free = problem.vehicle[vehicle_id].capacity_kg - state.load[vehicle_id]
    by_place = {}
    for r in problem.requests:
        if state.status[r.id] == NEW and r.weight_kg <= free + 1e-6:
            by_place.setdefault(r.pickup_location, []).append(r)

    options = set()
    for place, items in by_place.items():
        stop = state.stop[vehicle_id]
        if stop is not None and stop[0] == PICKUP and stop[1] == place:
            arrival = stop[2]
        elif problem.lower_travel(state.place[vehicle_id], place) is None:
            continue
        else:
            arrival = state.time[vehicle_id] + problem.travel_time(
                state.place[vehicle_id], place)

        open_items = sorted((r for r in items if r.pickup_end >= arrival),
                            key=lambda r: (r.pickup_end, r.id))

        def compatible(chosen):
            lo = max([arrival] + [r.pickup_start for r in chosen])
            hi = min(r.pickup_end for r in chosen)
            return lo <= hi and sum(r.weight_kg for r in chosen) <= free + 1e-6

        for seed in open_items:
            pool = [r for r in open_items if r is not seed and compatible([seed, r])]

            if len(pool) < full_limit:
                for size in range(len(pool) + 1):
                    for extra in combinations(pool, size):
                        chosen = (seed,) + extra
                        if compatible(chosen):
                            options.add(tuple(sorted(r.id for r in chosen)))
            else:
                orderings = (pool,
                             sorted(pool, key=lambda r: (r.weight_kg, r.id)),
                             sorted(pool, key=lambda r: (-r.weight_kg, r.id)))
                for ordering in orderings:
                    chosen = [seed]
                    for r in ordering:
                        if compatible(chosen + [r]):
                            chosen.append(r)
                    options.add(tuple(sorted(r.id for r in chosen)))

    return sorted(options)

def get_successors(problem, state, pickup_limit=8):
    status, owner = state.status, state.owner

    for vehicle in problem.vehicles:
        v = vehicle.id

        for ids in pickup_options(problem, state, v, pickup_limit):
            new, _ = apply_stop(problem, state, v, PICKUP, ids)
            if new:
                yield new

        carried = tuple(i for i in status if status[i] == PICKED and owner[i] == v)
        if carried:
            new, _ = apply_stop(problem, state, v, UNLOAD, carried)
            if new:
                yield new

        for ids in problem.orders.values():
            if all(status[i] == STORED for i in ids):
                new, _ = apply_stop(problem, state, v, LOAD, ids)
            elif all(status[i] == LOADED and owner[i] == v for i in ids):
                new, _ = apply_stop(problem, state, v, DELIVER, ids)
            else:
                continue
            if new:
                yield new

def earliest_at(problem, state, location):
    best = None
    for v in state.time:
        stop = state.stop[v]
        if state.place[v] == location and stop is not None:
            t = min(stop[2], state.time[v])
        else:
            travel = problem.lower_travel(state.place[v], location)
            if travel is None:
                continue
            t = state.time[v] + travel
        if best is None or t < best:
            best = t
    return best

def is_viable(problem, state):
    W = problem.warehouse_location
    request, status = problem.request, state.status
    lower = problem.lower_travel
    stop_t, handling = problem.stop_service, problem.warehouse_service

    unload_end = {}
    for r in problem.requests:
        if status[r.id] != NEW:
            continue
        there = earliest_at(problem, state, r.pickup_location)
        back = lower(r.pickup_location, W)
        if there is None or back is None:
            return False
        t = max(there, r.pickup_start)
        if t > r.pickup_end or t + stop_t + back > problem.horizon:
            return False
        unload_end[r.id] = t + stop_t + back + handling

    at_warehouse = earliest_at(problem, state, W)
    if at_warehouse is None:
        return False

    for ids in problem.orders.values():
        first = ids[0]
        if status[first] == DELIVERED:
            continue

        place = request[first].delivery_location
        w_start = max(request[i].delivery_start for i in ids)
        w_end = min(request[i].delivery_end for i in ids)
        back_home = lower(place, W)
        if back_home is None:
            return False

        if status[first] == LOADED:
            v = state.owner[first]
            travel = lower(state.place[v], place)
            if travel is None:
                return False
            t = max(state.time[v] + travel, w_start)
        else:
            ready = at_warehouse
            for i in ids:
                if status[i] == NEW:
                    ready = max(ready, unload_end[i])
                elif status[i] == PICKED:
                    v = state.owner[i]
                    travel = lower(state.place[v], W)
                    if travel is None:
                        return False
                    ready = max(ready, state.time[v] + travel + handling)
                else:
                    ready = max(ready, state.ready_at[i])

            for i in ids:
                if status[i] == STORED and ready - state.stored_at[i] > problem.dwell_max:
                    return False

            to_place = lower(W, place)
            if to_place is None:
                return False
            t = max(ready + handling + to_place, w_start)

        if t > w_end or t + stop_t + back_home > problem.horizon:
            return False

    return True

def _greedy_pass(problem, initial_state, heuristic, window, max_expansions,
                 pickup_limit):

    counter = 0
    priority_queue = [(heuristic(problem, initial_state), 0.0, counter, initial_state)]
    visited = set()
    expanded = 0

    while priority_queue:
        _, _, _, current = heapq.heappop(priority_queue)

        if is_goal(problem, current):
            start_times = {v: initial_state.time[v] for v in initial_state.time}
            valid, _, _ = check_solution(problem, current.routes, start_times)
            if valid:
                return current.routes, expanded
            continue

        key = current.key()
        if key in visited:
            continue
        visited.add(key)

        if expanded >= max_expansions:
            break
        expanded += 1

        moves = [s for s in get_successors(problem, current, pickup_limit)
                 if s.key() not in visited]

        if window is not None and moves:
            earliest = min(s.last[5] for s in moves)
            moves = [s for s in moves if s.last[5] <= earliest + window]

        for new_state in moves:
            if not is_viable(problem, new_state):
                continue
            counter += 1
            heapq.heappush(priority_queue, (heuristic(problem, new_state),
                                            new_state.distance, counter, new_state))

    return None, expanded

def greedy_search(problem, initial_state, heuristic, lookahead_hours=2,
                  max_expansions=20000, pickup_limit=4):
    window = None if lookahead_hours is None else timedelta(hours=lookahead_hours)
    total = 0

    for w in ([window, None] if window is not None else [None]):
        solution, expanded = _greedy_pass(problem, initial_state, heuristic,
                                          w, max_expansions, pickup_limit)
        total += expanded
        if solution is not None:
            return solution, total

    return None, total
