
"""Greedy Best-First para recolha e entrega com passagem pelo armazém."""

import heapq
from collections import defaultdict
from datetime import datetime, timedelta

from src.model import State, check_solution


def heuristic01(problem, state, request):
    return problem.get_distance(state.current_location, request.pickup_location)


def heuristic02(problem, state, request):
    return (
        heuristic01(problem, state, request)
        + problem.get_distance(request.pickup_location, problem.warehouse_location)
        + problem.get_distance(problem.warehouse_location, request.delivery_location)
    )


def greedy_search(problem, initial_state, heuristic, max_expansions=100000):
    """Devolve (rotas, nos_expandidos, completa).

    Quando completa=True, as rotas passaram no check_solution.
    max_expansions evita pesquisas intermináveis; None desativa o limite.
    """
    start_time = (
        initial_state.current_time
        if isinstance(initial_state, State) else initial_state
    )

    vehicles = {v.id: v for v in problem.vehicles}
    requests = {r.id: r for r in problem.requests}
    warehouse = problem.warehouse_location
    orders = defaultdict(list)
    producers = defaultdict(list)

    for r in problem.requests:
        orders[r.order_id].append(r.id)
        producers[(r.pickup_start.date(), r.pickup_location)].append(r.id)

    stop_min = float(problem.rules.get("stop_service_min", 10))
    warehouse_min = float(problem.rules.get("warehouse_handling_min", 15))
    dwell_max = timedelta(hours=float(problem.rules.get("max_warehouse_dwell_h", 24)))
    start_day = datetime.fromisoformat(problem.rules["start_date"])
    horizon = start_day + timedelta(days=int(problem.rules["horizon_days"]))

    def departure(v):
        return start_time[v] if isinstance(start_time, dict) else start_time

    def initial():
        return {
            "routes": {v: [] for v in vehicles},
            "where": {v: warehouse for v in vehicles},
            "time": {v: departure(v) for v in vehicles},
            "weight": {v: 0.0 for v in vehicles},
            "status": {i: 0 for i in requests},
            "owner": {},
            "stored_at": {},
            "ready_at": {},
            "last_stop": {v: None for v in vehicles},
        }

    # Situação do produto: 0=produtor, 1=veículo, 2=armazém,
    # 3=veículo para entrega, 4=entregue.
    def move(state, v, operation, ids):
        ids = tuple(ids)

        if operation == "pickup":
            if any(state["status"][i] != 0 for i in ids):
                return None
            destination = requests[ids[0]].pickup_location

        elif operation == "unload":
            if any(state["status"][i] != 1 or state["owner"].get(i) != v for i in ids):
                return None
            destination = warehouse

        elif operation == "load":
            if any(state["status"][i] != 2 for i in ids):
                return None
            destination = warehouse

        elif operation == "deliver":
            if any(state["status"][i] != 3 or state["owner"].get(i) != v for i in ids):
                return None
            destination = requests[ids[0]].delivery_location

        else:
            return None

        # Produtos da mesma paragem têm de estar no mesmo local.
        if operation in ("pickup", "deliver"):
            for i in ids:
                place = (
                    requests[i].pickup_location
                    if operation == "pickup"
                    else requests[i].delivery_location
                )
                if place != destination:
                    return None

        load_change = sum(requests[i].weight_kg for i in ids)

        if operation in ("unload", "deliver"):
            load_change = -load_change

        new_weight = state["weight"][v] + load_change

        if not (-1e-6 <= new_weight <= vehicles[v].capacity_kg + 1e-6):
            return None

        try:
            minutes = problem.get_duration(state["where"][v], destination)
        except KeyError:
            return None

        arrival = state["time"][v] + timedelta(minutes=minutes)
        previous = state["last_stop"][v]

        grouped = (
            previous is not None
            and previous[0:2] == (operation, destination)
        )

        # Operações consecutivas do mesmo tipo/local: um serviço apenas.
        if grouped:
            arrival = previous[2]
            stop_ids = previous[3] + ids
        else:
            stop_ids = ids

        service_start = arrival

        if operation == "pickup":
            service_start = max(
                service_start,
                *(requests[i].pickup_start for i in stop_ids)
            )
            if any(service_start > requests[i].pickup_end for i in stop_ids):
                return None

        elif operation == "deliver":
            service_start = max(
                service_start,
                *(requests[i].delivery_start for i in stop_ids)
            )
            if any(service_start > requests[i].delivery_end for i in stop_ids):
                return None

        elif operation == "load":
            service_start = max(
                service_start,
                *(state["ready_at"][i] for i in stop_ids)
            )
            if any(
                service_start - state["stored_at"][i] > dwell_max
                for i in stop_ids
            ):
                return None

        duration = (
            stop_min
            if operation in ("pickup", "deliver")
            else warehouse_min
        )

        finished = service_start + timedelta(minutes=duration)

        # As operações e o regresso ao armazém têm de caber no horizonte.
        if (
            finished
            + timedelta(minutes=problem.get_duration(destination, warehouse))
            > horizon
        ):
            return None

        next_state = {
            "routes": {k: route[:] for k, route in state["routes"].items()},
            "where": state["where"].copy(),
            "time": state["time"].copy(),
            "weight": state["weight"].copy(),
            "status": state["status"].copy(),
            "owner": state["owner"].copy(),
            "stored_at": state["stored_at"].copy(),
            "ready_at": state["ready_at"].copy(),
            "last_stop": state["last_stop"].copy(),
        }

        next_state["where"][v] = destination
        next_state["time"][v] = finished
        next_state["weight"][v] = max(0.0, new_weight)
        next_state["last_stop"][v] = (
            operation,
            destination,
            arrival,
            stop_ids
        )

        for i in ids:
            next_state["routes"][v].append((operation, i))

            if operation == "pickup":
                next_state["status"][i] = 1
                next_state["owner"][i] = v

            elif operation == "unload":
                next_state["status"][i] = 2
                next_state["owner"].pop(i, None)
                next_state["stored_at"][i] = arrival
                next_state["ready_at"][i] = finished

            elif operation == "load":
                next_state["status"][i] = 3
                next_state["owner"][i] = v

            else:
                next_state["status"][i] = 4
                next_state["owner"].pop(i, None)

        return next_state

    # Gerar os próximos movimentos possíveis.
    def successors(state):

        for v in vehicles:

            # Recolher todos os produtos disponíveis num produtor/dia.
            for group in producers.values():
                if all(state["status"][i] == 0 for i in group):
                    candidate = move(state, v, "pickup", group)
                    if candidate is not None:
                        yield candidate

            # Descarregar tudo o que foi recolhido por este veículo.
            carrying = [
                i for i in requests
                if state["status"][i] == 1
                and state["owner"][i] == v
            ]

            if carrying:
                candidate = move(state, v, "unload", carrying)
                if candidate is not None:
                    yield candidate

            # Carregar uma encomenda inteira.
            # Várias podem ser carregadas seguidamente.
            for group in orders.values():
                if all(state["status"][i] == 2 for i in group):
                    candidate = move(state, v, "load", group)
                    if candidate is not None:
                        yield candidate

            # Entregar uma encomenda inteira.
            # Depois pode seguir para outro cliente.
            for group in orders.values():
                if all(
                    state["status"][i] == 3
                    and state["owner"].get(i) == v
                    for i in group
                ):
                    candidate = move(state, v, "deliver", group)
                    if candidate is not None:
                        yield candidate

    # Identificar estados pelo presente, não pela ordem histórica dos veículos.
    def state_key(state):
        return (
            tuple(
                (state["status"][i], state["owner"].get(i))
                for i in requests
            ),
            tuple(
                (
                    state["where"][v],
                    state["time"][v],
                    state["last_stop"][v]
                )
                for v in vehicles
            ),
            tuple(
                (
                    i,
                    state["stored_at"][i],
                    state["ready_at"][i]
                )
                for i in requests
                if state["status"][i] == 2
            ),
        )

    # Priorizar tarefas urgentes, para não deixar passar prazos.
    deadlines = []

    for group in producers.values():
        deadlines.append((
            min(requests[i].pickup_end for i in group),
            "pickup",
            group
        ))

    for group in orders.values():
        deadlines.append((
            min(requests[i].delivery_end for i in group),
            "deliver",
            group
        ))

    deadlines.sort(key=lambda item: item[0])

    def h(state):
        urgency = 0.0

        for index, (_, kind, group) in enumerate(deadlines):
            finished = (
                all(state["status"][i] >= 1 for i in group)
                if kind == "pickup"
                else all(state["status"][i] == 4 for i in group)
            )

            if not finished:
                urgency += (len(deadlines) - index) ** 1.5

        # A função H1/H2 fornecida escolhe os locais mais promissores.
        distance = 0.0

        for r in problem.requests:
            status = state["status"][r.id]

            if status == 0:
                estimates = []

                for v in vehicles:
                    vehicle_state = State(
                        v,
                        state["where"][v],
                        state["time"][v]
                    )
                    estimates.append(
                        heuristic(problem, vehicle_state, r)
                    )

                distance += min(estimates)

            elif status == 1:
                v = state["owner"][r.id]
                distance += problem.get_distance(
                    state["where"][v],
                    warehouse
                )

            elif status == 2:
                distance += problem.get_distance(
                    warehouse,
                    r.delivery_location
                )

            elif status == 3:
                v = state["owner"][r.id]
                distance += problem.get_distance(
                    state["where"][v],
                    r.delivery_location
                )

        return (  
            round(distance, 3),  # 1.º: heurística H1 ou H2
            round(urgency, 6)    # 2.º: urgência para desempate
        )

    # Estimativa otimista para rejeitar estados com prazos já impossíveis.
    def feasible_future(state):

        ready_pickup = {}

        for group in producers.values():

            if state["status"][group[0]] != 0:
                continue

            r = requests[group[0]]

            arrival = min(
                max(
                    state["time"][v]
                    + timedelta(
                        minutes=problem.get_duration(
                            state["where"][v],
                            r.pickup_location
                        )
                    ),
                    max(requests[i].pickup_start for i in group)
                )
                for v in vehicles
            )

            if arrival > min(requests[i].pickup_end for i in group):
                return False

            ready_pickup[
                (r.pickup_start.date(), r.pickup_location)
            ] = arrival

        for group in orders.values():

            if all(state["status"][i] == 4 for i in group):
                continue

            latest = min(
                requests[i].delivery_end
                for i in group
            )

            ready = []

            for i in group:
                r = requests[i]
                status = state["status"][i]

                if status == 0:
                    arrival = ready_pickup[
                        (r.pickup_start.date(), r.pickup_location)
                    ]
                    ready.append(
                        arrival
                        + timedelta(
                            minutes=(
                                stop_min
                                + problem.get_duration(
                                    r.pickup_location,
                                    warehouse
                                )
                                + warehouse_min
                            )
                        )
                    )

                elif status == 1:
                    v = state["owner"][i]
                    ready.append(
                        state["time"][v]
                        + timedelta(
                            minutes=(
                                problem.get_duration(
                                    state["where"][v],
                                    warehouse
                                )
                                + warehouse_min
                            )
                        )
                    )

                elif status == 2:
                    ready.append(state["ready_at"][i])

                    earliest_load = min(
                        max(
                            state["time"][v]
                            + timedelta(
                                minutes=problem.get_duration(
                                    state["where"][v],
                                    warehouse
                                )
                            ),
                            state["ready_at"][i]
                        )
                        for v in vehicles
                    )

                    if (
                        earliest_load - state["stored_at"][i]
                        > dwell_max
                    ):
                        return False

                elif status == 3:
                    v = state["owner"][i]

                    if max(
                        state["time"][v]
                        + timedelta(
                            minutes=problem.get_duration(
                                state["where"][v],
                                r.delivery_location
                            )
                        ),
                        r.delivery_start
                    ) > r.delivery_end:
                        return False

            if ready:
                all_ready = max(ready)

                earliest_load = min(
                    max(
                        state["time"][v]
                        + timedelta(
                            minutes=problem.get_duration(
                                state["where"][v],
                                warehouse
                            )
                        ),
                        all_ready
                    )
                    for v in vehicles
                )

                optimistic_delivery = (
                    earliest_load
                    + timedelta(
                        minutes=(
                            warehouse_min
                            + problem.get_duration(
                                warehouse,
                                requests[group[0]].delivery_location
                            )
                        )
                    )
                )

                if optimistic_delivery > latest:
                    return False

        return True

    # Pesquisa Greedy: escolher sempre o estado com menor h(n).
    queue = [(h(initial()), 0, initial())]

    visited = set()
    counter = 0
    expanded = 0

    while queue:

        _, _, state = heapq.heappop(queue)

        key = state_key(state)

        if key in visited:
            continue

        visited.add(key)

        # Objetivo: todos os produtos entregues.
        if all(
            value == 4
            for value in state["status"].values()
        ):
            valid, _, _ = check_solution(
                problem,
                state["routes"],
                start_time
            )

            if valid:
                return state["routes"], expanded, True

        # Limite configurável de expansões.
        if (
            max_expansions is not None
            and expanded >= max_expansions
        ):
            break

        expanded += 1

        # Gerar e avaliar sucessores.
        for candidate in successors(state):

            candidate_key = state_key(candidate)

            if candidate_key in visited:
                continue

            if not feasible_future(candidate):
                continue

            counter += 1

            heapq.heappush(
                queue,
                (
                    h(candidate),
                    counter,
                    candidate
                )
            )

    # Nunca apresentar um plano incompleto como válido.
    return {
        v: [] for v in vehicles
    }, expanded, False
