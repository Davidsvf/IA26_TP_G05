
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from src.data_loader import load_data


@dataclass
class Vehicle:
    id: int
    capacity_kg: float
    center_id: int


@dataclass
class Request:
    id: int
    order_id: int
    pickup_location: int
    delivery_location: int
    weight_kg: float
    pickup_start: datetime
    pickup_end: datetime
    delivery_start: datetime
    delivery_end: datetime


@dataclass
class State:
    vehicle_id: int
    current_location: int
    current_time: datetime
    current_load: float = 0.0
    picked_requests: set = field(default_factory=set)
    warehouse_requests: set = field(default_factory=set)
    delivered_requests: set = field(default_factory=set)
    total_distance: float = 0.0


class Problem:

    def __init__(self, requests, vehicles, distances, durations,
                 warehouse_location, rules=None):

        self.requests = requests
        self.vehicles = vehicles
        self.distances = distances
        self.durations = durations
        self.warehouse_location = warehouse_location
        self.rules = rules or {}

        self.warehouse_required = (
            "warehouse" in self.rules.get("flow", "").lower()
        )

    def get_distance(self, a, b):
        if a == b:
            return 0.0
        return self.distances[a, b]

    def get_duration(self, a, b):
        if a == b:
            return 0.0
        return self.durations[a, b]

    @classmethod
    def from_csv(cls, folder):

        data = load_data(folder)

        producers = {}
        deliveries = {}
        warehouse = None

        for _, row in data["locations"].iterrows():

            location = int(row["location_id"])
            ref = int(row["ref_id"])

            if row["location_type"] == "producer":
                producers[ref] = location

            elif row["location_type"] == "delivery_location":
                deliveries[ref] = location

            elif row["location_type"] == "collection_center":
                warehouse = location

        products = data["products"].set_index("product_id")
        orders = data["orders"].set_index("order_id")

        requests = []

        for _, row in data["order_items"].iterrows():

            product = products.loc[int(row["product_id"])]
            order = orders.loc[int(row["order_id"])]

            if "weight_kg" in row.index:
                weight = float(row["weight_kg"])
            else:
                weight = (
                    float(row["quantity"])
                    * float(product["unit_weight_kg"])
                )

            requests.append(Request(
                id=int(row["order_item_id"]),
                order_id=int(row["order_id"]),
                pickup_location=producers[
                    int(product["producer_id"])
                ],
                delivery_location=deliveries[
                    int(order["delivery_location_id"])
                ],
                weight_kg=weight,
                pickup_start=datetime.fromisoformat(
                    str(row["availability_start"])
                ),
                pickup_end=datetime.fromisoformat(
                    str(row["availability_end"])
                ),
                delivery_start=datetime.fromisoformat(
                    str(row["delivery_start"])
                ),
                delivery_end=datetime.fromisoformat(
                    str(row["delivery_end"])
                )
            ))

        vehicles = []

        for _, row in data["vehicles"].iterrows():
            vehicles.append(Vehicle(
                id=int(row["vehicle_id"]),
                capacity_kg=float(row["capacity_kg"]),
                center_id=int(row["center_id"])
            ))

        distances = {}
        durations = {}

        for _, row in data["distance_matrix"].iterrows():

            pair = (
                int(row["origin_location_id"]),
                int(row["destination_location_id"])
            )

            distances[pair] = float(row["distance_km"])
            durations[pair] = float(row["duration_min"])

        rules = {}

        for _, row in data["rules"].iterrows():
            rules[str(row["rule"])] = str(row["value"])

        return cls(
            requests, vehicles, distances,
            durations, warehouse, rules
        )


def route_cost(problem, locations):
    total = 0.0

    for i in range(len(locations) - 1):
        total += problem.get_distance(
            locations[i], locations[i + 1]
        )

    return round(total, 2)


def check_capacity(vehicle, load):
    return -0.000001 <= load <= vehicle.capacity_kg + 0.000001


def check_time_window(arrival, start, end):
    return max(arrival, start) <= end


def check_solution(problem, solution, start_time, check_all=True):
    """
    Valida as rotas dos veiculos.

    Produtos da mesma encomenda devem ser carregados
    e entregues juntos.
    """

    requests = {r.id: r for r in problem.requests}
    vehicles = {v.id: v for v in problem.vehicles}
    warehouse = problem.warehouse_location

    if warehouse is None:
        return False, 0.0, "Armazém não encontrado"

    if set(solution) != set(vehicles):
        return False, 0.0, "Veículos inválidos"

    # Agrupar produtos por encomenda
    orders = {}

    for request in problem.requests:
        orders.setdefault(request.order_id, set()).add(request.id)

    # Agrupar operacoes consecutivas da mesma paragem
    stops = {}
    seen = set()

    for vehicle_id, route in solution.items():

        stops[vehicle_id] = []

        for action, request_id in route:

            if request_id not in requests:
                return False, 0.0, "Produto desconhecido"

            if action not in ("pickup", "unload", "load", "deliver"):
                return False, 0.0, "Operação desconhecida"

            if (action, request_id) in seen:
                return False, 0.0, "Operação repetida"

            seen.add((action, request_id))
            request = requests[request_id]

            if action == "pickup":
                destination = request.pickup_location
            elif action == "deliver":
                destination = request.delivery_location
            else:
                destination = warehouse

            if (
                stops[vehicle_id]
                and stops[vehicle_id][-1][0] == action
                and stops[vehicle_id][-1][1] == destination
            ):
                stops[vehicle_id][-1][2].append(request_id)

            else:
                stops[vehicle_id].append(
                    (action, destination, [request_id])
                )

    # Verificar se cada encomenda incluida esta completa
    active = {request_id for _, request_id in seen}

    if check_all and active != set(requests):
        return False, 0.0, "Encomendas por entregar"

    for request_id in active:

        request = requests[request_id]

        if not orders[request.order_id].issubset(active):
            return False, 0.0, "Encomenda incompleta"

        for action in ("pickup", "unload", "load", "deliver"):
            if (action, request_id) not in seen:
                return False, 0.0, "Faltam operações"

    # Carregar e entregar sempre a encomenda completa
    for route_stops in stops.values():

        for action, destination, ids in route_stops:

            if action in ("load", "deliver"):

                group = set(ids)

                for request_id in ids:
                    order_id = requests[request_id].order_id

                    if not orders[order_id].issubset(group):
                        return False, 0.0, "Encomenda dividida"

    # Regras temporais
    stop_min = float(problem.rules.get("stop_service_min", 0))
    warehouse_min = float(
        problem.rules.get("warehouse_handling_min", 0)
    )

    max_dwell = problem.rules.get("max_warehouse_dwell_h")

    if max_dwell is not None:
        max_dwell = float(max_dwell)

    first_day = datetime.fromisoformat(
        problem.rules.get("start_date", "1900-01-01")
    )

    horizon = first_day + timedelta(
        days=int(problem.rules.get("horizon_days", 99999))
    )

    # Estado de cada veiculo
    times = {}
    locations = {}
    loads = {}
    next_stop = {}

    for vehicle_id in vehicles:

        if isinstance(start_time, dict):
            departure = start_time[vehicle_id]
        else:
            departure = start_time

        if not first_day <= departure < horizon:
            return False, 0.0, "Partida fora do planeamento"

        times[vehicle_id] = departure
        locations[vehicle_id] = warehouse
        loads[vehicle_id] = 0.0
        next_stop[vehicle_id] = 0

    # Estado de cada produto
    status = {request_id: "new" for request_id in requests}
    owner = {}
    stored_at = {}
    available_at = {}

    total_distance = 0.0

    # Simulacao das rotas
    while any(
        next_stop[v] < len(stops[v])
        for v in vehicles
    ):

        best = None

        for vehicle_id in vehicles:

            index = next_stop[vehicle_id]

            if index >= len(stops[vehicle_id]):
                continue

            action, destination, ids = stops[vehicle_id][index]

            # Verificar se as operacoes ja podem acontecer
            ready = True

            for request_id in ids:

                if action == "pickup":
                    allowed = status[request_id] == "new"

                elif action == "unload":
                    allowed = (
                        status[request_id] == "picked"
                        and owner.get(request_id) == vehicle_id
                    )

                elif action == "load":
                    allowed = status[request_id] == "stored"

                else:
                    allowed = (
                        status[request_id] == "loaded"
                        and owner.get(request_id) == vehicle_id
                    )

                if not allowed:
                    ready = False

            if not ready:
                continue

            # Viagem ate a paragem
            origin = locations[vehicle_id]

            try:
                distance = problem.get_distance(origin, destination)
                duration = problem.get_duration(origin, destination)
            except KeyError:
                return False, 0.0, "Percurso inexistente"

            arrival = times[vehicle_id] + timedelta(
                minutes=duration
            )

            service_start = arrival

            # Janelas de recolha
            if action == "pickup":

                service_start = max(
                    arrival,
                    *[requests[i].pickup_start for i in ids]
                )

                if any(
                    service_start > requests[i].pickup_end
                    for i in ids
                ):
                    continue

            # Janelas de entrega
            elif action == "deliver":

                service_start = max(
                    arrival,
                    *[requests[i].delivery_start for i in ids]
                )

                if any(
                    service_start > requests[i].delivery_end
                    for i in ids
                ):
                    continue

            # Produtos so podem ser carregados apos a descarga
            elif action == "load":

                service_start = max(
                    arrival,
                    *[available_at[i] for i in ids]
                )

            if action in ("pickup", "deliver"):
                service_min = stop_min
            else:
                service_min = warehouse_min

            service_end = service_start + timedelta(
                minutes=service_min
            )

            candidate = (
                service_end, vehicle_id, action,
                destination, ids, arrival,
                service_start, distance
            )

            # Processar primeiro a operacao disponivel
            # que termina mais cedo
            if best is None or service_end < best[0]:
                best = candidate

        if best is None:
            return (
                False,
                round(total_distance, 2),
                "Ordem ou janelas temporais inválidas"
            )

        (
            service_end, vehicle_id, action, destination,
            ids, arrival, service_start, distance
        ) = best

        vehicle = vehicles[vehicle_id]

        weight = sum(
            requests[i].weight_kg for i in ids
        )

        # Atualizar e verificar a carga
        if action in ("pickup", "load"):
            new_load = loads[vehicle_id] + weight
        else:
            new_load = loads[vehicle_id] - weight

        if not check_capacity(vehicle, new_load):
            return (
                False,
                round(total_distance, 2),
                "Capacidade excedida"
            )

        # Permanencia maxima de 24 horas no armazem
        if action == "load" and max_dwell is not None:

            for request_id in ids:

                dwell = service_start - stored_at[request_id]

                if dwell > timedelta(hours=max_dwell):
                    return (
                        False,
                        round(total_distance, 2),
                        "Tempo no armazém excedido"
                    )

        if service_end > horizon:
            return (
                False,
                round(total_distance, 2),
                "Operação fora do planeamento"
            )

        # Atualizar o estado dos produtos
        for request_id in ids:

            if action == "pickup":
                status[request_id] = "picked"
                owner[request_id] = vehicle_id

            elif action == "unload":
                status[request_id] = "stored"

                # Contagem das 24h desde a chegada
                stored_at[request_id] = arrival

                # Disponivel apos terminar a descarga
                available_at[request_id] = service_end

                del owner[request_id]

            elif action == "load":
                status[request_id] = "loaded"
                owner[request_id] = vehicle_id

            else:
                status[request_id] = "delivered"
                del owner[request_id]

        loads[vehicle_id] = new_load
        times[vehicle_id] = service_end
        locations[vehicle_id] = destination

        total_distance += distance
        next_stop[vehicle_id] += 1

    # Todos os produtos atribuidos devem estar entregues
    for request_id in requests:

        if status[request_id] not in ("new", "delivered"):
            return (
                False,
                round(total_distance, 2),
                "Produtos por entregar"
            )

        if check_all and status[request_id] != "delivered":
            return (
                False,
                round(total_distance, 2),
                "Encomendas por entregar"
            )

    # Regresso dos veiculos ao armazem
    for vehicle_id in vehicles:

        origin = locations[vehicle_id]

        try:
            total_distance += problem.get_distance(origin, warehouse)
            times[vehicle_id] += timedelta(
                minutes=problem.get_duration(origin, warehouse)
            )
        except KeyError:
            return (
                False,
                round(total_distance, 2),
                "Regresso sem percurso"
            )

        if times[vehicle_id] > horizon:
            return (
                False,
                round(total_distance, 2),
                "Regresso fora do planeamento"
            )

    return (
        True,
        round(total_distance, 2),
        "Solução válida" if check_all else "Rotas válidas"
    )


def check_route(problem, vehicle, route, start_time):
    """Valida uma rota individual, sem transferencias."""

    solution = {v.id: [] for v in problem.vehicles}

    if vehicle.id not in solution:
        return False, 0.0, "Veículo desconhecido"

    solution[vehicle.id] = route

    return check_solution(
        problem,
        solution,
        start_time,
        check_all=False
    )
