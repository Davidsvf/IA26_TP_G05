"""Modelo do problema de recolha e entrega de produtos locais.

Contém as entidades (Vehicle, Request, State, Problem), a função de custo e
a verificação de rotas. Nada aqui depende de um dataset em particular: tudo
é lido dos CSV (incluindo rules.csv), por isso o mesmo código serve para
data_tiny, data_small ou qualquer outro dataset com o mesmo formato.

Fluxo de cada item:   produtor -> armazém -> local de entrega
    pickup   recolher o item no produtor
    unload   descarregá-lo no armazém
    load     carregar a encomenda (inteira) no armazém
    deliver  entregar a encomenda (inteira) ao cliente

CUSTO = distância total, em km, percorrida por todos os veículos (incluindo
o regresso final de cada veículo ao armazém). O tempo NÃO entra no custo:
só serve para validar janelas, horizonte e permanência no armazém.
"""

from collections import defaultdict, namedtuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.data_loader import load_data

# Operações possíveis numa rota
PICKUP, UNLOAD, LOAD, DELIVER = "pickup", "unload", "load", "deliver"
OPERATIONS = (PICKUP, UNLOAD, LOAD, DELIVER)

# Situação de cada item
NEW, PICKED, STORED, LOADED, DELIVERED = range(5)

# Tipo de motivo quando uma paragem não pode ser executada
WAIT = "wait"      # ainda falta outra operação (ex.: outro veículo descarregar)
BROKEN = "rule"    # viola uma regra do problema


# Entidades

@dataclass
class Vehicle:
    id: int
    capacity_kg: float
    center_id: int          # armazém de onde parte e onde termina


@dataclass
class Request:
    """Um item de encomenda (linha de order_items.csv)."""
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
class PickupGroup:
    """Itens do mesmo produtor com uma janela de recolha comum.

    Uma única paragem no produtor recolhe todos os itens do grupo.
    """
    location: int
    ids: tuple
    start: datetime         # abertura comum (a mais tardia)
    end: datetime           # fecho comum (o mais cedo)
    weight_kg: float


class State:
    """Estado do planeamento: onde está cada veículo e em que ponto está cada item.

    Os veículos e os itens são identificados pelos ids dos CSV.
    """

    def __init__(self, time, place, load, stop, status, owner,
                 stored_at, ready_at, routes, distance=0.0, last=None):
        self.time = time              # veículo -> hora a que fica livre
        self.place = place            # veículo -> localização atual
        self.load = load              # veículo -> kg a bordo
        self.stop = stop              # veículo -> última paragem (ver apply_stop)
        self.status = status          # item -> NEW/PICKED/STORED/LOADED/DELIVERED
        self.owner = owner            # item -> veículo que o transporta
        self.stored_at = stored_at    # item -> chegada ao armazém (limite de 24h)
        self.ready_at = ready_at      # item -> fim da descarga no armazém
        self.routes = routes          # veículo -> [(operação, item), ...]
        self.distance = distance      # g(n): km percorridos até aqui
        self.last = last              # descrição da última paragem (para registos)

    def copy(self):
        return State(
            self.time.copy(), self.place.copy(), self.load.copy(),
            self.stop.copy(), self.status.copy(), self.owner.copy(),
            self.stored_at.copy(), self.ready_at.copy(),
            {v: route[:] for v, route in self.routes.items()},
            self.distance, self.last
        )

    def key(self):
        """Identifica o estado pelo presente (não pela ordem em que lá se chegou)."""
        status = self.status
        return (
            tuple(self.time.values()),
            tuple(self.place.values()),
            tuple(round(x, 6) for x in self.load.values()),
            tuple(
                None if s is None else (s[0], s[1], s[3], tuple(sorted(s[4])))
                for s in self.stop.values()
            ),
            tuple(status.values()),
            tuple(self.owner.values()),
            tuple(
                (self.stored_at[i], self.ready_at[i]) if status[i] == STORED else None
                for i in status
            ),
        )

# Problema

def build_pickup_groups(requests):
    """Agrupa os itens de cada produtor em grupos com janela de recolha comum.

    Em cada produtor ordena os itens pelo fecho da janela e junta-lhes todos
    os que já estão abertos nesse instante. Se todos os itens de um produtor
    (num dia) partilharem uma janela, fica um grupo por produtor/dia.
    """
    by_place = defaultdict(list)
    for r in requests:
        by_place[r.pickup_location].append(r)

    groups = []
    for place, items in by_place.items():
        items = sorted(items, key=lambda r: (r.pickup_end, r.id))
        while items:
            deadline = items[0].pickup_end
            chosen = [r for r in items if r.pickup_start <= deadline]
            items = [r for r in items if r.pickup_start > deadline]
            groups.append(PickupGroup(
                location=place,
                ids=tuple(r.id for r in chosen),
                start=max(r.pickup_start for r in chosen),
                end=min(r.pickup_end for r in chosen),
                weight_kg=sum(r.weight_kg for r in chosen),
            ))

    groups.sort(key=lambda g: (g.end, g.location))
    return groups


class Problem:

    def __init__(self, requests, vehicles, distances, durations,
                 warehouse_location, rules=None, names=None,
                 warehouse_capacity=None):

        self.requests = list(requests)
        self.vehicles = list(vehicles)
        self.distances = distances
        self.durations = durations
        self.warehouse_location = warehouse_location
        self.warehouse_capacity = warehouse_capacity
        self.rules = rules or {}
        self.names = names or {}

        self.request = {r.id: r for r in self.requests}
        self.vehicle = {v.id: v for v in self.vehicles}

        # Itens de cada encomenda
        orders = defaultdict(list)
        for r in self.requests:
            orders[r.order_id].append(r.id)
        self.orders = {o: tuple(ids) for o, ids in orders.items()}

        # Grupos de recolha por produtor
        self.pickup_groups = build_pickup_groups(self.requests)
        self.group_of = {
            i: n for n, g in enumerate(self.pickup_groups) for i in g.ids
        }

        # Regras (lidas de rules.csv; os valores por omissão são os dos datasets)
        self.stop_service = timedelta(
            minutes=float(self.rules.get("stop_service_min", 10)))
        self.warehouse_service = timedelta(
            minutes=float(self.rules.get("warehouse_handling_min", 15)))
        self.dwell_max = timedelta(
            hours=float(self.rules.get("max_warehouse_dwell_h", 24)))

        if "start_date" in self.rules:
            self.start_time = datetime.fromisoformat(self.rules["start_date"])
        else:
            first = min(r.pickup_start for r in self.requests)
            self.start_time = first.replace(hour=0, minute=0, second=0,
                                            microsecond=0)

        if "horizon_days" in self.rules:
            days = int(self.rules["horizon_days"])
        else:
            last = max(r.delivery_end for r in self.requests)
            days = (last.date() - self.start_time.date()).days + 1
        self.horizon = self.start_time + timedelta(days=days)

        self._travel = {}
        self._lower = self._shortest_times()
        self._lower_km = self._shortest_km()

    # ---------------- distâncias e tempos ----------------

    def get_distance(self, a, b):
        if a == b:
            return 0.0
        return self.distances[a, b]

    def get_duration(self, a, b):
        """Duração da viagem em minutos."""
        if a == b:
            return 0.0
        return self.durations[a, b]

    def travel_time(self, a, b):
        """Duração da viagem como timedelta (com cache)."""
        pair = (a, b)
        if pair not in self._travel:
            self._travel[pair] = timedelta(minutes=self.get_duration(a, b))
        return self._travel[pair]

    def _shortest_times(self):
        """Menor tempo possível entre locais (Floyd-Warshall sobre a matriz).

        Serve para a poda de viabilidade: é um limite inferior, por isso nunca
        rejeita um estado que ainda tenha solução, mesmo que a matriz tenha
        uma ligação indireta mais rápida do que a direta.
        """
        places = {self.warehouse_location}
        places |= {r.pickup_location for r in self.requests}
        places |= {r.delivery_location for r in self.requests}
        places = sorted(places)
        inf = float("inf")
        best = {(a, b): (0.0 if a == b else self.durations.get((a, b), inf))
                for a in places for b in places}
        for k in places:
            for a in places:
                for b in places:
                    if best[a, k] + best[k, b] < best[a, b]:
                        best[a, b] = best[a, k] + best[k, b]
        return {pair: timedelta(minutes=m) for pair, m in best.items() if m < inf}

    def _shortest_km(self):
        """Menor distância (km) possível entre locais (Floyd-Warshall)."""
        places = sorted({self.warehouse_location}
                        | {r.pickup_location for r in self.requests}
                        | {r.delivery_location for r in self.requests})
        inf = float("inf")
        best = {(a, b): (0.0 if a == b else self.distances.get((a, b), inf))
                for a in places for b in places}
        for k in places:
            for a in places:
                for b in places:
                    if best[a, k] + best[k, b] < best[a, b]:
                        best[a, b] = best[a, k] + best[k, b]
        return best

    def lower_distance(self, a, b):
        """Limite inferior da distância entre dois locais (km), para heurísticas
        admissíveis. Infinito se não houver percurso."""
        return self._lower_km.get((a, b), float("inf"))

    def lower_travel(self, a, b):
        """Limite inferior do tempo de viagem (None se não houver percurso)."""
        return self._lower.get((a, b))

    def name(self, location):
        return self.names.get(location, f"local {location}")

    # ---------------- estado inicial ----------------

    def initial_state(self, start_time=None):
        """Todos os veículos no armazém, à hora de partida, sem nada feito.

        start_time pode ser uma data/hora (igual para todos) ou um dicionário
        veículo -> data/hora. Por omissão usa o início do horizonte (rules.csv).
        """
        if start_time is None:
            start_time = self.start_time

        times = {}
        for v in self.vehicles:
            t = start_time[v.id] if isinstance(start_time, dict) else start_time
            times[v.id] = t

        ids = [r.id for r in self.requests]
        return State(
            time=times,
            place={v.id: self.warehouse_location for v in self.vehicles},
            load={v.id: 0.0 for v in self.vehicles},
            stop={v.id: None for v in self.vehicles},
            status={i: NEW for i in ids},
            owner={i: None for i in ids},
            stored_at={i: None for i in ids},
            ready_at={i: None for i in ids},
            routes={v.id: [] for v in self.vehicles},
        )

    def stop_place(self, action, request_id):
        """Localização onde a operação é feita."""
        r = self.request[request_id]
        if action == PICKUP:
            return r.pickup_location
        if action == DELIVER:
            return r.delivery_location
        return self.warehouse_location

    def subset(self, order_ids):
        """Problema só com algumas encomendas (útil para testar com menos dados).

        order_ids pode ser uma lista de ids ou um inteiro N (as primeiras N).
        """
        if isinstance(order_ids, int):
            order_ids = list(self.orders)[:order_ids]
        wanted = set(order_ids)
        return Problem(
            [r for r in self.requests if r.order_id in wanted],
            self.vehicles, self.distances, self.durations,
            self.warehouse_location, self.rules, self.names,
            self.warehouse_capacity
        )

    def validate(self):
        """Devolve uma lista de avisos sobre a coerência da instância."""
        issues = []
        biggest = max(v.capacity_kg for v in self.vehicles)

        for g in self.pickup_groups:
            if g.weight_kg > biggest:
                issues.append(f"Recolha em {self.name(g.location)} pesa "
                              f"{g.weight_kg:.1f} kg (> {biggest} kg)")

        for order, ids in self.orders.items():
            weight = sum(self.request[i].weight_kg for i in ids)
            if weight > biggest:
                issues.append(f"Encomenda {order} pesa {weight:.1f} kg "
                              f"(> {biggest} kg)")
            start = max(self.request[i].delivery_start for i in ids)
            end = min(self.request[i].delivery_end for i in ids)
            if start > end:
                issues.append(f"Encomenda {order} sem janela de entrega comum")

        places = {self.warehouse_location}
        places |= {r.pickup_location for r in self.requests}
        places |= {r.delivery_location for r in self.requests}
        for a in places:
            for b in places:
                if a != b and (a, b) not in self.distances:
                    issues.append(f"Falta a distância {a} -> {b}")

        total = sum(r.weight_kg for r in self.requests)
        if self.warehouse_capacity is not None and total > self.warehouse_capacity:
            issues.append(f"Peso total ({total:.1f} kg) excede a capacidade "
                          f"do armazém ({self.warehouse_capacity} kg), "
                          f"que não é verificada")
        return issues

    # ---------------- leitura dos CSV ----------------

    @classmethod
    def from_csv(cls, folder):

        data = load_data(folder)

        producers = {}      # producer_id -> location_id
        deliveries = {}     # delivery_location_id -> location_id
        centers = {}        # center_id -> location_id
        names = {}

        for _, row in data["locations"].iterrows():

            location = int(row["location_id"])
            ref = int(row["ref_id"])
            names[location] = str(row["name"])

            if row["location_type"] == "producer":
                producers[ref] = location
            elif row["location_type"] == "delivery_location":
                deliveries[ref] = location
            elif row["location_type"] == "collection_center":
                centers[ref] = location

        vehicles = [
            Vehicle(
                id=int(row["vehicle_id"]),
                capacity_kg=float(row["capacity_kg"]),
                center_id=int(row["center_id"]),
            )
            for _, row in data["vehicles"].iterrows()
        ]

        # O armazém é identificado pelo id do centro de recolha dos veículos
        used = {v.center_id for v in vehicles}
        if len(used) != 1:
            raise ValueError(
                "Só é suportado um armazém; veículos usam os centros "
                f"{sorted(used)}")
        center_id = used.pop()
        if center_id not in centers:
            raise ValueError(f"Centro de recolha {center_id} não existe em locations.csv")
        warehouse = centers[center_id]

        capacity = None
        if "collection_centers" in data:
            table = data["collection_centers"].set_index("center_id")
            if center_id in table.index and "capacity_kg" in table.columns:
                capacity = float(table.loc[center_id, "capacity_kg"])

        products = data["products"].set_index("product_id")
        orders = data["orders"].set_index("order_id")

        requests = []

        for _, row in data["order_items"].iterrows():

            product = products.loc[int(row["product_id"])]
            order = orders.loc[int(row["order_id"])]

            if "weight_kg" in row.index:
                weight = float(row["weight_kg"])
            else:
                weight = float(row["quantity"]) * float(product["unit_weight_kg"])

            requests.append(Request(
                id=int(row["order_item_id"]),
                order_id=int(row["order_id"]),
                pickup_location=producers[int(product["producer_id"])],
                delivery_location=deliveries[int(order["delivery_location_id"])],
                weight_kg=weight,
                pickup_start=datetime.fromisoformat(str(row["availability_start"])),
                pickup_end=datetime.fromisoformat(str(row["availability_end"])),
                delivery_start=datetime.fromisoformat(str(row["delivery_start"])),
                delivery_end=datetime.fromisoformat(str(row["delivery_end"])),
            ))

        distances = {}
        durations = {}

        for _, row in data["distance_matrix"].iterrows():
            pair = (int(row["origin_location_id"]),
                    int(row["destination_location_id"]))
            distances[pair] = float(row["distance_km"])
            durations[pair] = float(row["duration_min"])

        rules = {}
        if "rules" in data:
            for _, row in data["rules"].iterrows():
                rules[str(row["rule"])] = str(row["value"])

        return cls(requests, vehicles, distances, durations,
                   warehouse, rules, names, capacity)


# Custo e verificações elementares

def route_cost(problem, locations):
    """Custo de um percurso: soma das distâncias (km) entre localizações."""
    total = 0.0
    for i in range(len(locations) - 1):
        total += problem.get_distance(locations[i], locations[i + 1])
    return round(total, 2)


def check_capacity(vehicle, load):
    return -0.000001 <= load <= vehicle.capacity_kg + 0.000001


def check_time_window(arrival, start, end):
    """O serviço pode começar dentro da janela [start, end]? (espera se cedo)"""
    return max(arrival, start) <= end


# Regras de uma paragem 

def apply_stop(problem, state, vehicle_id, action, ids):
    """Faz o veículo ir ao local da operação e executá-la sobre os itens ids.

    Devolve (novo_estado, None) se for possível, ou (None, (tipo, motivo)).
    É aqui que estão TODAS as regras: ordem das operações, capacidade,
    janelas de recolha/entrega, 24h no armazém, tempos de serviço, encomendas
    carregadas e entregues por inteiro e horizonte de planeamento.

    Se a última paragem do veículo foi do mesmo tipo e no mesmo local, tenta-se
    juntar a operação a essa paragem (serviço feito uma só vez). Se juntar
    violar uma janela ou as 24h, faz-se uma paragem nova no mesmo local.
    """
    ids = tuple(ids)
    new, why = _apply(problem, state, vehicle_id, action, ids, True)

    if new is None and why[0] == BROKEN:
        stop = state.stop[vehicle_id]
        if stop is not None and stop[0] == action and action in OPERATIONS \
                and stop[1] == problem.stop_place(action, ids[0]):
            retry = _apply(problem, state, vehicle_id, action, ids, False)
            if retry[0] is not None:
                return retry
    return new, why


def _apply(problem, state, vehicle_id, action, ids, allow_extend):
    request = problem.request
    warehouse = problem.warehouse_location
    status, owner = state.status, state.owner

    # 1) Situação dos itens e local da operação
    if action == PICKUP:
        if any(status[i] != NEW for i in ids):
            return None, (WAIT, "item já recolhido")
    elif action == UNLOAD:
        if any(status[i] != PICKED or owner[i] != vehicle_id for i in ids):
            return None, (WAIT, "item não está neste veículo")
    elif action == LOAD:
        if any(status[i] != STORED for i in ids):
            return None, (WAIT, "item ainda não está no armazém")
    elif action == DELIVER:
        if any(status[i] != LOADED or owner[i] != vehicle_id for i in ids):
            return None, (WAIT, "item não foi carregado neste veículo")
    else:
        return None, (BROKEN, "operação desconhecida")

    place = problem.stop_place(action, ids[0])
    if any(problem.stop_place(action, i) != place for i in ids):
        return None, (BROKEN, "itens de locais diferentes na mesma paragem")

    # A encomenda é carregada e entregue por inteiro
    if action in (LOAD, DELIVER):
        chosen = set(ids)
        for order in {request[i].order_id for i in ids}:
            if not set(problem.orders[order]) <= chosen:
                return None, (BROKEN, f"encomenda {order} dividida")

    # 2) Chegada: junta-se à paragem anterior ou viaja até ao local
    stop = state.stop[vehicle_id]
    extend = (allow_extend and stop is not None
              and stop[0] == action and stop[1] == place)

    try:
        if extend:
            arrival, members, leg = stop[2], stop[4] + ids, 0.0
            stop_leg = stop[5]
        else:
            origin = state.place[vehicle_id]
            arrival = state.time[vehicle_id] + problem.travel_time(origin, place)
            members = ids
            leg = stop_leg = problem.get_distance(origin, place)
        back = problem.travel_time(place, warehouse)
    except KeyError:
        return None, (BROKEN, "percurso inexistente na matriz de distâncias")

    # 3) Início do serviço (espera pela abertura da janela) e janelas
    if action == PICKUP:
        window = (max(request[i].pickup_start for i in members),
                  min(request[i].pickup_end for i in members))
        if not check_time_window(arrival, *window):
            return None, (BROKEN, "janela de recolha ultrapassada")
        start = max(arrival, window[0])

    elif action == DELIVER:
        window = (max(request[i].delivery_start for i in members),
                  min(request[i].delivery_end for i in members))
        if not check_time_window(arrival, *window):
            return None, (BROKEN, "janela de entrega ultrapassada")
        start = max(arrival, window[0])

    elif action == LOAD:
        start = max([arrival] + [state.ready_at[i] for i in members])
        if any(start - state.stored_at[i] > problem.dwell_max for i in members):
            return None, (BROKEN, "mais de 24h no armazém")

    else:
        start = arrival

    if action in (PICKUP, DELIVER):
        finish = start + problem.stop_service
    else:
        finish = start + problem.warehouse_service

    # O veículo tem de conseguir terminar e regressar ao armazém no horizonte
    if finish + back > problem.horizon:
        return None, (BROKEN, "fora do horizonte de planeamento")

    # 4) Capacidade
    weight = sum(request[i].weight_kg for i in ids)
    if action in (PICKUP, LOAD):
        new_load = state.load[vehicle_id] + weight
    else:
        new_load = state.load[vehicle_id] - weight

    if not check_capacity(problem.vehicle[vehicle_id], new_load):
        return None, (BROKEN, "capacidade do veículo excedida")

    # 5) Novo estado
    new = state.copy()
    new.time[vehicle_id] = finish
    new.place[vehicle_id] = place
    new.load[vehicle_id] = max(0.0, new_load)
    new.stop[vehicle_id] = (action, place, arrival, start, members, stop_leg)
    new.distance += leg
    new.last = (vehicle_id, action, place, members, arrival, start, finish,
                stop_leg, extend)

    for i in ids:
        new.routes[vehicle_id].append((action, i))

        if action == PICKUP:
            new.status[i] = PICKED
            new.owner[i] = vehicle_id
        elif action == UNLOAD:
            new.status[i] = STORED
            new.owner[i] = None
            new.stored_at[i] = arrival      # as 24h contam desde a chegada
        elif action == LOAD:
            new.status[i] = LOADED
            new.owner[i] = vehicle_id
        else:
            new.status[i] = DELIVERED
            new.owner[i] = None

    if action == UNLOAD:
        for i in members:                   # disponível quando acaba a descarga
            new.ready_at[i] = finish

    return new, None


def apply_run(problem, state, vehicle_id, action, ids):
    """Executa operações consecutivas do mesmo tipo e local (uma sequência da rota).

    Vai juntando à mesma paragem enquanto as janelas deixam; quando não deixam,
    começa outra paragem no mesmo local. Devolve (estado, eventos, None) ou
    (None, None, (tipo, motivo)).
    """
    if action in (LOAD, DELIVER):          # encomendas inteiras
        by_order = {}
        for i in ids:
            by_order.setdefault(problem.request[i].order_id, []).append(i)
        units = list(by_order.values())
    else:
        units = [[i] for i in ids]

    events = []
    for unit in units:
        new, why = apply_stop(problem, state, vehicle_id, action, unit)
        if new is None:
            return None, None, why
        if new.last[8] and events:          # juntou-se à paragem anterior
            events[-1] = new.last[:8]
        else:
            events.append(new.last[:8])
        state = new
    return state, events, None


# Verificação de soluções

Result = namedtuple("Result", "valid distance message events")


def simulate(problem, solution, start_time=None, check_all=True):
    """Executa as rotas e valida-as. Devolve Result(valid, distance, message, events).

    solution: {veículo: [(operação, item), ...]}. Operações consecutivas do
    mesmo tipo e local formam uma só paragem. Os veículos são independentes,
    exceto no armazém: um veículo só carrega o que outro já descarregou.
    """
    events = []

    def fail(message, state=None):
        distance = round(state.distance, 2) if state else 0.0
        return Result(False, distance, message, events)

    for vehicle_id in solution:
        if vehicle_id not in problem.vehicle:
            return fail(f"Veículo desconhecido: {vehicle_id}")

    # Agrupar as operações de cada veículo em paragens
    stops = {v.id: [] for v in problem.vehicles}
    seen = set()

    for vehicle_id, route in solution.items():
        for action, request_id in route:

            if request_id not in problem.request:
                return fail(f"Produto desconhecido: {request_id}")
            if action not in OPERATIONS:
                return fail(f"Operação desconhecida: {action}")
            if (action, request_id) in seen:
                return fail(f"Operação repetida: {action} {request_id}")
            seen.add((action, request_id))

            place = problem.stop_place(action, request_id)
            last = stops[vehicle_id][-1] if stops[vehicle_id] else None

            if last and last[0] == action and last[1] == place:
                last[2].append(request_id)
            else:
                stops[vehicle_id].append((action, place, [request_id]))

    # Cada item incluído tem de fazer o percurso completo
    active = {request_id for _, request_id in seen}

    if check_all and active != set(problem.request):
        return fail("Encomendas por entregar")

    for request_id in active:
        order = problem.request[request_id].order_id
        if not set(problem.orders[order]) <= active:
            return fail(f"Encomenda {order} incompleta")
        for action in OPERATIONS:
            if (action, request_id) not in seen:
                return fail(f"Faltam operações no item {request_id}")

    # Simulação (o veículo cuja próxima paragem termina primeiro avança)
    if start_time is None:
        start_time = problem.start_time

    state = problem.initial_state(start_time)

    for t in state.time.values():
        if not problem.start_time <= t < problem.horizon:
            return fail("Partida fora do horizonte de planeamento")

    index = {v: 0 for v in stops}

    while any(index[v] < len(stops[v]) for v in stops):

        best = None

        for vehicle_id in stops:
            if index[vehicle_id] >= len(stops[vehicle_id]):
                continue

            action, _, ids = stops[vehicle_id][index[vehicle_id]]
            new, new_events, why = apply_run(problem, state, vehicle_id, action, ids)

            if new is None:
                if why[0] == BROKEN:
                    return fail(f"Veículo {vehicle_id}, {action} {ids}: {why[1]}", state)
                continue    # depende de outro veículo

            if best is None or new.time[vehicle_id] < best[1].time[best[0]]:
                best = (vehicle_id, new, new_events)

        if best is None:
            return fail("Ordem das operações inválida", state)

        vehicle_id, state, new_events = best
        events.extend(new_events)
        index[vehicle_id] += 1

    if not all(s in (NEW, DELIVERED) for s in state.status.values()):
        return fail("Produtos por entregar", state)

    # Regresso de cada veículo ao armazém
    warehouse = problem.warehouse_location

    for vehicle_id in stops:
        place = state.place[vehicle_id]

        try:
            leg = problem.get_distance(place, warehouse)
            arrival = state.time[vehicle_id] + problem.travel_time(place, warehouse)
        except KeyError:
            return fail("Regresso sem percurso", state)

        if arrival > problem.horizon:
            return fail("Regresso fora do horizonte", state)

        if leg > 0:
            events.append((vehicle_id, "return", warehouse, (), state.time[vehicle_id],
                           state.time[vehicle_id], arrival, leg))
        state.distance += leg
        state.time[vehicle_id] = arrival

    message = "Solução válida" if check_all else "Rotas válidas"
    return Result(True, round(state.distance, 2), message, events)


def check_solution(problem, solution, start_time=None, check_all=True):
    """Devolve (válida, distância total em km, mensagem)."""
    result = simulate(problem, solution, start_time, check_all)
    return result.valid, result.distance, result.message


def check_route(problem, vehicle, route, start_time=None):
    """Valida a rota de um único veículo (não exige entregar tudo)."""
    if vehicle.id not in problem.vehicle:
        return False, 0.0, "Veículo desconhecido"
    return check_solution(problem, {vehicle.id: route}, start_time, check_all=False)


def solution_cost(problem, solution, start_time=None):
    """Custo de uma solução completa: km totais, ou infinito se for inválida."""
    valid, distance, _ = check_solution(problem, solution, start_time)
    return distance if valid else float("inf")