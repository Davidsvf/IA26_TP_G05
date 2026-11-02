"""
Como se usa:
    problem = Problem.from_csv("data_tiny")
    valida, km, mensagem = check_solution(problem, solucao)
    custo = solution_cost(problem, solucao)        # km, ou infinito se inválida

Uma solução é um dicionário  {id_do_veículo: [(operação, id_do_item), ...]}
Todo o item faz 4 operações, sempre por esta ordem:
    pickup  -> recolher no produtor
    unload  -> descarregar no armazém
    load    -> carregar no armazém (para ir ao cliente)
    deliver -> entregar ao cliente
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PICKUP, UNLOAD, LOAD, DELIVER = "pickup", "unload", "load", "deliver"
OPERATIONS = (PICKUP, UNLOAD, LOAD, DELIVER)
RETURN = "return"                       # regresso final ao armazém

# Onde está cada item ao longo da sua "viagem"
NEW, PICKED, STORED, LOADED, DELIVERED = range(5)

# Estado em que o item tem de estar para se poder fazer cada operação
REQUIRED_STATUS = {PICKUP: NEW, UNLOAD: PICKED, LOAD: STORED, DELIVER: LOADED}

# Tipos de recusa: WAIT = "ainda não" (falta outro veículo fazer algo),
#                  BROKEN = "viola uma regra"
WAIT, BROKEN = "wait", "rule"


# 1) Fichas simples

@dataclass
class Vehicle:
    id: int
    capacity_kg: float


@dataclass
class Request:
    """Um item de uma encomenda: o que levar, de onde, para onde e quando."""
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
class Stop:
    """Uma paragem já feita por um veículo (serve para registo e impressão)."""
    action: str
    place: int
    arrival: datetime       # chegada ao local
    start: datetime         # início do serviço (no "return": hora de saída do último local)
    finish: datetime        # fim do serviço
    items: tuple
    km: float               # km percorridos para chegar a esta paragem


# 2) Problem: tudo o que NÃO muda durante o planeamento
class Problem:

    def __init__(self, requests, vehicles, km, minutes, warehouse,
                 rules=None, names=None):
        self.requests = list(requests)
        self.vehicles = list(vehicles)
        self.km = km                    # (origem, destino) -> km
        self.minutes = minutes          # (origem, destino) -> minutos
        self.warehouse = warehouse      # id do local do armazém
        self.rules = rules or {}
        self.names = names or {}

        self.request = {r.id: r for r in self.requests}
        self.vehicle = {v.id: v for v in self.vehicles}

        self.orders = {}                # id da encomenda -> ids dos seus itens
        for r in self.requests:
            self.orders.setdefault(r.order_id, []).append(r.id)

        # Regras do rules.csv
        minutes_rule = lambda key, default: timedelta(
            minutes=float(self.rules.get(key, default)))
        self.service = minutes_rule("stop_service_min", 10)         # produtor/cliente
        self.handling = minutes_rule("warehouse_handling_min", 15)  # armazém
        self.max_dwell = timedelta(
            hours=float(self.rules.get("max_warehouse_dwell_h", 24)))

        # Início e fim do planeamento
        if "start_date" in self.rules:
            self.start = datetime.fromisoformat(self.rules["start_date"])
        else:
            first = min(r.pickup_start for r in self.requests)
            self.start = first.replace(hour=0, minute=0, second=0, microsecond=0)
        if "horizon_days" in self.rules:
            days = int(float(self.rules["horizon_days"]))
        else:
            last = max(r.delivery_end for r in self.requests)
            days = (last.date() - self.start.date()).days + 1
        self.horizon = self.start + timedelta(days=days)

    # --- consultas -----------------------------------------------------
    def distance(self, a, b):
        return 0.0 if a == b else self.km[a, b]          # KeyError se não existir

    def travel(self, a, b):
        return timedelta(minutes=0 if a == b else self.minutes[a, b])

    def place_of(self, action, request):
        """Onde se faz cada operação."""
        if action == PICKUP:
            return request.pickup_location
        if action == DELIVER:
            return request.delivery_location
        return self.warehouse                            # unload e load

    def name(self, place):
        return self.names.get(place, f"local {place}")

    def validate(self):
        """Verificações simples aos dados. Lista vazia = nada de errado detetado
        (não garante que exista solução)."""
        issues = []
        biggest = max(v.capacity_kg for v in self.vehicles)
        for order, ids in self.orders.items():
            reqs = [self.request[i] for i in ids]
            if sum(r.weight_kg for r in reqs) > biggest:
                issues.append(f"Encomenda {order} pesa mais que o maior veículo")
            if max(r.delivery_start for r in reqs) > min(r.delivery_end for r in reqs):
                issues.append(f"Encomenda {order} sem janela de entrega comum")
        places = {self.warehouse}
        places |= {r.pickup_location for r in self.requests}
        places |= {r.delivery_location for r in self.requests}
        for a in places:
            for b in places:
                if a != b and (a, b) not in self.km:
                    issues.append(f"Falta a distância {a} -> {b}")
        return issues

    def initial_state(self, start_time=None):
        return State(self, start_time or self.start)

    # --- leitura dos CSV -----------------------------------------------
    @classmethod
    def from_csv(cls, folder):
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"Pasta não encontrada: {folder}")
        t = {f.stem: pd.read_csv(f) for f in sorted(folder.glob("*.csv"))}

        # Cada local (produtor, cliente, armazém) tem um location_id, que é o
        # número usado na matriz de distâncias.
        locations = t["locations"]
        place_id = {(row.location_type, int(row.ref_id)): int(row.location_id)
                    for row in locations.itertuples()}
        names = {int(i): str(n) for i, n in zip(locations.location_id, locations.name)}

        vehicles = [Vehicle(int(v.vehicle_id), float(v.capacity_kg))
                    for v in t["vehicles"].itertuples()]
        centers = set(t["vehicles"]["center_id"])
        if len(centers) != 1:
            raise ValueError("Só é suportado um armazém")
        warehouse = place_id["collection_center", int(centers.pop())]

        products = t["products"].set_index("product_id")
        orders = t["orders"].set_index("order_id")
        requests = []
        for it in t["order_items"].itertuples():
            if "weight_kg" in t["order_items"].columns:
                weight = float(it.weight_kg)
            else:                                   # peso = quantidade x peso unitário
                weight = float(it.quantity) * float(
                    products.loc[it.product_id, "unit_weight_kg"])
            producer = int(products.loc[it.product_id, "producer_id"])
            delivery = int(orders.loc[it.order_id, "delivery_location_id"])
            when = [datetime.fromisoformat(str(x)) for x in (
                it.availability_start, it.availability_end,
                it.delivery_start, it.delivery_end)]
            requests.append(Request(
                int(it.order_item_id), int(it.order_id),
                place_id["producer", producer],
                place_id["delivery_location", delivery],
                weight, *when))

        dm = t["distance_matrix"]
        pairs = list(zip(dm.origin_location_id.astype(int),
                         dm.destination_location_id.astype(int)))
        km = dict(zip(pairs, dm.distance_km.astype(float)))
        minutes = dict(zip(pairs, dm.duration_min.astype(float)))

        rules = {}
        if "rules" in t:
            rules = {str(r): str(v) for r, v in zip(t["rules"].rule, t["rules"].value)}

        return cls(requests, vehicles, km, minutes, warehouse, rules, names)



# 3) State

class State:

    def __init__(self, problem, start_time):
        v_ids = [v.id for v in problem.vehicles]
        r_ids = [r.id for r in problem.requests]
        # por veículo
        self.time = {v: start_time for v in v_ids}              # hora a que fica livre
        self.place = {v: problem.warehouse for v in v_ids}      # onde está
        self.load = {v: 0.0 for v in v_ids}                     # kg a bordo
        self.stops = {v: [] for v in v_ids}                     # paragens já feitas
        # por item
        self.status = {i: NEW for i in r_ids}                   # NEW ... DELIVERED
        self.owner = {i: None for i in r_ids}                   # veículo que o leva
        self.stored_at = {i: None for i in r_ids}               # chegada ao armazém
        self.ready_at = {i: None for i in r_ids}                # fim da descarga
        self.km = 0.0                                           # km percorridos

    def copy(self):
        return deepcopy(self)


# 4) As regras: o que acontece quando um veículo faz uma paragem

def wrong_status(s, v, action, ids):
    """Devolve um motivo se algum item ainda não está pronto para esta operação."""
    for i in ids:
        if s.status[i] != REQUIRED_STATUS[action]:
            return "item ainda não está na fase certa"
        if action in (UNLOAD, DELIVER) and s.owner[i] != v:
            return "item não está neste veículo"
    return None


def _try_stop(p, s, v, action, ids, merge):
    """Tenta fazer uma paragem do veículo v. Se correr bem, altera o estado s
    e devolve None. Se não, devolve (tipo, motivo) e NÃO altera nada."""
    why = wrong_status(s, v, action, ids)
    if why:
        return WAIT, why

    # Todos os itens da paragem têm de ser do mesmo local
    place = p.place_of(action, p.request[ids[0]])
    if any(p.place_of(action, p.request[i]) != place for i in ids):
        return BROKEN, "itens de locais diferentes na mesma paragem"

    # Carregar/entregar: as encomendas vão sempre inteiras
    if action in (LOAD, DELIVER):
        for order in {p.request[i].order_id for i in ids}:
            if not set(p.orders[order]) <= set(ids):
                return BROKEN, f"encomenda {order} dividida"

    prev = s.stops[v][-1] if s.stops[v] else None
    # Operações iguais seguidas no mesmo local contam como UMA só paragem
    merge = merge and prev is not None and prev.action == action and prev.place == place

    # Chegada
    try:
        if merge:
            arrival, members, leg = prev.arrival, prev.items + tuple(ids), 0.0
        else:
            arrival = s.time[v] + p.travel(s.place[v], place)
            members, leg = tuple(ids), p.distance(s.place[v], place)
        back = p.travel(place, p.warehouse)
    except KeyError:
        return BROKEN, "percurso inexistente na matriz de distâncias"

    # Início do serviço: espera pela abertura da janela, mas não pode chegar
    # depois do fecho
    group = [p.request[i] for i in members]
    if action == PICKUP:
        opens = max(r.pickup_start for r in group)
        closes = min(r.pickup_end for r in group)
    elif action == DELIVER:
        opens = max(r.delivery_start for r in group)
        closes = min(r.delivery_end for r in group)
    if action in (PICKUP, DELIVER):
        if max(arrival, opens) > closes:
            return BROKEN, "janela de tempo ultrapassada"
        start = max(arrival, opens)
    elif action == LOAD:
        # só se pode carregar depois de tudo estar descarregado...
        start = max([arrival] + [s.ready_at[i] for i in members])
        # ...e nenhum item pode estar mais de 24h no armazém
        if any(start - s.stored_at[i] > p.max_dwell for i in members):
            return BROKEN, "mais de 24h no armazém"
    else:                                   # unload
        start = arrival

    finish = start + (p.service if action in (PICKUP, DELIVER) else p.handling)
    if finish + back > p.horizon:
        return BROKEN, "fora do horizonte de planeamento"

    # Capacidade
    weight = sum(p.request[i].weight_kg for i in ids)
    new_load = s.load[v] + (weight if action in (PICKUP, LOAD) else -weight)
    if not -1e-6 <= new_load <= p.vehicle[v].capacity_kg + 1e-6:
        return BROKEN, "capacidade do veículo excedida"

    # Tudo certo: atualizar o estado
    s.time[v], s.place[v], s.load[v] = finish, place, max(0.0, new_load)
    s.km += leg
    stop = Stop(action, place, arrival, start, finish, members,
                leg if not merge else prev.km)
    if merge:
        s.stops[v][-1] = stop
    else:
        s.stops[v].append(stop)

    for i in ids:
        if action == PICKUP:
            s.status[i], s.owner[i] = PICKED, v
        elif action == UNLOAD:
            s.status[i], s.owner[i], s.stored_at[i] = STORED, None, arrival
        elif action == LOAD:
            s.status[i], s.owner[i] = LOADED, v
        else:
            s.status[i], s.owner[i] = DELIVERED, None
    if action == UNLOAD:
        for i in members:
            s.ready_at[i] = finish
    return None


def apply_stop(p, s, v, action, ids):
    """Faz uma paragem. Tenta juntá-la à anterior (mesmo local e operação);
    se a junção violar uma janela, faz uma paragem separada.
    Devolve None se correu bem, ou (tipo, motivo) se não."""
    # Validar os argumentos antes de mexer em qualquer coisa
    if v not in p.vehicle:
        return BROKEN, f"veículo desconhecido: {v}"
    if action not in OPERATIONS:
        return BROKEN, f"operação desconhecida: {action}"
    ids = list(ids)
    if not ids:
        return BROKEN, "lista de itens vazia"
    unknown = [i for i in ids if i not in p.request]
    if unknown:
        return BROKEN, f"item desconhecido: {unknown[0]}"
    if len(set(ids)) != len(ids):
        return BROKEN, "itens repetidos na mesma paragem"

    error = _try_stop(p, s, v, action, ids, merge=True)
    if error and error[0] == BROKEN and _try_stop(p, s, v, action, ids, merge=False) is None:
        return None
    return error


# 5) O "fiscal": verifica uma solução completa

def _stops_of(p, route):
    """Junta operações iguais seguidas no mesmo local: [(operação, local, [itens])]"""
    blocks = []
    for action, i in route:
        place = p.place_of(action, p.request[i])
        if blocks and blocks[-1][0] == action and blocks[-1][1] == place:
            blocks[-1][2].append(i)
        else:
            blocks.append((action, place, [i]))
    return blocks


def simulate(p, solution, check_all=True, start_time=None):
    """Simula a solução. Devolve (válida?, estado final, mensagem)."""
    def fail(message, state=None):
        return False, state, message

    # 1) Verificações básicas
    seen, blocks = set(), {v.id: [] for v in p.vehicles}
    for v, route in solution.items():
        if v not in p.vehicle:
            return fail(f"Veículo desconhecido: {v}")
        for action, i in route:
            if i not in p.request:
                return fail(f"Item desconhecido: {i}")
            if action not in OPERATIONS:
                return fail(f"Operação desconhecida: {action}")
            if (action, i) in seen:
                return fail(f"Operação repetida: {action} {i}")
            seen.add((action, i))
        blocks[v] = _stops_of(p, route)

    active = {i for _, i in seen}
    if check_all and active != set(p.request):
        return fail("Encomendas por entregar")
    for i in active:
        order = p.request[i].order_id
        if not set(p.orders[order]) <= active:
            return fail(f"Encomenda {order} incompleta")
        if any((action, i) not in seen for action in OPERATIONS):
            return fail(f"Faltam operações no item {i}")

    # 2) Simulação: cada veículo avança enquanto puder; se está à espera de
    #    outro veículo (ex.: item ainda não chegou ao armazém), passa-se ao seguinte
    s = p.initial_state(start_time)
    if not all(p.start <= t < p.horizon for t in s.time.values()):
        return fail("Partida fora do horizonte de planeamento")

    pos = {v: 0 for v in blocks}
    while any(pos[v] < len(blocks[v]) for v in blocks):
        moved = False
        for v in blocks:
            while pos[v] < len(blocks[v]):
                action, place, ids = blocks[v][pos[v]]
                if wrong_status(s, v, action, ids):
                    break                                   # à espera de outro veículo
                if action in (PICKUP, UNLOAD):
                    units = [[i] for i in ids]              # item a item
                else:                                       # load/deliver: encomendas inteiras
                    by_order = {}
                    for i in ids:
                        by_order.setdefault(p.request[i].order_id, []).append(i)
                    units = list(by_order.values())
                for unit in units:
                    error = apply_stop(p, s, v, action, unit)
                    if error:
                        return fail(f"Veículo {v}, {action} {unit}: {error[1]}", s)
                pos[v] += 1
                moved = True
        if not moved:
            return fail("Ordem das operações inválida", s)

    if not all(st in (NEW, DELIVERED) for st in s.status.values()):
        return fail("Produtos por entregar", s)

    # 3) Regresso de todos os veículos ao armazém
    for v in blocks:
        try:
            leg = p.distance(s.place[v], p.warehouse)
            arrival = s.time[v] + p.travel(s.place[v], p.warehouse)
        except KeyError:
            return fail("Regresso sem percurso na matriz", s)
        if arrival > p.horizon:
            return fail("Regresso fora do horizonte", s)
        if leg > 0:
            s.stops[v].append(Stop(RETURN, p.warehouse, arrival, s.time[v], arrival, (), leg))
        s.km += leg
        s.time[v], s.place[v] = arrival, p.warehouse

    return True, s, "Solução válida" if check_all else "Rotas válidas"


def check_solution(p, solution, start_time=None, check_all=True):
    """(válida?, km totais, mensagem)"""
    ok, s, message = simulate(p, solution, check_all, start_time)
    return ok, round(s.km, 2) if s else 0.0, message


def check_route(p, vehicle, route, start_time=None):
    """Valida a rota de um só veículo (sem exigir que entregue tudo)."""
    if vehicle.id not in p.vehicle:
        return False, 0.0, "Veículo desconhecido"
    return check_solution(p, {vehicle.id: route}, start_time, check_all=False)


def solution_cost(p, solution, start_time=None):
    """Função de custo: km totais se a solução é válida, infinito se não for."""
    ok, km, _ = check_solution(p, solution, start_time)
    return km if ok else float("inf")


def print_solution(p, solution):
    """Mostra o plano de cada veículo: hora, operação, local, itens e km."""
    ok, s, message = simulate(p, solution)
    print(message, f"| total: {round(s.km, 2) if s else 0} km")
    if not s:
        return
    for v, stops in s.stops.items():
        print(f"\nVeículo {v} (capacidade {p.vehicle[v].capacity_kg:.0f} kg)")
        for st in stops:
            when = st.arrival if st.action == RETURN else st.start
            hour = when.strftime("%d/%m %H:%M")
            items = list(st.items) if st.items else ""
            print(f"  {hour}  {st.action:<8} {p.name(st.place):<28} {str(items):<24} +{st.km:.1f} km")
