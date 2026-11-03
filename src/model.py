"""
model.py — modelo do problema.

Lê os dados, aplica as regras de cada paragem e verifica soluções completas.
É usado por todos os algoritmos (Greedy, A*, Hill-Climbing e CSP) para validar
e avaliar as rotas.

Uma solução é um dicionário {veículo: [(operação, item), ...]}.
Cada item passa por: pickup -> unload -> load -> deliver.

Exemplo:
    problem = Problem.from_csv("data_tiny")
    valida, km, mensagem = check_solution(problem, solucao)
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
# Operações de cada item e regresso final ao armazém
PICKUP, UNLOAD, LOAD, DELIVER = "pickup", "unload", "load", "deliver"
OPERATIONS = (PICKUP, UNLOAD, LOAD, DELIVER)
RETURN = "return"

# Fases de um item: no produtor, recolhido, no armazém, carregado, entregue
NEW, PICKED, STORED, LOADED, DELIVERED = range(5)

# Fase necessária para cada operação
REQUIRED_STATUS = {PICKUP: NEW, UNLOAD: PICKED, LOAD: STORED, DELIVER: LOADED}

# Motivos de recusa: WAIT = ainda não é possível (depende do outro veículo);
# BROKEN = viola uma regra
WAIT, BROKEN = "wait", "rule"


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------
@dataclass
class Vehicle:
    """Veículo e respetiva capacidade (kg)."""
    id: int
    capacity_kg: float


@dataclass
class Request:
    """Item de uma encomenda: origem, destino, peso e janelas de tempo."""
    id: int
    order_id: int               # encomenda a que pertence
    pickup_location: int        # produtor
    delivery_location: int      # local de entrega
    weight_kg: float
    pickup_start: datetime      # janela de recolha
    pickup_end: datetime
    delivery_start: datetime    # janela de entrega
    delivery_end: datetime


@dataclass
class Stop:
    """Paragem realizada por um veículo."""
    action: str
    place: int
    arrival: datetime       # hora de chegada
    start: datetime         # início do serviço (no regresso: hora de partida)
    finish: datetime        # fim do serviço
    items: tuple
    km: float               # km até esta paragem


# ---------------------------------------------------------------------------
# Problem: dados fixos da instância
# ---------------------------------------------------------------------------
class Problem:

    def __init__(self, requests, vehicles, km, minutes, warehouse,
                 rules=None, names=None):
        self.requests = list(requests)  # itens a transportar
        self.vehicles = list(vehicles)
        self.km = km                    # (origem, destino) -> km
        self.minutes = minutes          # (origem, destino) -> minutos
        self.warehouse = warehouse      # local do armazém
        self.rules = rules or {}        # regras do rules.csv
        self.names = names or {}        # local -> nome

        # Acesso direto por id
        self.request = {item.id: item for item in self.requests}
        self.vehicle = {veiculo.id: veiculo for veiculo in self.vehicles}

        # Itens de cada encomenda
        self.orders = {}
        for item in self.requests:
            self.orders.setdefault(item.order_id, []).append(item.id)

        # Tempos definidos no rules.csv
        def minutos(chave, omissao):
            return timedelta(minutes=float(self.rules.get(chave, omissao)))

        self.service = minutos("stop_service_min", 10)           # em produtores e clientes
        self.handling = minutos("warehouse_handling_min", 15)    # no armazém
        self.max_dwell = timedelta(hours=float(self.rules.get("max_warehouse_dwell_h", 24)))

        # Início e fim do horizonte de planeamento
        if "start_date" in self.rules:
            self.start = datetime.fromisoformat(self.rules["start_date"])
        else:
            primeira = min(item.pickup_start for item in self.requests)
            self.start = primeira.replace(hour=0, minute=0, second=0, microsecond=0)

        if "horizon_days" in self.rules:
            dias = int(float(self.rules["horizon_days"]))
        else:
            ultima = max(item.delivery_end for item in self.requests)
            dias = (ultima.date() - self.start.date()).days + 1
        self.horizon = self.start + timedelta(days=dias)

    # --- Consultas ----------------------------------------------------------
    def distance(self, origem, destino):
        """Distância em km entre dois locais."""
        return 0.0 if origem == destino else self.km[origem, destino]

    def travel(self, origem, destino):
        """Tempo de viagem entre dois locais."""
        return timedelta(minutes=0 if origem == destino else self.minutes[origem, destino])

    def place_of(self, operacao, item):
        """Local onde se realiza a operação sobre o item."""
        if operacao == PICKUP:
            return item.pickup_location
        if operacao == DELIVER:
            return item.delivery_location
        return self.warehouse

    def name(self, local):
        """Nome do local."""
        return self.names.get(local, f"local {local}")

    def validate(self):
        """Verificações básicas aos dados. Devolve a lista de problemas encontrados."""
        problemas = []
        maior = max(veiculo.capacity_kg for veiculo in self.vehicles)

        for encomenda, ids in self.orders.items():
            itens = [self.request[i] for i in ids]
            if sum(item.weight_kg for item in itens) > maior:
                problemas.append(f"Encomenda {encomenda} pesa mais que o maior veículo")
            if max(item.delivery_start for item in itens) > min(item.delivery_end for item in itens):
                problemas.append(f"Encomenda {encomenda} sem janela de entrega comum")

        locais = {self.warehouse}
        locais |= {item.pickup_location for item in self.requests}
        locais |= {item.delivery_location for item in self.requests}
        for a in locais:
            for b in locais:
                if a != b and (a, b) not in self.km:
                    problemas.append(f"Falta a distância {a} -> {b}")
        return problemas

    def initial_state(self, start_time=None):
        """Estado inicial: veículos vazios no armazém e itens por recolher."""
        return State(self, start_time or self.start)

    # --- Leitura dos CSV ----------------------------------------------------
    @classmethod
    def from_csv(cls, folder):
        """Cria o problema a partir dos CSV de uma pasta (ex.: "data_tiny")."""
        pasta = Path(folder)
        if not pasta.is_dir():
            raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
        tabelas = {f.stem: pd.read_csv(f) for f in sorted(pasta.glob("*.csv"))}

        # Converte (tipo, id) no location_id usado na matriz de distâncias
        locais = tabelas["locations"]
        local_de = {(linha.location_type, int(linha.ref_id)): int(linha.location_id)
                    for linha in locais.itertuples()}
        nomes = {int(i): str(n) for i, n in zip(locais.location_id, locais.name)}

        # Veículos (todos partem do mesmo armazém)
        veiculos = [Vehicle(int(linha.vehicle_id), float(linha.capacity_kg))
                    for linha in tabelas["vehicles"].itertuples()]
        centros = set(tabelas["vehicles"]["center_id"])
        if len(centros) != 1:
            raise ValueError("Só é suportado um armazém")
        armazem = local_de["collection_center", int(centros.pop())]

        # Itens das encomendas
        produtos = tabelas["products"].set_index("product_id")
        encomendas = tabelas["orders"].set_index("order_id")
        itens = []
        for linha in tabelas["order_items"].itertuples():
            if "weight_kg" in tabelas["order_items"].columns:
                peso = float(linha.weight_kg)
            else:                                   # peso = quantidade x peso unitário
                peso = float(linha.quantity) * float(
                    produtos.loc[linha.product_id, "unit_weight_kg"])
            produtor = int(produtos.loc[linha.product_id, "producer_id"])
            cliente = int(encomendas.loc[linha.order_id, "delivery_location_id"])
            horas = [datetime.fromisoformat(str(x)) for x in (
                linha.availability_start, linha.availability_end,
                linha.delivery_start, linha.delivery_end)]
            itens.append(Request(
                int(linha.order_item_id), int(linha.order_id),
                local_de["producer", produtor],
                local_de["delivery_location", cliente],
                peso, *horas))

        # Matriz de distâncias e tempos
        matriz = tabelas["distance_matrix"]
        pares = list(zip(matriz.origin_location_id.astype(int),
                         matriz.destination_location_id.astype(int)))
        km = dict(zip(pares, matriz.distance_km.astype(float)))
        minutos = dict(zip(pares, matriz.duration_min.astype(float)))

        # Regras
        regras = {}
        if "rules" in tabelas:
            regras = {str(r): str(v) for r, v in
                      zip(tabelas["rules"].rule, tabelas["rules"].value)}

        return cls(itens, veiculos, km, minutos, armazem, regras, nomes)


# ---------------------------------------------------------------------------
# State: situação atual do planeamento
# ---------------------------------------------------------------------------
class State:

    def __init__(self, problem, start_time):
        veiculos = [veiculo.id for veiculo in problem.vehicles]
        itens = [item.id for item in problem.requests]

        # Por veículo
        self.time = {v: start_time for v in veiculos}            # hora a que fica livre
        self.place = {v: problem.warehouse for v in veiculos}    # local atual
        self.load = {v: 0.0 for v in veiculos}                   # carga (kg)
        self.stops = {v: [] for v in veiculos}                   # paragens realizadas

        # Por item
        self.status = {i: NEW for i in itens}                    # fase do item
        self.owner = {i: None for i in itens}                    # veículo que o leva
        self.stored_at = {i: None for i in itens}                # chegada ao armazém
        self.ready_at = {i: None for i in itens}                 # fim da descarga

        self.km = 0.0                                            # km percorridos

    def copy(self):
        """Cópia independente do estado."""
        return deepcopy(self)


# ---------------------------------------------------------------------------
# Regras de uma paragem
# ---------------------------------------------------------------------------
def wrong_status(estado, veiculo, operacao, ids):
    """Devolve o motivo se algum item não estiver pronto para a operação."""
    for i in ids:
        if estado.status[i] != REQUIRED_STATUS[operacao]:
            return "item ainda não está na fase certa"
        if operacao in (UNLOAD, DELIVER) and estado.owner[i] != veiculo:
            return "item não está neste veículo"
    return None


def _tentar_paragem(problem, estado, veiculo, operacao, ids, juntar):
    """Tenta aplicar uma paragem. Se for válida, atualiza o estado e devolve None;
    caso contrário, devolve (tipo, motivo) sem alterar o estado."""

    # 1) Fase dos itens
    motivo = wrong_status(estado, veiculo, operacao, ids)
    if motivo:
        return WAIT, motivo

    # 2) Todos os itens no mesmo local
    local = problem.place_of(operacao, problem.request[ids[0]])
    if any(problem.place_of(operacao, problem.request[i]) != local for i in ids):
        return BROKEN, "itens de locais diferentes na mesma paragem"

    # 3) Carga e entrega de encomendas completas
    if operacao in (LOAD, DELIVER):
        for encomenda in {problem.request[i].order_id for i in ids}:
            if not set(problem.orders[encomenda]) <= set(ids):
                return BROKEN, f"encomenda {encomenda} dividida"

    # 4) Chegada (operações iguais seguidas no mesmo local formam uma só paragem)
    anterior = estado.stops[veiculo][-1] if estado.stops[veiculo] else None
    juntar = (juntar and anterior is not None
              and anterior.action == operacao and anterior.place == local)
    try:
        if juntar:
            chegada, membros, km_viagem = anterior.arrival, anterior.items + tuple(ids), 0.0
        else:
            chegada = estado.time[veiculo] + problem.travel(estado.place[veiculo], local)
            membros, km_viagem = tuple(ids), problem.distance(estado.place[veiculo], local)
        regresso = problem.travel(local, problem.warehouse)
    except KeyError:
        return BROKEN, "percurso inexistente na matriz de distâncias"

    # 5) Início do serviço
    itens = [problem.request[i] for i in membros]
    if operacao == PICKUP:
        abre = max(item.pickup_start for item in itens)
        fecha = min(item.pickup_end for item in itens)
    elif operacao == DELIVER:
        abre = max(item.delivery_start for item in itens)
        fecha = min(item.delivery_end for item in itens)

    if operacao in (PICKUP, DELIVER):
        # pode esperar pela abertura, mas não começar depois do fecho
        if max(chegada, abre) > fecha:
            return BROKEN, "janela de tempo ultrapassada"
        inicio = max(chegada, abre)
    elif operacao == LOAD:
        # a carga começa depois de a descarga terminar
        inicio = max([chegada] + [estado.ready_at[i] for i in membros])
        # e respeita o máximo de 24 h no armazém
        if any(inicio - estado.stored_at[i] > problem.max_dwell for i in membros):
            return BROKEN, "mais de 24h no armazém"
    else:                                       # descarregar
        inicio = chegada

    # 6) Fim do serviço e regresso dentro do horizonte
    duracao = problem.service if operacao in (PICKUP, DELIVER) else problem.handling
    fim = inicio + duracao
    if fim + regresso > problem.horizon:
        return BROKEN, "fora do horizonte de planeamento"

    # 7) Capacidade do veículo
    peso = sum(problem.request[i].weight_kg for i in ids)
    nova_carga = estado.load[veiculo] + (peso if operacao in (PICKUP, LOAD) else -peso)
    if not -1e-6 <= nova_carga <= problem.vehicle[veiculo].capacity_kg + 1e-6:
        return BROKEN, "capacidade do veículo excedida"

    # 8) Atualizar o estado
    estado.time[veiculo] = fim
    estado.place[veiculo] = local
    estado.load[veiculo] = max(0.0, nova_carga)
    estado.km += km_viagem

    paragem = Stop(operacao, local, chegada, inicio, fim, membros,
                   km_viagem if not juntar else anterior.km)
    if juntar:
        estado.stops[veiculo][-1] = paragem
    else:
        estado.stops[veiculo].append(paragem)

    for i in ids:
        if operacao == PICKUP:
            estado.status[i], estado.owner[i] = PICKED, veiculo
        elif operacao == UNLOAD:
            estado.status[i], estado.owner[i] = STORED, None
            estado.stored_at[i] = chegada          # início da contagem das 24 h
        elif operacao == LOAD:
            estado.status[i], estado.owner[i] = LOADED, veiculo
        else:
            estado.status[i], estado.owner[i] = DELIVERED, None
    if operacao == UNLOAD:
        for i in membros:
            estado.ready_at[i] = fim               # disponível após a descarga
    return None


def apply_stop(problem, estado, veiculo, operacao, ids):
    """
    Aplica uma paragem do veículo aos itens indicados.
    Tenta juntá-la à paragem anterior e, se isso violar uma regra, faz uma
    paragem separada. Devolve None se for válida, ou (tipo, motivo).
    """
    # Validação dos argumentos
    if veiculo not in problem.vehicle:
        return BROKEN, f"veículo desconhecido: {veiculo}"
    if operacao not in OPERATIONS:
        return BROKEN, f"operação desconhecida: {operacao}"
    ids = list(ids)
    if not ids:
        return BROKEN, "lista de itens vazia"
    desconhecidos = [i for i in ids if i not in problem.request]
    if desconhecidos:
        return BROKEN, f"item desconhecido: {desconhecidos[0]}"
    if len(set(ids)) != len(ids):
        return BROKEN, "itens repetidos na mesma paragem"

    erro = _tentar_paragem(problem, estado, veiculo, operacao, ids, juntar=True)
    if erro and erro[0] == BROKEN:
        if _tentar_paragem(problem, estado, veiculo, operacao, ids, juntar=False) is None:
            return None
    return erro


# ---------------------------------------------------------------------------
# Verificação de soluções completas
# ---------------------------------------------------------------------------
def _paragens_da_rota(problem, rota):
    """Agrupa operações iguais seguidas no mesmo local: [(operação, local, [itens])]."""
    paragens = []
    for operacao, i in rota:
        local = problem.place_of(operacao, problem.request[i])
        if paragens and paragens[-1][0] == operacao and paragens[-1][1] == local:
            paragens[-1][2].append(i)
        else:
            paragens.append((operacao, local, [i]))
    return paragens


def simulate(problem, solution, check_all=True, start_time=None, parcial=False):
    """
    Simula a solução e devolve (válida?, estado final, mensagem).
    check_all=True: exige que todas as encomendas sejam entregues.
    parcial=True: aceita planos incompletos (usado pelo CSP); os itens
    carregados sem recolha no plano são considerados já no armazém.
    """
    def falha(mensagem, estado=None):
        return False, estado, mensagem

    # 1) Verificações básicas
    vistas = set()                              # (operação, item) já vistos
    paragens = {veiculo.id: [] for veiculo in problem.vehicles}
    for veiculo, rota in solution.items():
        if veiculo not in problem.vehicle:
            return falha(f"Veículo desconhecido: {veiculo}")
        for operacao, i in rota:
            if i not in problem.request:
                return falha(f"Item desconhecido: {i}")
            if operacao not in OPERATIONS:
                return falha(f"Operação desconhecida: {operacao}")
            if (operacao, i) in vistas:
                return falha(f"Operação repetida: {operacao} {i}")
            vistas.add((operacao, i))
        paragens[veiculo] = _paragens_da_rota(problem, rota)

    usados = {i for _, i in vistas}
    if not parcial:
        if check_all and usados != set(problem.request):
            return falha("Encomendas por entregar")
        for i in usados:
            encomenda = problem.request[i].order_id
            if not set(problem.orders[encomenda]) <= usados:
                return falha(f"Encomenda {encomenda} incompleta")
            if any((operacao, i) not in vistas for operacao in OPERATIONS):
                return falha(f"Faltam operações no item {i}")

    # 2) Simulação: cada veículo avança enquanto pode; se depender do outro
    #    veículo, passa-se ao seguinte
    estado = problem.initial_state(start_time)
    if not all(problem.start <= t < problem.horizon for t in estado.time.values()):
        return falha("Partida fora do horizonte de planeamento")

    if parcial:
        # Itens sem recolha no plano: considerados no armazém, sem contar as 24 h
        for i in usados:
            if (LOAD, i) in vistas and (UNLOAD, i) not in vistas:
                estado.status[i] = STORED
                estado.ready_at[i] = estado.time[problem.vehicles[0].id]
                estado.stored_at[i] = problem.horizon

    posicao = {v: 0 for v in paragens}          # próxima paragem de cada veículo
    while any(posicao[v] < len(paragens[v]) for v in paragens):
        avancou = False
        for v in paragens:
            while posicao[v] < len(paragens[v]):
                operacao, local, ids = paragens[v][posicao[v]]
                if wrong_status(estado, v, operacao, ids):
                    break                       # à espera de outro veículo

                if operacao in (PICKUP, UNLOAD):
                    unidades = [[i] for i in ids]        # item a item
                else:                                    # carga/entrega por encomenda
                    por_encomenda = {}
                    for i in ids:
                        por_encomenda.setdefault(problem.request[i].order_id, []).append(i)
                    unidades = list(por_encomenda.values())

                for unidade in unidades:
                    erro = apply_stop(problem, estado, v, operacao, unidade)
                    if erro:
                        return falha(f"Veículo {v}, {operacao} {unidade}: {erro[1]}", estado)
                posicao[v] += 1
                avancou = True
        if not avancou:
            return falha("Ordem das operações inválida", estado)

    if not parcial and not all(fase in (NEW, DELIVERED) for fase in estado.status.values()):
        return falha("Produtos por entregar", estado)

    # 3) Regresso ao armazém
    for v in paragens:
        try:
            km_viagem = problem.distance(estado.place[v], problem.warehouse)
            chegada = estado.time[v] + problem.travel(estado.place[v], problem.warehouse)
        except KeyError:
            return falha("Regresso sem percurso na matriz", estado)
        if chegada > problem.horizon:
            return falha("Regresso fora do horizonte", estado)
        if km_viagem > 0:
            estado.stops[v].append(Stop(RETURN, problem.warehouse, chegada,
                                        estado.time[v], chegada, (), km_viagem))
        estado.km += km_viagem
        estado.time[v], estado.place[v] = chegada, problem.warehouse

    return True, estado, "Solução válida" if check_all else "Rotas válidas"


def check_solution(problem, solution, start_time=None, check_all=True):
    """Verifica uma solução: devolve (válida?, km totais, mensagem)."""
    valida, estado, mensagem = simulate(problem, solution, check_all, start_time)
    return valida, round(estado.km, 2) if estado else 0.0, mensagem


def check_route(problem, vehicle, route, start_time=None):
    """Valida a rota de um só veículo (sem exigir que entregue tudo)."""
    if vehicle.id not in problem.vehicle:
        return False, 0.0, "Veículo desconhecido"
    return check_solution(problem, {vehicle.id: route}, start_time, check_all=False)


def solution_cost(problem, solution, start_time=None):
    """Função de custo: km totais, ou infinito se a solução for inválida."""
    valida, km, _ = check_solution(problem, solution, start_time)
    return km if valida else float("inf")


def print_solution(problem, solution):
    """Mostra o plano de cada veículo."""
    valida, estado, mensagem = simulate(problem, solution)
    print(mensagem, f"| total: {round(estado.km, 2) if estado else 0} km")
    if not estado:
        return
    for v, paragens in estado.stops.items():
        print(f"\nVeículo {v} (capacidade {problem.vehicle[v].capacity_kg:.0f} kg)")
        for p in paragens:
            quando = p.arrival if p.action == RETURN else p.start
            hora = quando.strftime("%d/%m %H:%M")
            itens = list(p.items) if p.items else ""
            print(f"  {hora}  {p.action:<8} {problem.name(p.place):<28} {str(itens):<24} +{p.km:.1f} km")
