"""
informed.py  --  Pesquisa informada: Greedy Best-First Search e A*.

GREEDY (primeira parte do ficheiro)
  Parte do estado inicial (veículos no armazém, nada feito) e escolhe sempre o
  estado que PARECE mais perto do fim, segundo uma heurística. Pára na primeira
  solução completa que encontra (não garante a melhor).
  Peças:
    - heurísticas   : estimam o que ainda falta fazer (H1 e H2);
    - sucessores    : movimentos possíveis a partir de um estado;
    - estado_viavel : deita fora estados que já não têm hipótese (becos sem saída);
    - greedy_search : o algoritmo, com uma fila de prioridade (heapq).
  Para ser rápido, só considera algumas opções em cada estado (as 3 encomendas
  mais urgentes, poucos grupos de recolha) e, numa primeira passagem, só
  movimentos que comecem nas 2 horas seguintes.
  As regras do problema são verificadas pelo apply_stop do model.py.

A* (segunda parte do ficheiro)
  A* sobre o problema COMPLETO (sem relaxar nada): f = g + h, com g = km feitos
  e h = estimativa admissível dos km que faltam. Usa sucessores completos
  (recolhas e descargas item a item), por isso, se chegasse ao fim, a solução
  seria a ótima. Nos nossos dados o espaço de estados é demasiado grande e não
  chega ao fim dentro do limite de expansões (ver o notebook).
"""

import heapq
import time
from datetime import timedelta

from src.model import (
    PICKUP, UNLOAD, LOAD, DELIVER,
    NEW, PICKED, STORED, LOADED, DELIVERED,
    apply_stop, check_solution,
)


# ---------------------------------------------------------------------------
# Heurísticas
# ---------------------------------------------------------------------------
def heuristica1(problem, estado):
    """H1: número de operações que ainda faltam.
    Cada item faz 4 operações: recolher, descarregar, carregar e entregar."""
    return sum(DELIVERED - fase for fase in estado.status.values())


def heuristica2(problem, estado):
    """H2: primeiro, quantos locais ainda faltam visitar (produtores e clientes);
    em caso de empate, os km desses locais ao armazém; por fim, a H1."""
    armazem = problem.warehouse
    produtores = {r.pickup_location for r in problem.requests
                  if estado.status[r.id] == NEW}
    clientes = {r.delivery_location for r in problem.requests
                if estado.status[r.id] != DELIVERED}
    km = sum(problem.distance(armazem, local) for local in produtores)
    km += sum(problem.distance(armazem, local) for local in clientes)
    return len(produtores) + len(clientes), round(km, 2), heuristica1(problem, estado)


# ---------------------------------------------------------------------------
# Estados
# ---------------------------------------------------------------------------
def terminou(problem, estado):
    """O trabalho acabou quando todos os itens estão entregues."""
    return all(fase == DELIVERED for fase in estado.status.values())


def chave_estado(problem, estado):
    """'Impressão digital' do estado, para não explorar duas vezes o mesmo."""
    itens = tuple((i, estado.status[i], estado.owner[i],
                   estado.stored_at[i] if estado.status[i] == STORED else None,
                   estado.ready_at[i] if estado.status[i] == STORED else None)
                  for i in sorted(problem.request))

    veiculos = []
    for veiculo in problem.vehicles:
        v = veiculo.id
        ultima = None
        if estado.stops[v]:
            p = estado.stops[v][-1]
            ultima = (p.action, tuple(sorted(p.items)), p.arrival)
        veiculos.append((v, estado.place[v], estado.time[v],
                         round(estado.load[v], 5), ultima))
    return itens, tuple(veiculos)


def chegada_mais_cedo(problem, estado, veiculo, operacao, local):
    """Hora mais cedo a que o veículo pode estar no local para esta operação.
    Se a última paragem foi a mesma operação no mesmo local, junta-se a ela."""
    ultima = estado.stops[veiculo][-1] if estado.stops[veiculo] else None
    if ultima is not None and ultima.action == operacao and ultima.place == local:
        return ultima.arrival
    return estado.time[veiculo] + problem.travel(estado.place[veiculo], local)


# ---------------------------------------------------------------------------
# Sucessores (movimentos possíveis)
# ---------------------------------------------------------------------------
def grupos_recolha(problem, estado, veiculo):
    """Grupos de itens do mesmo produtor que se podem recolher juntos
    (cabem no veículo e as janelas de recolha têm uma hora em comum)."""
    livre = problem.vehicle[veiculo].capacity_kg - estado.load[veiculo]

    por_produtor = {}
    for r in problem.requests:
        if estado.status[r.id] == NEW and r.weight_kg <= livre + 1e-6:
            por_produtor.setdefault(r.pickup_location, []).append(r)

    grupos = set()
    for local, itens in por_produtor.items():
        try:
            chegada = chegada_mais_cedo(problem, estado, veiculo, PICKUP, local)
        except KeyError:
            continue
        itens.sort(key=lambda r: (r.pickup_end, r.id))     # os que fecham primeiro

        # Pontos de partida: os 3 que fecham primeiro e o último (para ser rápido)
        for inicial in itens[:3] + itens[-1:]:
            inicio = max(chegada, inicial.pickup_start)
            if inicio > inicial.pickup_end:                # já fechou
                continue

            escolhidos = [inicial]
            peso = inicial.weight_kg
            for r in itens:
                if r.id == inicial.id or peso + r.weight_kg > livre + 1e-6:
                    continue
                novo_inicio = max(inicio, r.pickup_start)
                if novo_inicio <= min([r.pickup_end] + [x.pickup_end for x in escolhidos]):
                    escolhidos.append(r)
                    peso += r.weight_kg
                    inicio = novo_inicio
            grupos.add(tuple(sorted(r.id for r in escolhidos)))

    return sorted(grupos, key=lambda ids: (-len(ids), ids))   # grupos maiores primeiro


def sucessores(problem, estado):
    """Gera (novo estado, veículo, operação, itens) para cada movimento válido.
    O estado recebido nunca é alterado: cada movimento trabalha numa cópia."""
    for veiculo in problem.vehicles:
        v = veiculo.id
        movimentos = []

        # 1) Descarregar no armazém tudo o que o veículo leva
        recolhidos = [i for i in estado.status
                      if estado.status[i] == PICKED and estado.owner[i] == v]
        if recolhidos:
            movimentos.append((UNLOAD, recolhidos))

        # 2) Carregar e entregar encomendas (só as 3 mais urgentes de cada)
        no_armazem = [ids for ids in problem.orders.values()
                      if all(estado.status[i] == STORED for i in ids)]
        no_veiculo = [ids for ids in problem.orders.values()
                      if all(estado.status[i] == LOADED and estado.owner[i] == v for i in ids)]
        urgencia = lambda ids: min(problem.request[i].delivery_end for i in ids)
        no_armazem.sort(key=urgencia)
        no_veiculo.sort(key=urgencia)
        for ids in no_armazem[:3]:
            movimentos.append((LOAD, ids))
        for ids in no_veiculo[:3]:
            movimentos.append((DELIVER, ids))

        # 3) Recolher um grupo de itens num produtor
        for ids in grupos_recolha(problem, estado, v):
            movimentos.append((PICKUP, ids))

        # Só ficam os movimentos que cumprem as regras (apply_stop)
        for operacao, ids in movimentos:
            novo = estado.copy()
            if apply_stop(problem, novo, v, operacao, ids) is None:
                yield novo, v, operacao, ids


def estado_viavel(problem, estado):
    """Diz se o estado ainda pode levar a uma solução (cenário otimista).
    Deita fora os 'becos sem saída': itens que já não se recolhem a tempo
    ou encomendas que já não se entregam dentro da janela."""
    armazem = problem.warehouse
    pronto = {}                     # item -> hora mais cedo a que fica pronto no armazém

    for r in problem.requests:
        fase = estado.status[r.id]
        if fase == NEW:
            chegadas = []
            for v in estado.place:
                try:
                    chegadas.append(chegada_mais_cedo(problem, estado, v, PICKUP, r.pickup_location))
                except KeyError:
                    pass
            if not chegadas:
                return False
            t = max(min(chegadas), r.pickup_start)
            if t > r.pickup_end:                            # já não se recolhe a tempo
                return False
            try:
                pronto[r.id] = t + problem.travel(r.pickup_location, armazem) + problem.handling
            except KeyError:
                return False
        elif fase == PICKED:
            v = estado.owner[r.id]
            try:
                pronto[r.id] = estado.time[v] + problem.travel(estado.place[v], armazem) + problem.handling
            except KeyError:
                return False
        elif fase == STORED:
            pronto[r.id] = estado.ready_at[r.id]

    for ids in problem.orders.values():
        if all(estado.status[i] == DELIVERED for i in ids):
            continue
        r = problem.request[ids[0]]
        if all(estado.status[i] == LOADED for i in ids):    # já vai num veículo
            v = estado.owner[ids[0]]
            try:
                chegada = chegada_mais_cedo(problem, estado, v, DELIVER, r.delivery_location)
            except KeyError:
                return False
        else:                                               # ainda tem de passar pelo armazém
            try:
                carga = max(pronto[i] for i in ids if i in pronto)
                chegada = carga + problem.travel(armazem, r.delivery_location)
            except (KeyError, ValueError):
                return False

        inicio = max(chegada, max(problem.request[i].delivery_start for i in ids))
        fim = min(problem.request[i].delivery_end for i in ids)
        try:
            if inicio > fim or inicio + problem.travel(r.delivery_location, armazem) > problem.horizon:
                return False                                # já não se entrega a tempo
        except KeyError:
            return False
    return True


# ---------------------------------------------------------------------------
# Greedy Best-First Search
# ---------------------------------------------------------------------------
def _uma_passagem(problem, estado_inicial, heuristica, max_expansoes, janela):
    """Uma pesquisa Greedy. Se 'janela' não for None, só se expandem movimentos
    que começam até 'janela' depois do movimento mais cedo possível."""
    rotas_vazias = {v.id: [] for v in problem.vehicles}
    fila = [(heuristica(problem, estado_inicial), 0, estado_inicial, rotas_vazias)]
    visitados = set()
    expandidos = 0
    contador = 0                    # desempate: o heapq nunca compara estados

    while fila and expandidos < max_expansoes:
        # Tira da fila o estado com menor heurística (o que parece mais perto do fim)
        _, _, estado, rotas = heapq.heappop(fila)
        chave = chave_estado(problem, estado)
        if chave in visitados:
            continue
        visitados.add(chave)

        # Chegou ao fim? Confirma a solução e devolve-a
        if terminou(problem, estado):
            inicio = estado_inicial.time[problem.vehicles[0].id]
            valida, _, _ = check_solution(problem, rotas, start_time=inicio)
            if valida:
                return rotas, expandidos
            continue
        expandidos += 1

        # Gera os movimentos e, se houver janela, fica só com os mais próximos no tempo
        movimentos = list(sucessores(problem, estado))
        if janela is not None and movimentos:
            mais_cedo = min(novo.stops[v][-1].start for novo, v, _, _ in movimentos)
            movimentos = [(novo, v, op, ids) for novo, v, op, ids in movimentos
                          if novo.stops[v][-1].start <= mais_cedo + janela]

        # Põe na fila os estados novos que ainda têm hipótese
        for novo, v, operacao, ids in movimentos:
            if not estado_viavel(problem, novo):
                continue
            if chave_estado(problem, novo) in visitados:
                continue
            novas_rotas = {vid: rota[:] for vid, rota in rotas.items()}
            novas_rotas[v].extend((operacao, i) for i in ids)
            contador += 1
            heapq.heappush(fila, (heuristica(problem, novo), contador, novo, novas_rotas))

    return None, expandidos


def greedy_search(problem, estado_inicial=None, heuristica=heuristica1,
                  janela_horas=2, max_expansoes=5000):
    """
    Greedy Best-First Search.
    1.ª passagem: só movimentos nas próximas 'janela_horas' (mais rápido).
    Se não encontrar solução, 2.ª passagem sem esse limite.
    Cada passagem expande no máximo 'max_expansoes' estados.

    Devolve (rotas, nós expandidos). Rotas = None se não encontrar solução.
    As rotas estão no formato do model.py: {veículo: [(operação, item), ...]}
    """
    estado = estado_inicial if estado_inicial is not None else problem.initial_state()
    janela = None if janela_horas is None else timedelta(hours=janela_horas)

    total = 0
    for j in ([janela, None] if janela is not None else [None]):
        solucao, expandidos = _uma_passagem(problem, estado, heuristica, max_expansoes, j)
        total += expandidos
        if solucao is not None:
            return solucao, total
    return None, total


# =============================================================================
# A*  --  f(n) = g(n) + h(n)   (sobre o problema COMPLETO, sem relaxar nada)
#   g(n): km já percorridos (estado.km)
#   h(n): estimativa que nunca exagera dos km que ainda faltam
#         (inclui o regresso dos veículos ao armazém)
# Reutiliza do Greedy: terminou, chave_estado e estado_viavel.
# =============================================================================

def distancias_minimas(problem):
    """Km mínimos entre todos os pares de locais (Floyd-Warshall).
    Calcula-se uma vez e fica guardado no próprio problema."""
    if not hasattr(problem, "_distancias_minimas"):
        locais = sorted({problem.warehouse}
                        | {r.pickup_location for r in problem.requests}
                        | {r.delivery_location for r in problem.requests})
        d = {(a, b): 0.0 if a == b else problem.km.get((a, b), float("inf"))
             for a in locais for b in locais}
        for k in locais:
            for a in locais:
                for b in locais:
                    d[a, b] = min(d[a, b], d[a, k] + d[k, b])
        problem._distancias_minimas = d
    return problem._distancias_minimas


def astar_h1(problem, estado):
    """H1 (A*): cada veículo ainda tem de regressar ao armazém."""
    d = distancias_minimas(problem)
    return sum(d[estado.place[v], problem.warehouse] for v in estado.place)


def astar_h2(problem, estado):
    """H2 (A*): km mínimos da tarefa por fazer mais cara (recolher num produtor
    ou entregar uma encomenda), somados ao regresso dos outros veículos."""
    d = distancias_minimas(problem)
    W = problem.warehouse
    regresso = {v: d[estado.place[v], W] for v in estado.place}
    base = sum(regresso.values())                # = astar_h1
    melhor = base

    # Recolhas por fazer: algum veículo vai ao produtor e depois ao armazém
    for P in {r.pickup_location for r in problem.requests if estado.status[r.id] == NEW}:
        melhor = max(melhor, min(base - regresso[v] + d[estado.place[v], P] + d[P, W]
                                 for v in estado.place))

    # Encomendas por entregar
    for ids in problem.orders.values():
        primeiro = ids[0]
        if estado.status[primeiro] == DELIVERED:
            continue
        C = problem.request[primeiro].delivery_location
        if estado.status[primeiro] == LOADED:    # o veículo que a leva vai ao cliente
            v = estado.owner[primeiro]
            melhor = max(melhor, base - regresso[v] + d[estado.place[v], C] + d[C, W])
        else:                                    # ainda tem de passar pelo armazém
            melhor = max(melhor, min(base - regresso[v] + d[estado.place[v], W] + d[W, C] + d[C, W]
                                     for v in estado.place))
    return melhor


def astar_sucessores(problem, estado):
    """
    Gera TODOS os movimentos elementares possíveis para o A*.

    As recolhas e as descargas são geradas item a item. É intencional: assim
    nenhuma escolha "gulosa" de grupos esconde combinações válidas. Repetir a
    mesma operação no mesmo local junta-a à paragem anterior (apply_stop).
    O apply_stop trata da capacidade, das janelas e das 24 h no armazém.

    Carregar e entregar é sempre a encomenda completa, como exige o modelo.
    """
    for veiculo in problem.vehicles:
        v = veiculo.id

        # 1) Recolher: cada item por recolher é uma alternativa
        for r in problem.requests:
            if estado.status[r.id] != NEW:
                continue
            novo = estado.copy()
            if apply_stop(problem, novo, v, PICKUP, [r.id]) is None:
                yield novo

        # 2) Descarregar item a item (pode convir deixar alguns no veículo)
        recolhidos = [i for i in estado.status
                      if estado.status[i] == PICKED and estado.owner[i] == v]
        for i in recolhidos:
            novo = estado.copy()
            if apply_stop(problem, novo, v, UNLOAD, [i]) is None:
                yield novo

        # 3) Carregar ou entregar uma encomenda completa
        for ids in problem.orders.values():
            if all(estado.status[i] == STORED for i in ids):
                operacao = LOAD
            elif all(estado.status[i] == LOADED and estado.owner[i] == v for i in ids):
                operacao = DELIVER
            else:
                continue
            novo = estado.copy()
            if apply_stop(problem, novo, v, operacao, ids) is None:
                yield novo


def rotas_do_estado(problem, estado):
    """Converte as paragens do estado no formato do model.py:
    {veículo: [(operação, item), ...]}"""
    rotas = {v.id: [] for v in problem.vehicles}
    for v, paragens in estado.stops.items():
        for paragem in paragens:
            for i in paragem.items:
                rotas[v].append((paragem.action, i))
    return rotas


def _itens_por_fase(estado):
    """Quantos itens estão em cada fase (para as estatísticas)."""
    nomes = (("por recolher", NEW), ("recolhidos", PICKED), ("no armazém", STORED),
             ("carregados", LOADED), ("entregues", DELIVERED))
    return {nome: sum(f == fase for f in estado.status.values()) for nome, fase in nomes}


def a_star_search(problem, estado_inicial=None, heuristica=astar_h2,
                  max_expansoes=20000, max_segundos=None, estatisticas=False):
    """
    A* sobre o problema original, sem retirar restrições.

    g(n): km já percorridos.
    h(n): limite inferior dos km que ainda faltam (heurística admissível).
    O teste de fim é feito quando o estado SAI da fila (garante o ótimo) e a
    solução só é devolvida depois de passar o check_solution().

    Pára quando encontra a solução, quando atinge 'max_expansoes' ou quando
    passam 'max_segundos' (se for dado).

    Devolve (rotas, nós expandidos). Rotas = None se não encontrar solução.
    Com estatisticas=True devolve (rotas, nós expandidos, info), em que info
    mostra o tamanho da pesquisa: estados gerados, repetidos, ainda na fila,
    e em que fase estão os itens do estado mais avançado que foi explorado.
    """
    relogio = time.perf_counter()
    inicio = estado_inicial if estado_inicial is not None else problem.initial_state()
    W = problem.warehouse

    def prioridade(estado):
        if terminou(problem, estado):
            # Estado final: só falta o regresso ao armazém (custo real)
            h = sum(distancias_minimas(problem)[estado.place[v], W] for v in estado.place)
        else:
            h = heuristica(problem, estado)
        return estado.km + h

    def progresso(estado):
        """Para escolher o estado mais avançado: mais itens entregues,
        depois mais operações feitas e, em empate, menos km."""
        return (sum(f == DELIVERED for f in estado.status.values()),
                sum(estado.status.values()), -estado.km)

    contador = 0                                # desempate: o heapq nunca compara estados
    fila = [(prioridade(inicio), contador, inicio)]
    best_g = {chave_estado(problem, inicio): inicio.km}   # menor custo conhecido por estado
    expandidos, gerados, repetidos = 0, 1, 0
    mais_avancado = inicio
    solucao = None
    motivo = "fila vazia"

    while fila:
        if expandidos >= max_expansoes:
            motivo = "limite de expansões"
            break
        if max_segundos is not None and time.perf_counter() - relogio >= max_segundos:
            motivo = "limite de tempo"
            break

        _, _, estado = heapq.heappop(fila)

        # Ignora entradas antigas: já há um caminho mais barato para este estado
        chave = chave_estado(problem, estado)
        if estado.km > best_g.get(chave, float("inf")) + 1e-9:
            continue

        if progresso(estado) > progresso(mais_avancado):
            mais_avancado = estado

        # Teste de fim quando o estado SAI da fila
        if terminou(problem, estado):
            rotas = rotas_do_estado(problem, estado)
            valida, _, _ = check_solution(problem, rotas,
                                          start_time=inicio.time[problem.vehicles[0].id])
            if valida:
                solucao = rotas
                motivo = "solução encontrada"
                break
            continue                            # nunca devolver uma solução inválida

        expandidos += 1

        for novo in astar_sucessores(problem, estado):
            # Poda otimista: só elimina estados que já não podem cumprir
            # uma janela ou o horizonte (não estraga o ótimo)
            if not estado_viavel(problem, novo):
                continue

            nova_chave = chave_estado(problem, novo)
            if novo.km + 1e-9 >= best_g.get(nova_chave, float("inf")):
                repetidos += 1
                continue                        # já havia caminho igual ou melhor

            best_g[nova_chave] = novo.km
            contador += 1
            gerados += 1
            heapq.heappush(fila, (prioridade(novo), contador, novo))

    if not estatisticas:
        return solucao, expandidos

    info = {
        "motivo de paragem": motivo,
        "expandidos": expandidos,
        "gerados": gerados,
        "repetidos": repetidos,
        "ainda na fila": len(fila),
        "estado mais avançado": _itens_por_fase(mais_avancado),
        "tempo (s)": round(time.perf_counter() - relogio, 2),
    }
    return solucao, expandidos, info
