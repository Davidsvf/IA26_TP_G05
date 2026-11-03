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
  A* sobre o problema relaxado (sem janelas de tempo). Ver a explicação nessa secção.
"""

import heapq
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
# Distâncias mínimas (usadas pelo A* relaxado)
# =============================================================================

def shortest_km(problem):
    """Km mínimos entre todos os pares de locais (Floyd-Warshall).
    Calcula-se uma vez e fica guardado no próprio problema."""
    if not hasattr(problem, "_shortest_km"):
        places = sorted({problem.warehouse}
                        | {r.pickup_location for r in problem.requests}
                        | {r.delivery_location for r in problem.requests})
        d = {(a, b): 0.0 if a == b else problem.km.get((a, b), float("inf"))
             for a in places for b in places}
        for k in places:
            for a in places:
                for b in places:
                    d[a, b] = min(d[a, b], d[a, k] + d[k, b])
        problem._shortest_km = d
    return problem._shortest_km


# =============================================================================
# A* sobre o PROBLEMA RELAXADO  --  encontra a solução ÓTIMA desta versão
#
# Relaxação: são retiradas as JANELAS DE TEMPO (recolha e entrega).
# Mantém-se no problema: horizonte de 2 dias, máximo de 24h no armazém,
# capacidade dos veículos, produtor -> armazém -> cliente e encomendas inteiras.
#
# "Relaxar e verificar": a pesquisa não controla as horas, por isso resolve o
# problema sem janelas, sem horizonte e sem 24h. No fim, o check_solution
# confirma se a solução cumpre o horizonte e as 24h. Se cumprir, é também a
# solução ótima do problema em que só as janelas são relaxadas (repor regras
# nunca torna possível uma solução mais barata). Se não cumprir, é dado um
# aviso e essa garantia deixa de existir.
#
# Sem janelas, o problema reduz-se a "que veículo visita que locais e por que
# ordem". Considera-se ainda que:
#   - num produtor recolhe-se tudo de uma vez;
#   - no armazém descarrega-se tudo o que o veículo leva;
#   - as encomendas para o mesmo cliente carregam-se e entregam-se juntas.
# Sem capacidade isto não piora o ótimo (voltar a um local só acrescenta km).
# Com capacidade, a solução é a ótima entre as que visitam cada produtor uma
# vez e servem cada cliente de uma vez (não se divide uma recolha/entrega).
#
# Estado (tuplos pequenos, muito mais rápido que o State do model.py):
#   (posição de cada veículo, fase de cada produtor, fase de cada cliente)
#   produtor: 0 = por recolher, k = no veículo k, DONE = no armazém
#   cliente:  0 = por carregar, k = no veículo k, DONE = entregue
# =============================================================================

from dataclasses import replace

from src.model import Problem


def relax_problem(problem, keep_capacity=True):
    """Versão relaxada: as janelas de recolha e de entrega passam a cobrir todo
    o horizonte. O horizonte e o limite de 24h no armazém mantêm-se.
    Com keep_capacity=False os veículos ficam também sem limite de carga."""
    requests = [replace(r, pickup_start=problem.start, pickup_end=problem.horizon,
                        delivery_start=problem.start, delivery_end=problem.horizon)
                for r in problem.requests]
    vehicles = list(problem.vehicles)
    if not keep_capacity:
        vehicles = [replace(v, capacity_kg=float("inf")) for v in problem.vehicles]
    return Problem(requests, vehicles, problem.km, problem.minutes,
                   problem.warehouse, problem.rules, problem.names)


def _relaxed_context(problem):
    """Dados fixos usados pela pesquisa relaxada."""
    producers = sorted({r.pickup_location for r in problem.requests})
    groups = {}                                   # cliente -> itens a entregar lá
    for r in problem.requests:
        groups.setdefault(r.delivery_location, []).append(r.id)
    dests = sorted(groups)
    index = {P: j for j, P in enumerate(producers)}
    # produtores de que cada cliente precisa (para poder carregar no armazém)
    need = [tuple(sorted({index[problem.request[i].pickup_location] for i in groups[D]}))
            for D in dests]
    # pesos: tudo o que se recolhe num produtor / se entrega num cliente
    w_prod = [sum(r.weight_kg for r in problem.requests if r.pickup_location == P)
              for P in producers]
    w_dest = [sum(problem.request[i].weight_kg for i in groups[D]) for D in dests]
    return {"d": shortest_km(problem), "W": problem.warehouse,
            "producers": producers, "dests": dests, "groups": groups,
            "need": need, "DONE": len(problem.vehicles) + 1,
            "capacity": [v.capacity_kg for v in problem.vehicles],
            "w_prod": w_prod, "w_dest": w_dest}


def _relaxed_load(ctx, state, k):
    """Kg que o veículo k leva neste estado (recolhas + encomendas carregadas)."""
    code = k + 1
    return (sum(w for w, s in zip(ctx["w_prod"], state[1]) if s == code)
            + sum(w for w, s in zip(ctx["w_dest"], state[2]) if s == code))


def relaxed_h1(ctx, state):
    """H1: cada veículo ainda tem de regressar ao armazém."""
    d, W = ctx["d"], ctx["W"]
    return sum(d[p, W] for p in state[0])


def relaxed_h2(ctx, state):
    """H2: árvore geradora mínima (MST) que liga o armazém, os veículos e os
    locais ainda por visitar. Os percursos dos veículos têm de ligar todos
    esses pontos, por isso nunca são mais curtos que a MST (admissível)."""
    d, W, DONE = ctx["d"], ctx["W"], ctx["DONE"]
    pos, prod, deliv = state
    points = set(pos)
    points.update(ctx["producers"][j] for j, s in enumerate(prod) if s == 0)
    points.update(ctx["dests"][t] for t, s in enumerate(deliv) if s != DONE)
    points.discard(W)

    dist = {b: d[W, b] for b in points}           # algoritmo de Prim a partir do armazém
    total = 0.0
    while dist:
        b = min(dist, key=dist.get)
        total += dist.pop(b)
        for c in dist:
            dist[c] = min(dist[c], d[b, c])
    return max(total, relaxed_h1(ctx, state))


def _relaxed_moves(ctx, state):
    """Gera (custo em km, novo estado, movimento) para cada veículo.
    Recolher e carregar só são possíveis se couberem no veículo."""
    d, W, DONE = ctx["d"], ctx["W"], ctx["DONE"]
    producers, dests, need = ctx["producers"], ctx["dests"], ctx["need"]
    pos, prod, deliv = state

    for k in range(len(pos)):
        here, code = pos[k], k + 1
        moved = lambda place: pos[:k] + (place,) + pos[k + 1:]

        free = ctx["capacity"][k] - _relaxed_load(ctx, state, k)

        # Recolher tudo num produtor ainda por visitar (se couber)
        for j, s in enumerate(prod):
            if s == 0 and ctx["w_prod"][j] <= free + 1e-6:
                P = producers[j]
                yield (d[here, P], (moved(P), prod[:j] + (code,) + prod[j + 1:], deliv),
                       (k, PICKUP, P))

        # Descarregar no armazém tudo o que o veículo leva
        if code in prod:
            yield (d[here, W], (moved(W), tuple(DONE if s == code else s for s in prod), deliv),
                   (k, UNLOAD, W))

        for t, s in enumerate(deliv):
            # Carregar as encomendas de um cliente (produtos já no armazém e cabem)
            if (s == 0 and all(prod[j] == DONE for j in need[t])
                    and ctx["w_dest"][t] <= free + 1e-6):
                yield (d[here, W], (moved(W), prod, deliv[:t] + (code,) + deliv[t + 1:]),
                       (k, LOAD, dests[t]))
            # Entregar no cliente as encomendas que este veículo leva
            elif s == code:
                D = dests[t]
                yield (d[here, D], (moved(D), prod, deliv[:t] + (DONE,) + deliv[t + 1:]),
                       (k, DELIVER, D))


def _relaxed_routes(problem, ctx, state, parent):
    """Reconstrói o caminho e converte-o no formato do model.py:
    {veículo: [(operação, item), ...]}"""
    path = []
    while parent[state] is not None:
        state, move = parent[state]
        path.append(move)
    path.reverse()

    vids = [v.id for v in problem.vehicles]
    routes = {v: [] for v in vids}
    carrying = {v: [] for v in vids}
    for k, action, place in path:
        v = vids[k]
        if action == PICKUP:
            ids = [r.id for r in problem.requests if r.pickup_location == place]
            carrying[v] += ids
        elif action == UNLOAD:
            ids, carrying[v] = carrying[v], []
        else:                                     # LOAD ou DELIVER: itens desse cliente
            ids = ctx["groups"][place]
        routes[v] += [(action, i) for i in ids]
    return routes


def relaxed_a_star(problem, heuristic=relaxed_h2, max_expansions=None):
    """
    A* sobre o problema relaxado (usar com relax_problem(problem)).
    Respeita a capacidade dos veículos do problema recebido. No fim verifica
    o horizonte e o limite de 24h com o check_solution (ver comentário acima).
    Com heurística admissível, devolve a solução de MENOR custo (km) dessa versão.
    Devolve (rotas, nós expandidos). Rotas = None se não encontrar solução.
    """
    ctx = _relaxed_context(problem)
    DONE = ctx["DONE"]

    def is_final(state):
        return all(s == DONE for s in state[2])

    def priority(state, g):
        if is_final(state):                       # custo real: km + regresso ao armazém
            return g + relaxed_h1(ctx, state)
        return g + heuristic(ctx, state)

    start = (tuple(ctx["W"] for _ in problem.vehicles),
             tuple(0 for _ in ctx["producers"]),
             tuple(0 for _ in ctx["dests"]))
    best_g = {start: 0.0}                          # menor custo conhecido por estado
    parent = {start: None}                         # para reconstruir o caminho
    counter = 0
    queue = [(priority(start, 0.0), counter, 0.0, start)]
    expanded = 0

    while queue and (max_expansions is None or expanded < max_expansions):
        _, _, g, state = heapq.heappop(queue)
        if g > best_g[state] + 1e-9:               # entrada desatualizada
            continue

        # Teste de fim quando o estado SAI da fila (garante o ótimo)
        if is_final(state):
            routes = _relaxed_routes(problem, ctx, state, parent)
            # Verifica as regras de tempo que a pesquisa não controla (horizonte, 24h)
            valid, _, message = check_solution(problem, routes)
            if not valid:
                print("Aviso: a melhor rota não cumpre as regras de tempo "
                      f"({message}); o ótimo deixa de estar garantido.")
            return routes, expanded
        expanded += 1

        for cost, new_state, move in _relaxed_moves(ctx, state):
            new_g = g + cost
            if new_g < best_g.get(new_state, float("inf")) - 1e-9:
                best_g[new_state] = new_g
                parent[new_state] = (state, move)
                counter += 1
                heapq.heappush(queue, (priority(new_state, new_g), counter, new_g, new_state))

    return None, expanded
