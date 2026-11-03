"""
informed.py — pesquisa informada: Greedy Best-First Search e A*.

Greedy: expande sempre o estado com menor heurística e devolve a primeira
solução válida. Para ser rápido, considera apenas algumas opções em cada
estado e, numa primeira passagem, só movimentos nas 2 horas seguintes.

A*: expande o estado com menor f = g + h (g = km percorridos, h = estimativa
admissível dos km restantes), sobre o problema completo. Garante a solução
ótima se terminar, mas nestes dados não termina dentro do limite de expansões.

As regras são verificadas pelo apply_stop e pelo check_solution do model.py.
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
    """H1: número de operações que faltam (4 por item)."""
    return sum(DELIVERED - fase for fase in estado.status.values())


def heuristica2(problem, estado):
    """H2: locais por visitar; em empate, km desses locais ao armazém e depois a H1."""
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
    """Objetivo: todos os itens entregues."""
    return all(fase == DELIVERED for fase in estado.status.values())


def chave_estado(problem, estado):
    """Chave que identifica o estado (para detetar estados repetidos)."""
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
    """Hora mais cedo de chegada do veículo ao local.
    Se a última paragem foi a mesma operação no mesmo local, junta-se a ela."""
    ultima = estado.stops[veiculo][-1] if estado.stops[veiculo] else None
    if ultima is not None and ultima.action == operacao and ultima.place == local:
        return ultima.arrival
    return estado.time[veiculo] + problem.travel(estado.place[veiculo], local)


# ---------------------------------------------------------------------------
# Sucessores
# ---------------------------------------------------------------------------
def grupos_recolha(problem, estado, veiculo):
    """Grupos de itens do mesmo produtor que cabem no veículo e têm
    janelas de recolha compatíveis."""
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
        itens.sort(key=lambda r: (r.pickup_end, r.id))     # janelas que fecham primeiro

        # Sementes: os 3 primeiros e o último (limita o número de grupos)
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
    """Gera (novo estado, veículo, operação, itens) para cada movimento válido."""
    for veiculo in problem.vehicles:
        v = veiculo.id
        movimentos = []

        # 1) Descarregar tudo no armazém
        recolhidos = [i for i in estado.status
                      if estado.status[i] == PICKED and estado.owner[i] == v]
        if recolhidos:
            movimentos.append((UNLOAD, recolhidos))

        # 2) Carregar e entregar (as 3 encomendas mais urgentes)
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

        # 3) Recolher um grupo num produtor
        for ids in grupos_recolha(problem, estado, v):
            movimentos.append((PICKUP, ids))

        # Mantêm-se apenas os movimentos válidos
        for operacao, ids in movimentos:
            novo = estado.copy()
            if apply_stop(problem, novo, v, operacao, ids) is None:
                yield novo, v, operacao, ids


def estado_viavel(problem, estado):
    """Verifica, de forma otimista, se o estado ainda pode levar a uma solução
    (recolhas e entregas ainda possíveis dentro das janelas)."""
    armazem = problem.warehouse
    pronto = {}                     # item -> hora mais cedo em que fica pronto no armazém

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
            if t > r.pickup_end:                            # recolha impossível
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
        if all(estado.status[i] == LOADED for i in ids):    # já carregada
            v = estado.owner[ids[0]]
            try:
                chegada = chegada_mais_cedo(problem, estado, v, DELIVER, r.delivery_location)
            except KeyError:
                return False
        else:                                               # ainda por carregar
            try:
                carga = max(pronto[i] for i in ids if i in pronto)
                chegada = carga + problem.travel(armazem, r.delivery_location)
            except (KeyError, ValueError):
                return False

        inicio = max(chegada, max(problem.request[i].delivery_start for i in ids))
        fim = min(problem.request[i].delivery_end for i in ids)
        try:
            if inicio > fim or inicio + problem.travel(r.delivery_location, armazem) > problem.horizon:
                return False                                # entrega impossível
        except KeyError:
            return False
    return True


# ---------------------------------------------------------------------------
# Greedy Best-First Search
# ---------------------------------------------------------------------------
def _uma_passagem(problem, estado_inicial, heuristica, max_expansoes, janela):
    """Uma passagem do Greedy. Com 'janela', só expande movimentos que começam
    até 'janela' depois do movimento mais cedo."""
    rotas_vazias = {v.id: [] for v in problem.vehicles}
    fila = [(heuristica(problem, estado_inicial), 0, estado_inicial, rotas_vazias)]
    visitados = set()
    expandidos = 0
    contador = 0                    # desempate na fila

    while fila and expandidos < max_expansoes:
        # Estado com menor heurística
        _, _, estado, rotas = heapq.heappop(fila)
        chave = chave_estado(problem, estado)
        if chave in visitados:
            continue
        visitados.add(chave)

        # Teste de objetivo
        if terminou(problem, estado):
            inicio = estado_inicial.time[problem.vehicles[0].id]
            valida, _, _ = check_solution(problem, rotas, start_time=inicio)
            if valida:
                return rotas, expandidos
            continue
        expandidos += 1

        # Sucessores (limitados pela janela, se existir)
        movimentos = list(sucessores(problem, estado))
        if janela is not None and movimentos:
            mais_cedo = min(novo.stops[v][-1].start for novo, v, _, _ in movimentos)
            movimentos = [(novo, v, op, ids) for novo, v, op, ids in movimentos
                          if novo.stops[v][-1].start <= mais_cedo + janela]

        # Novos estados viáveis para a fila
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
    Faz uma 1.ª passagem com a janela de 'janela_horas' e, se não encontrar
    solução, uma 2.ª sem janela (cada uma com até 'max_expansoes' expansões).
    Devolve (rotas, nós expandidos); rotas = None se não houver solução.
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
# A*: f(n) = g(n) + h(n), sobre o problema completo
#   g(n): km percorridos; h(n): estimativa admissível dos km restantes
# Reutiliza terminou, chave_estado e estado_viavel do Greedy.
# =============================================================================

def distancias_minimas(problem):
    """Distâncias mínimas entre todos os locais (Floyd-Warshall), calculadas uma vez."""
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
    """H1: soma das distâncias dos veículos ao armazém."""
    d = distancias_minimas(problem)
    return sum(d[estado.place[v], problem.warehouse] for v in estado.place)


def astar_h2(problem, estado):
    """H2: custo mínimo da tarefa pendente mais cara (visitar um produtor ou
    entregar uma encomenda), mais o regresso dos outros veículos."""
    d = distancias_minimas(problem)
    W = problem.warehouse
    regresso = {v: d[estado.place[v], W] for v in estado.place}
    base = sum(regresso.values())                # = astar_h1
    melhor = base

    # Produtores por visitar: ir ao produtor e voltar ao armazém
    for P in {r.pickup_location for r in problem.requests if estado.status[r.id] == NEW}:
        melhor = max(melhor, min(base - regresso[v] + d[estado.place[v], P] + d[P, W]
                                 for v in estado.place))

    # Encomendas por entregar
    for ids in problem.orders.values():
        primeiro = ids[0]
        if estado.status[primeiro] == DELIVERED:
            continue
        C = problem.request[primeiro].delivery_location
        if estado.status[primeiro] == LOADED:    # já carregada: ir ao cliente
            v = estado.owner[primeiro]
            melhor = max(melhor, base - regresso[v] + d[estado.place[v], C] + d[C, W])
        else:                                    # ainda por carregar
            melhor = max(melhor, min(base - regresso[v] + d[estado.place[v], W] + d[W, C] + d[C, W]
                                     for v in estado.place))
    return melhor


def astar_sucessores(problem, estado):
    """
    Todos os movimentos elementares: recolher ou descarregar um item, carregar
    ou entregar uma encomenda completa. Operações iguais seguidas no mesmo
    local juntam-se numa só paragem (apply_stop).
    """
    for veiculo in problem.vehicles:
        v = veiculo.id

        # 1) Recolher um item
        for r in problem.requests:
            if estado.status[r.id] != NEW:
                continue
            novo = estado.copy()
            if apply_stop(problem, novo, v, PICKUP, [r.id]) is None:
                yield novo

        # 2) Descarregar um item
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
    """Converte as paragens do estado em rotas {veículo: [(operação, item), ...]}."""
    rotas = {v.id: [] for v in problem.vehicles}
    for v, paragens in estado.stops.items():
        for paragem in paragens:
            for i in paragem.items:
                rotas[v].append((paragem.action, i))
    return rotas


def _itens_por_fase(estado):
    """Número de itens em cada fase."""
    nomes = (("por recolher", NEW), ("recolhidos", PICKED), ("no armazém", STORED),
             ("carregados", LOADED), ("entregues", DELIVERED))
    return {nome: sum(f == fase for f in estado.status.values()) for nome, fase in nomes}


def a_star_search(problem, estado_inicial=None, heuristica=astar_h2,
                  max_expansoes=20000, max_segundos=None, estatisticas=False):
    """
    A* sobre o problema completo: f = g + h (g = km, h = heurística admissível).
    O objetivo é testado quando o estado sai da fila, o que garante o ótimo.
    Pára ao encontrar a solução ou ao atingir 'max_expansoes' / 'max_segundos'.

    Devolve (rotas, nós expandidos); rotas = None se não houver solução.
    Com estatisticas=True devolve também um dicionário com o tamanho da
    pesquisa (gerados, repetidos, fila) e o estado mais avançado.
    """
    relogio = time.perf_counter()
    inicio = estado_inicial if estado_inicial is not None else problem.initial_state()
    W = problem.warehouse

    def prioridade(estado):
        if terminou(problem, estado):
            # Estado final: h = regresso ao armazém
            h = sum(distancias_minimas(problem)[estado.place[v], W] for v in estado.place)
        else:
            h = heuristica(problem, estado)
        return estado.km + h

    def progresso(estado):
        """Ordem de progresso: itens entregues, operações feitas e menos km."""
        return (sum(f == DELIVERED for f in estado.status.values()),
                sum(estado.status.values()), -estado.km)

    contador = 0                                # desempate na fila
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

        # Ignorar entradas desatualizadas
        chave = chave_estado(problem, estado)
        if estado.km > best_g.get(chave, float("inf")) + 1e-9:
            continue

        if progresso(estado) > progresso(mais_avancado):
            mais_avancado = estado

        # Teste de objetivo à saída da fila
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
            # Poda otimista (não elimina soluções válidas)
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
