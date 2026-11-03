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


# ---------------------------------------------------------------------------
# A* PARCIAL — código independente do Greedy acima
# ---------------------------------------------------------------------------
from src.model import Problem


def _astar_distancias(problem):
    """Menores distâncias em km entre os locais usados na encomenda."""
    locais = ({problem.warehouse}
              | {r.pickup_location for r in problem.requests}
              | {r.delivery_location for r in problem.requests})

    distancias = {
        (origem, destino): (0.0 if origem == destino
                            else problem.km.get((origem, destino), float("inf")))
        for origem in locais for destino in locais
    }

    for intermedio in locais:
        for origem in locais:
            for destino in locais:
                distancias[origem, destino] = min(
                    distancias[origem, destino],
                    distancias[origem, intermedio]
                    + distancias[intermedio, destino]
                )
    return distancias


def _astar_heuristica(problem, estado, distancias):
    """Estimativa inferior dos km que faltam para concluir a encomenda."""
    veiculo = problem.vehicles[0].id
    armazem = problem.warehouse
    posicao = estado.place[veiculo]
    estimativa = distancias[posicao, armazem]

    for item in problem.requests:
        fase = estado.status[item.id]
        if fase == NEW:
            km = (distancias[posicao, item.pickup_location]
                  + distancias[item.pickup_location, armazem]
                  + distancias[armazem, item.delivery_location]
                  + distancias[item.delivery_location, armazem])
        elif fase == PICKED:
            km = (distancias[posicao, armazem]
                  + distancias[armazem, item.delivery_location]
                  + distancias[item.delivery_location, armazem])
        elif fase == STORED:
            km = (distancias[armazem, item.delivery_location]
                  + distancias[item.delivery_location, armazem])
        elif fase == LOADED:
            km = (distancias[posicao, item.delivery_location]
                  + distancias[item.delivery_location, armazem])
        else:
            km = 0.0
        estimativa = max(estimativa, km)

    return estimativa


def _astar_sucessores(problem, estado):
    """Gera movimentos para uma única encomenda, respeitando apply_stop."""
    veiculo = problem.vehicles[0].id
    movimentos = []

    recolhidos = [i for i in estado.status if estado.status[i] == PICKED]
    if recolhidos:
        movimentos.append((UNLOAD, recolhidos))

    for ids in problem.orders.values():
        if all(estado.status[i] == STORED for i in ids):
            movimentos.append((LOAD, ids))
        if all(estado.status[i] == LOADED for i in ids):
            movimentos.append((DELIVER, ids))

    produtores = {}
    for item in problem.requests:
        if estado.status[item.id] == NEW:
            produtores.setdefault(item.pickup_location, []).append(item.id)

    for ids in produtores.values():
        movimentos.append((PICKUP, ids))
        if len(ids) > 1:
            for i in ids:
                movimentos.append((PICKUP, [i]))

    for operacao, ids in movimentos:
        novo = estado.copy()
        if apply_stop(problem, novo, veiculo, operacao, ids) is None:
            yield novo


def _astar_uma_encomenda(problem, encomenda, veiculo, max_expansoes):
    """A* para uma encomenda com um veículo, sujeito às regras originais."""
    itens = [problem.request[i] for i in problem.orders[encomenda]]
    reduzido = Problem(itens, [veiculo], problem.km, problem.minutes,
                       problem.warehouse, problem.rules, problem.names)

    distancias = _astar_distancias(reduzido)
    inicial = reduzido.initial_state()
    fila = [(_astar_heuristica(reduzido, inicial, distancias), 0, 0, inicial)]
    melhores = {chave_estado(reduzido, inicial): 0.0}
    contador = 0
    expandidos = 0

    while fila and (max_expansoes is None or expandidos < max_expansoes):
        _, _, _, atual = heapq.heappop(fila)
        chave = chave_estado(reduzido, atual)
        if atual.km > melhores.get(chave, float("inf")) + 1e-9:
            continue

        if terminou(reduzido, atual):
            rota = [(p.action, i) for p in atual.stops[veiculo.id] for i in p.items]
            valida, _, _ = check_solution(reduzido, {veiculo.id: rota})
            if valida:
                return rota, expandidos
            continue

        expandidos += 1
        for novo in _astar_sucessores(reduzido, atual):
            chave = chave_estado(reduzido, novo)
            if novo.km >= melhores.get(chave, float("inf")) - 1e-9:
                continue
            melhores[chave] = novo.km
            contador += 1
            entregues = sum(fase == DELIVERED for fase in novo.status.values())
            f = novo.km + _astar_heuristica(reduzido, novo, distancias)
            heapq.heappush(fila, (f, -entregues, contador, novo))

    return None, expandidos


def _astar_grupo(problem, encomendas, veiculo, max_expansoes):
    """A* para várias encomendas transportadas pelo mesmo veículo.

    Ao pesquisar as encomendas juntas, pode intercalar recolhas,
    descargas, cargas e entregas, partilhando deslocações.
    """
    ids = [i for e in encomendas for i in problem.orders[e]]
    itens = [problem.request[i] for i in ids]
    reduzido = Problem(itens, [veiculo], problem.km, problem.minutes,
                       problem.warehouse, problem.rules, problem.names)

    distancias = _astar_distancias(reduzido)
    inicial = reduzido.initial_state()
    fila = [(_astar_heuristica(reduzido, inicial, distancias), 0, inicial)]
    melhores = {chave_estado(reduzido, inicial): 0.0}
    contador = 0
    expandidos = 0

    while fila and (max_expansoes is None or expandidos < max_expansoes):
        _, _, atual = heapq.heappop(fila)
        chave = chave_estado(reduzido, atual)
        if atual.km > melhores.get(chave, float("inf")) + 1e-9:
            continue

        if terminou(reduzido, atual):
            rota = [(p.action, i) for p in atual.stops[veiculo.id]
                    for i in p.items]
            valido, _, _ = check_solution(reduzido, {veiculo.id: rota})
            if valido:
                return rota, expandidos
            continue

        expandidos += 1
        for novo in _astar_sucessores(reduzido, atual):
            chave = chave_estado(reduzido, novo)
            if novo.km >= melhores.get(chave, float("inf")) - 1e-9:
                continue
            melhores[chave] = novo.km
            contador += 1
            custo_estimado = novo.km + _astar_heuristica(reduzido, novo, distancias)
            heapq.heappush(fila, (custo_estimado, contador, novo))

    return None, expandidos


def a_star_parcial(problem, max_encomendas=8, max_expansoes=80,
                   largura=20, vizinhos_pares=3, max_validacoes=4000):
    """A* parcial: encomendas isoladas/pares + combinação em feixe.

    Guarda várias combinações, experimenta inserir os blocos em posições
    diferentes e valida sempre com check_solution(check_all=False).
    Prioriza o número de encomendas e depois os km.
    Não garante a solução ótima nem o máximo de encomendas.

    max_expansoes é o limite POR pesquisa A* individual/de par.
    """
    if max_encomendas < 1 or largura < 1:
        return None, 0

    veiculos = [v.id for v in problem.vehicles]
    encomendas = list(problem.orders)
    total_expandidos = 0
    blocos_possiveis = []

    # A* para cada encomenda individual.
    for encomenda in encomendas:
        for veiculo in problem.vehicles:
            rota, n = _astar_uma_encomenda(
                problem, encomenda, veiculo, max_expansoes
            )
            total_expandidos += n
            if rota is not None:
                blocos_possiveis.append((frozenset((encomenda,)),
                                        veiculo.id, rota))

    # A* para pares com produtores/clientes próximos. Os pares permitem
    # intercalar operações e, por isso, aumentar o número de entregas.
    if max_encomendas >= 2 and vizinhos_pares > 0:
        pares_vistos = set()
        for encomenda in encomendas:
            item_a = problem.request[problem.orders[encomenda][0]]
            proximidades = []
            for outra in encomendas:
                if outra == encomenda:
                    continue
                item_b = problem.request[problem.orders[outra][0]]
                km = (problem.distance(item_a.pickup_location,
                                       item_b.pickup_location)
                      + problem.distance(item_a.delivery_location,
                                         item_b.delivery_location))
                proximidades.append((km, outra))

            for _, outra in sorted(proximidades)[:vizinhos_pares]:
                par = tuple(sorted((encomenda, outra)))
                if par in pares_vistos:
                    continue
                pares_vistos.add(par)
                for veiculo in problem.vehicles:
                    rota, n = _astar_grupo(
                        problem, par, veiculo, max_expansoes
                    )
                    total_expandidos += n
                    if rota is not None:
                        blocos_possiveis.append((frozenset(par),
                                                veiculo.id, rota))

    def construir_rotas(blocos):
        return {v: [operacao for _, rota in blocos[v] for operacao in rota]
                for v in veiculos}

    # Cada solução mantém os blocos de encomendas por veículo.
    feixe = [(0.0, {v: () for v in veiculos}, frozenset())]
    melhor = None
    melhor_quantidade = 0
    melhor_km = float("inf")
    validacoes = 0
    limite_atingido = False

    for _ in range(max_encomendas):
        candidatos = {}
        for _, blocos, feitas in feixe:
            for grupo, veiculo, rota in blocos_possiveis:
                if grupo & feitas or len(feitas) + len(grupo) > max_encomendas:
                    continue

                for posicao in range(len(blocos[veiculo]) + 1):
                    if validacoes >= max_validacoes:
                        limite_atingido = True
                        break
                    novos_blocos = blocos.copy()
                    novos_blocos[veiculo] = (
                        blocos[veiculo][:posicao]
                        + ((grupo, rota),)
                        + blocos[veiculo][posicao:]
                    )
                    assinatura = tuple(
                        (v, tuple(tuple(sorted(g)) for g, _ in novos_blocos[v]))
                        for v in veiculos
                    )
                    if assinatura in candidatos:
                        continue

                    solucao = construir_rotas(novos_blocos)
                    validacoes += 1
                    valido, km, _ = check_solution(
                        problem, solucao, check_all=False
                    )
                    if valido:
                        candidatos[assinatura] = (
                            km, novos_blocos, feitas | grupo
                        )
                if limite_atingido:
                    break
            if limite_atingido:
                break

        if not candidatos:
            break

        feixe = sorted(candidatos.values(),
                       key=lambda x: (-len(x[2]), x[0]))[:largura]
        for km, blocos, feitas in feixe:
            if len(feitas) > melhor_quantidade or (
                    len(feitas) == melhor_quantidade and km < melhor_km):
                melhor_quantidade = len(feitas)
                melhor_km = km
                melhor = construir_rotas(blocos)

        if melhor_quantidade >= max_encomendas or limite_atingido:
            break

    return melhor, total_expandidos