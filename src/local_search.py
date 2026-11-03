"""
local_search.py — pesquisa local: Hill-Climbing a partir da solução do Greedy.

Em cada iteração passa para o melhor vizinho válido (menos km) e pára quando
nenhum vizinho é melhor (ótimo local).

Vizinhos:
  - trocar: troca duas paragens de um veículo;
  - mover:  muda uma paragem de posição;
  - passar: passa a carga e a entrega de uma encomenda para o outro veículo.
"""

from src.model import check_solution, LOAD, DELIVER


def dividir_paragens(problem, rota):
    """Agrupa operações seguidas do mesmo tipo e local numa paragem."""
    paragens = []

    for operacao, item in rota:
        local = problem.place_of(operacao, problem.request[item])

        if paragens:
            op_ant, item_ant = paragens[-1][0]
            local_ant = problem.place_of(op_ant, problem.request[item_ant])

            if operacao == op_ant and local == local_ant:
                paragens[-1].append((operacao, item))
                continue

        paragens.append([(operacao, item)])

    return paragens


def juntar(paragens):
    """Converte uma lista de paragens numa rota."""
    return [op for paragem in paragens for op in paragem]


def copiar(solucao):
    """Cópia independente de uma solução."""
    return {v: rota.copy() for v, rota in solucao.items()}


# ---------------------------------------------------------------------------
# Vizinhos
# ---------------------------------------------------------------------------
def vizinhos_trocar(problem, solucao):
    """Troca duas paragens na rota de um veículo."""
    for veiculo, rota in solucao.items():
        paragens = dividir_paragens(problem, rota)

        for i in range(len(paragens)):
            for j in range(i + 1, len(paragens)):
                novas = paragens.copy()
                novas[i], novas[j] = novas[j], novas[i]

                candidato = copiar(solucao)
                candidato[veiculo] = juntar(novas)
                yield candidato


def vizinhos_mover(problem, solucao):
    """Muda uma paragem para outra posição na rota do mesmo veículo."""
    for veiculo, rota in solucao.items():
        paragens = dividir_paragens(problem, rota)

        for i in range(len(paragens)):
            resto = paragens[:i] + paragens[i + 1:]

            for j in range(len(resto) + 1):
                if j == i:                      # mesma posição
                    continue
                novas = resto[:j] + [paragens[i]] + resto[j:]

                candidato = copiar(solucao)
                candidato[veiculo] = juntar(novas)
                yield candidato


def vizinhos_passar(problem, solucao):
    """Passa a carga e a entrega de uma encomenda para outro veículo,
    inserindo-as juntas numa posição da sua rota."""
    for origem, rota in solucao.items():
        encomendas = sorted({problem.request[i].order_id
                             for op, i in rota if op == DELIVER})

        for encomenda in encomendas:
            itens = problem.orders[encomenda]

            # Retira a carga e a entrega da rota de origem
            nova_origem = [(op, i) for op, i in rota
                           if not (op in (LOAD, DELIVER) and i in itens)]
            bloco = [(LOAD, i) for i in itens] + [(DELIVER, i) for i in itens]

            for destino, rota_destino in solucao.items():
                if destino == origem:
                    continue
                paragens = dividir_paragens(problem, rota_destino)

                for j in range(len(paragens) + 1):
                    candidato = copiar(solucao)
                    candidato[origem] = nova_origem
                    candidato[destino] = juntar(paragens[:j]) + bloco + juntar(paragens[j:])
                    yield candidato


def gerar_vizinhos(problem, solucao):
    """Todos os vizinhos de uma solução."""
    yield from vizinhos_trocar(problem, solucao)
    yield from vizinhos_mover(problem, solucao)
    yield from vizinhos_passar(problem, solucao)


# ---------------------------------------------------------------------------
# Hill-Climbing
# ---------------------------------------------------------------------------
def hill_climbing(problem, solucao_inicial, max_iteracoes=None):
    """
    Hill-Climbing : em cada iteração escolhe o melhor
    vizinho válido; para num ótimo local ou ao atingir max_iteracoes.
    Devolve (melhor solução, custo por iteração, nº de soluções avaliadas).
    """
    if solucao_inicial is None:
        raise ValueError("Não há solução inicial (o Greedy não encontrou solução).")

    valida, custo, mensagem = check_solution(problem, solucao_inicial)
    if not valida:
        raise ValueError(f"Solução inicial inválida: {mensagem}")

    melhor = copiar(solucao_inicial)
    historico = [custo]                         # custo por iteração
    avaliacoes = 0

    while max_iteracoes is None or len(historico) - 1 < max_iteracoes:
        melhor_vizinho = None
        custo_vizinho = custo

        for candidato in gerar_vizinhos(problem, melhor):
            valida, novo_custo, _ = check_solution(problem, candidato)
            avaliacoes += 1

            if valida and novo_custo < custo_vizinho - 0.000001:
                melhor_vizinho = candidato
                custo_vizinho = novo_custo

        if melhor_vizinho is None:              # ótimo local
            break

        melhor = melhor_vizinho
        custo = custo_vizinho
        historico.append(custo)

    return melhor, historico, avaliacoes
