"""
csp.py — o problema como CSP (Problema de Satisfação de Restrições).

O CSP decide com que veículo e em que viagem se faz cada tarefa. Cada veículo
faz viagens em sequência e, em cada viagem, vai aos produtores, ao armazém e
aos clientes. As horas são calculadas pela simulação do model.py.

Variáveis: ("R", i) = recolha do item i; ("E", o) = entrega da encomenda o.
Domínio:   pares (veículo, nº da viagem).

Restrições:
  1. a tarefa cabe no veículo (unária);
  2. as recolhas e as cargas de cada viagem cabem no veículo;
  3. no mesmo veículo, a recolha é na mesma viagem ou antes da entrega (binária);
  4. janelas de tempo, 24 h no armazém e horizonte (verificadas por simulação).
Restrição soft: preferir os valores com menos km.

Algoritmos: AC-3, forward checking e backtracking.
Heurística de variável: urgência (janela que fecha mais cedo) ou MRV.
"""

import time

from src.model import (
    PICKUP, UNLOAD, LOAD, DELIVER,
    simulate, check_solution,
)


# ---------------------------------------------------------------------------
# Formulação: variáveis, domínios e restrições
# ---------------------------------------------------------------------------
class CSP:

    def __init__(self, problem, max_viagens=10):
        self.problem = problem

        # Variáveis: recolha de cada item e entrega de cada encomenda
        self.variaveis = ([("R", item.id) for item in problem.requests]
                          + [("E", encomenda) for encomenda in problem.orders])

        # Peso e fecho da janela de cada tarefa
        self.peso = {}
        self.fecho = {}
        for item in problem.requests:
            self.peso[("R", item.id)] = item.weight_kg
            self.fecho[("R", item.id)] = item.pickup_end
        for encomenda, itens in problem.orders.items():
            self.peso[("E", encomenda)] = sum(problem.request[i].weight_kg for i in itens)
            self.fecho[("E", encomenda)] = min(problem.request[i].delivery_end for i in itens)

        # Vizinhos (precedência): recolha do item <-> entrega da encomenda
        self.vizinhos = {X: set() for X in self.variaveis}
        for encomenda, itens in problem.orders.items():
            for i in itens:
                self.vizinhos[("R", i)].add(("E", encomenda))
                self.vizinhos[("E", encomenda)].add(("R", i))

        # Domínios (restrição 1): (veículo, viagem) com capacidade suficiente
        capacidade = {v.id: v.capacity_kg for v in problem.vehicles}
        self.dominios = {X: [(v, k) for v in capacidade for k in range(max_viagens)
                             if self.peso[X] <= capacidade[v] + 1e-6]
                         for X in self.variaveis}
        self.capacidade = capacidade

    # --- Construção do plano ------------------------------------------------
    def operacoes_viagem(self, tarefas):
        """Operações de uma viagem: recolhas (por produtor), descarga, cargas e
        entregas (por cliente)."""
        p = self.problem
        itens = [i for tipo, i in tarefas if tipo == "R"]
        encomendas = [o for tipo, o in tarefas if tipo == "E"]

        # Recolhas por produtor, primeiro o mais urgente
        por_produtor = {}
        for i in sorted(itens, key=lambda i: (p.request[i].pickup_end, i)):
            por_produtor.setdefault(p.request[i].pickup_location, []).append(i)
        produtores = sorted(por_produtor, key=lambda P: p.request[por_produtor[P][0]].pickup_end)

        # Entregas por cliente, primeiro o mais urgente
        por_cliente = {}
        for o in sorted(encomendas, key=lambda o: (self.fecho[("E", o)], o)):
            por_cliente.setdefault(p.request[p.orders[o][0]].delivery_location, []).append(o)
        clientes = sorted(por_cliente, key=lambda C: self.fecho[("E", por_cliente[C][0])])

        operacoes = []
        for P in produtores:
            operacoes += [(PICKUP, i) for i in por_produtor[P]]
        operacoes += [(UNLOAD, i) for i in sorted(itens)]
        cargas = [o for C in clientes for o in por_cliente[C]]
        operacoes += [(LOAD, i) for o in cargas for i in p.orders[o]]
        operacoes += [(DELIVER, i) for o in cargas for i in p.orders[o]]
        return operacoes

    def rotas(self, atribuicao):
        """Converte a atribuição em rotas {veículo: [(operação, item), ...]},
        com as viagens de cada veículo por ordem."""
        viagens = {}
        for X, valor in atribuicao.items():
            viagens.setdefault(valor, []).append(X)
        rotas = {v.id: [] for v in self.problem.vehicles}
        for (veiculo, k) in sorted(viagens, key=lambda valor: (valor[0], valor[1])):
            rotas[veiculo] += self.operacoes_viagem(viagens[(veiculo, k)])
        return rotas

    def simular(self, atribuicao):
        """Simula o plano (mesmo incompleto). Devolve os km, ou None se for inválido."""
        completo = len(atribuicao) == len(self.variaveis)
        valida, estado, _ = simulate(self.problem, self.rotas(atribuicao),
                                     check_all=completo, parcial=not completo)
        return round(estado.km, 2) if valida else None

    # --- Restrições verificadas sem simulação --------------------------------
    def precedencia(self, X, valor_x, Y, valor_y):
        """Restrição 3: no mesmo veículo, a recolha é na mesma viagem ou antes."""
        if X[0] == "E":                          # ordem (recolha, entrega)
            X, valor_x, Y, valor_y = Y, valor_y, X, valor_x
        if valor_x[0] != valor_y[0]:
            return True                          # veículos diferentes: verificado na simulação
        return valor_x[1] <= valor_y[1]

    def capacidade_viagem(self, X, valor, atribuicao):
        """Restrição 2: as recolhas da viagem cabem no veículo, e as cargas também."""
        tipo = X[0]
        carga = self.peso[X] + sum(self.peso[Y] for Y, w in atribuicao.items()
                                   if w == valor and Y[0] == tipo)
        return carga <= self.capacidade[valor[0]] + 1e-6

    def viagem_seguinte(self, valor, atribuicao):
        """Quebra de simetria: a viagem k só pode ser usada depois das anteriores."""
        veiculo, k = valor
        usadas = {w[1] for w in atribuicao.values() if w[0] == veiculo}
        return k <= len(usadas)

    def baratas_ok(self, X, valor, atribuicao):
        """Restrições que não precisam de simulação."""
        if not self.viagem_seguinte(valor, atribuicao):
            return False
        if not self.capacidade_viagem(X, valor, atribuicao):
            return False
        return all(self.precedencia(X, valor, Y, atribuicao[Y])
                   for Y in self.vizinhos[X] if Y in atribuicao)


# ---------------------------------------------------------------------------
# AC-3: consistência de arcos
# ---------------------------------------------------------------------------
def _compativeis(csp, X, x, Y, y):
    """Compatibilidade entre X=x e Y=y (precedência e capacidade da viagem)."""
    if Y in csp.vizinhos[X] and not csp.precedencia(X, x, Y, y):
        return False
    if x == y and X[0] == Y[0]:                  # mesma viagem e mesmo tipo: capacidade
        return csp.peso[X] + csp.peso[Y] <= csp.capacidade[x[0]] + 1e-6
    return True


def ac3(csp, dominios):
    """Remove os valores sem valor compatível nas variáveis vizinhas.
    Devolve (consistente?, nº de valores removidos)."""
    fila = [(X, Y) for X in csp.variaveis for Y in csp.vizinhos[X]]
    removidos = 0
    while fila:
        X, Y = fila.pop()
        mudou = False
        for x in list(dominios[X]):
            if not any(_compativeis(csp, X, x, Y, y) for y in dominios[Y]):
                dominios[X].remove(x)
                removidos += 1
                mudou = True
        if mudou:
            if not dominios[X]:
                return False, removidos          # domínio vazio
            fila += [(Z, X) for Z in csp.vizinhos[X] if Z != Y]
    return True, removidos


# ---------------------------------------------------------------------------
# Heurísticas
# ---------------------------------------------------------------------------
def escolher_variavel(csp, atribuicao, dominios, ordem):
    """Escolhe a próxima variável:
      - "urgencia": janela que fecha mais cedo (por omissão);
      - "mrv": menos valores possíveis (desempate: urgência e grau);
      - "lista": ordem da lista (sem heurística)."""
    livres = [X for X in csp.variaveis if X not in atribuicao]
    if ordem == "lista":
        return livres[0]
    if ordem == "mrv":
        grau = lambda X: sum(1 for Y in csp.vizinhos[X] if Y not in atribuicao)
        return min(livres, key=lambda X: (len(dominios[X]), csp.fecho[X], -grau(X)))
    return min(livres, key=lambda X: csp.fecho[X])


def valores_possiveis(csp, X, atribuicao, dominios, soft):
    """Valores de X que cumprem todas as restrições, incluindo a simulação.
    Com soft=True, ordenados pelos km do plano (restrição soft)."""
    possiveis = []
    for valor in dominios[X]:
        if not csp.baratas_ok(X, valor, atribuicao):
            continue
        km = csp.simular({**atribuicao, X: valor})
        if km is not None:
            possiveis.append((km, valor))
    if soft:
        possiveis.sort()
    return [valor for _, valor in possiveis]


# ---------------------------------------------------------------------------
# Forward checking
# ---------------------------------------------------------------------------
def forward_checking(csp, atribuicao, dominios):
    """Remove dos domínios por atribuir os valores que violam a capacidade da
    viagem ou a precedência (a ordem das viagens não entra, porque uma viagem
    ainda indisponível pode ficar disponível mais tarde).
    Devolve os novos domínios, ou None se algum ficar vazio."""
    novos = {}
    for Y in csp.variaveis:
        if Y in atribuicao:
            novos[Y] = dominios[Y]
            continue
        restantes = [y for y in dominios[Y]
                     if csp.capacidade_viagem(Y, y, atribuicao)
                     and all(csp.precedencia(Y, y, X, atribuicao[X])
                             for X in csp.vizinhos[Y] if X in atribuicao)]
        if not restantes:
            return None
        novos[Y] = restantes
    return novos


# ---------------------------------------------------------------------------
# Backtracking
# ---------------------------------------------------------------------------
def backtracking(csp, dominios, usar_fc=True, ordem="urgencia", soft=True, max_segundos=120):
    """Backtracking: atribui uma variável de cada vez e volta atrás quando não
    há valor possível. Devolve (atribuição ou None, estatísticas)."""
    estat = {"atribuicoes": 0, "backtracks": 0, "esgotou_tempo": False}
    limite = time.time() + max_segundos
    atribuicao = {}

    def procurar(dominios):
        if time.time() > limite:
            estat["esgotou_tempo"] = True
            return False
        if len(atribuicao) == len(csp.variaveis):
            return True                          # plano completo (já simulado)

        X = escolher_variavel(csp, atribuicao, dominios, ordem)
        for valor in valores_possiveis(csp, X, atribuicao, dominios, soft):
            atribuicao[X] = valor
            estat["atribuicoes"] += 1

            novos = forward_checking(csp, atribuicao, dominios) if usar_fc else dominios
            if novos is not None and procurar(novos):
                return True

            del atribuicao[X]                    # backtrack
            estat["backtracks"] += 1
            if estat["esgotou_tempo"]:
                return False
        return False

    encontrou = procurar(dominios)
    return (dict(atribuicao) if encontrou else None), estat


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------
def resolver_csp(problem, max_viagens=10, usar_ac3=True, usar_fc=True,
                 ordem="urgencia", soft=True, max_segundos=120):
    """
    Constrói e resolve o CSP (AC-3 e backtracking).
    Devolve (rotas ou None, atribuição ou None, estatísticas).
    """
    inicio = time.time()
    csp = CSP(problem, max_viagens)
    dominios = {X: list(D) for X, D in csp.dominios.items()}
    estat = {"variaveis": len(csp.variaveis),
             "valores_iniciais": sum(len(D) for D in dominios.values()),
             "removidos_ac3": 0}

    if usar_ac3:
        possivel, estat["removidos_ac3"] = ac3(csp, dominios)
        if not possivel:
            estat.update(atribuicoes=0, backtracks=0, esgotou_tempo=False,
                         segundos=round(time.time() - inicio, 2))
            return None, None, estat

    atribuicao, pesquisa = backtracking(csp, dominios, usar_fc, ordem, soft, max_segundos)
    estat.update(pesquisa)
    estat["segundos"] = round(time.time() - inicio, 2)
    rotas = csp.rotas(atribuicao) if atribuicao else None
    return rotas, atribuicao, estat
