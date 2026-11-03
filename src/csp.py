"""
csp.py  --  Problema de Satisfação de Restrições (CSP).

Ideia: em vez de construir a rota passo a passo, o CSP decide COM QUE VEÍCULO
e EM QUE VIAGEM se faz cada tarefa, respeitando todas as restrições. Não
procura a solução com menos km: procura uma solução VIÁVEL (os km entram como
restrição "soft", através da ordem em que se experimentam os valores).

Viagens em sequência: cada veículo faz a 1.ª viagem, depois a 2.ª, depois a
3.ª... Não se marcam horas: cada viagem começa logo que a anterior acaba, e
as horas são calculadas pela simulação do model.py (igual ao check_solution).
Em cada viagem, a ordem é sempre:
    produtores (recolher) -> armazém (descarregar e carregar) -> clientes (entregar)

Variáveis
  - ("R", i): recolha do item i               (40 no tiny, 80 no small)
  - ("E", o): entrega da encomenda o          (12 no tiny, 25 no small)

Domínio de cada variável: pares (veículo, nº da viagem).

Restrições (hard)
  1. Capacidade (unária): a tarefa, sozinha, cabe no veículo.
  2. Capacidade da viagem: o que se recolhe numa viagem cabe no veículo, e o
     que se carrega também.
  3. Precedência (binária, entre a recolha de um item e a entrega da sua
     encomenda): se forem no mesmo veículo, a recolha é na mesma viagem ou
     numa viagem anterior.
  4. Janelas de tempo, máximo de 24h no armazém, horizonte e ordem entre
     veículos (um veículo pode ter de esperar que o outro descarregue): são
     verificadas simulando o plano com o model.py.
  No fim, o plano completo é validado pelo check_solution.

Restrição soft
  - Menos km: em cada variável, experimentam-se primeiro os valores que dão
    menos km no total (por exemplo, juntar a uma viagem que já vai ao mesmo
    produtor ou cliente).

Algoritmos: backtracking, forward checking e AC-3.
Heurísticas
  - Escolha da variável: urgência (a tarefa cuja janela fecha mais cedo), que
    constrói o plano "pela ordem do relógio". Para comparação, também MRV
    (menos valores possíveis, com desempate pela urgência e pelo grau).
  - Escolha do valor: menos km primeiro (restrição soft).
"""

import time

from src.model import (
    PICKUP, UNLOAD, LOAD, DELIVER,
    simulate, check_solution,
)


# ---------------------------------------------------------------------------
# O problema como CSP: variáveis, domínios e restrições
# ---------------------------------------------------------------------------
class CSP:

    def __init__(self, problem, max_viagens=10):
        self.problem = problem

        # Variáveis: recolha de cada item e entrega de cada encomenda
        self.variaveis = ([("R", item.id) for item in problem.requests]
                          + [("E", encomenda) for encomenda in problem.orders])

        # Peso e "urgência" (fecho da janela) de cada tarefa
        self.peso = {}
        self.fecho = {}
        for item in problem.requests:
            self.peso[("R", item.id)] = item.weight_kg
            self.fecho[("R", item.id)] = item.pickup_end
        for encomenda, itens in problem.orders.items():
            self.peso[("E", encomenda)] = sum(problem.request[i].weight_kg for i in itens)
            self.fecho[("E", encomenda)] = min(problem.request[i].delivery_end for i in itens)

        # Vizinhos pela precedência: recolha de um item <-> entrega da sua encomenda
        self.vizinhos = {X: set() for X in self.variaveis}
        for encomenda, itens in problem.orders.items():
            for i in itens:
                self.vizinhos[("R", i)].add(("E", encomenda))
                self.vizinhos[("E", encomenda)].add(("R", i))

        # Domínios (restrição 1): (veículo, viagem) em que a tarefa cabe no veículo
        capacidade = {v.id: v.capacity_kg for v in problem.vehicles}
        self.dominios = {X: [(v, k) for v in capacidade for k in range(max_viagens)
                             if self.peso[X] <= capacidade[v] + 1e-6]
                         for X in self.variaveis}
        self.capacidade = capacidade

    # --- Construir o plano --------------------------------------------------
    def operacoes_viagem(self, tarefas):
        """Operações de uma viagem, por esta ordem: recolher (por produtor),
        descarregar, carregar e entregar (por cliente)."""
        p = self.problem
        itens = [i for tipo, i in tarefas if tipo == "R"]
        encomendas = [o for tipo, o in tarefas if tipo == "E"]

        # Recolhas agrupadas por produtor; primeiro o produtor cuja janela fecha mais cedo
        por_produtor = {}
        for i in sorted(itens, key=lambda i: (p.request[i].pickup_end, i)):
            por_produtor.setdefault(p.request[i].pickup_location, []).append(i)
        produtores = sorted(por_produtor, key=lambda P: p.request[por_produtor[P][0]].pickup_end)

        # Entregas agrupadas por cliente; primeiro o cliente cuja janela fecha mais cedo
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
        """Plano no formato do model.py: {veículo: [(operação, item), ...]}.
        Cada veículo faz as suas viagens pela ordem (1.ª, 2.ª, ...)."""
        viagens = {}
        for X, valor in atribuicao.items():
            viagens.setdefault(valor, []).append(X)
        rotas = {v.id: [] for v in self.problem.vehicles}
        for (veiculo, k) in sorted(viagens, key=lambda valor: (valor[0], valor[1])):
            rotas[veiculo] += self.operacoes_viagem(viagens[(veiculo, k)])
        return rotas

    def simular(self, atribuicao):
        """Simula o plano (mesmo incompleto) com o model.py.
        Devolve os km, ou None se violar alguma regra."""
        completo = len(atribuicao) == len(self.variaveis)
        valida, estado, _ = simulate(self.problem, self.rotas(atribuicao),
                                     check_all=completo, parcial=not completo)
        return round(estado.km, 2) if valida else None

    # --- Restrições "baratas" (não precisam de simulação) --------------------
    def precedencia(self, X, valor_x, Y, valor_y):
        """Restrição 3: no mesmo veículo, a recolha é na mesma viagem ou antes."""
        if X[0] == "E":                          # pôr sempre (recolha, entrega)
            X, valor_x, Y, valor_y = Y, valor_y, X, valor_x
        if valor_x[0] != valor_y[0]:
            return True                          # veículos diferentes: vê-se na simulação
        return valor_x[1] <= valor_y[1]

    def capacidade_viagem(self, X, valor, atribuicao):
        """Restrição 2: as recolhas da viagem cabem no veículo, e as cargas também."""
        tipo = X[0]
        carga = self.peso[X] + sum(self.peso[Y] for Y, w in atribuicao.items()
                                   if w == valor and Y[0] == tipo)
        return carga <= self.capacidade[valor[0]] + 1e-6

    def viagem_seguinte(self, valor, atribuicao):
        """As viagens de um veículo usam-se por ordem: só se abre a viagem k
        se as anteriores já tiverem tarefas (evita planos repetidos)."""
        veiculo, k = valor
        usadas = {w[1] for w in atribuicao.values() if w[0] == veiculo}
        return k <= len(usadas)

    def baratas_ok(self, X, valor, atribuicao):
        """Todas as restrições que não precisam de simulação."""
        if not self.viagem_seguinte(valor, atribuicao):
            return False
        if not self.capacidade_viagem(X, valor, atribuicao):
            return False
        return all(self.precedencia(X, valor, Y, atribuicao[Y])
                   for Y in self.vizinhos[X] if Y in atribuicao)


# ---------------------------------------------------------------------------
# AC-3: consistência de arcos (antes da pesquisa)
# ---------------------------------------------------------------------------
def _compativeis(csp, X, x, Y, y):
    """Restrição binária entre X=x e Y=y (precedência e capacidade da viagem)."""
    if Y in csp.vizinhos[X] and not csp.precedencia(X, x, Y, y):
        return False
    if x == y and X[0] == Y[0]:                  # mesma viagem e mesmo tipo: cabem juntas?
        return csp.peso[X] + csp.peso[Y] <= csp.capacidade[x[0]] + 1e-6
    return True


def ac3(csp, dominios):
    """Tira de cada domínio os valores que não têm 'apoio' nas variáveis
    vizinhas (para cada valor de X tem de existir um valor de Y compatível).
    Devolve (consegue continuar?, nº de valores removidos)."""
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
                return False, removidos          # domínio vazio: não há solução
            fila += [(Z, X) for Z in csp.vizinhos[X] if Z != Y]
    return True, removidos


# ---------------------------------------------------------------------------
# Heurísticas
# ---------------------------------------------------------------------------
def escolher_variavel(csp, atribuicao, dominios, ordem):
    """Que variável atribuir a seguir:
      - "urgencia": a tarefa cuja janela fecha mais cedo (por omissão);
      - "mrv": a que tem menos valores possíveis (desempate: urgência e grau);
      - "lista": a primeira por atribuir, pela ordem da lista (sem heurística)."""
    livres = [X for X in csp.variaveis if X not in atribuicao]
    if ordem == "lista":
        return livres[0]
    if ordem == "mrv":
        grau = lambda X: sum(1 for Y in csp.vizinhos[X] if Y not in atribuicao)
        return min(livres, key=lambda X: (len(dominios[X]), csp.fecho[X], -grau(X)))
    return min(livres, key=lambda X: csp.fecho[X])


def valores_possiveis(csp, X, atribuicao, dominios, soft):
    """Valores de X que cumprem TODAS as restrições (incluindo a simulação).
    Restrição soft (menos km): ordenados pelos km do plano com esse valor."""
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
    """Depois de uma atribuição, tira dos domínios das variáveis por atribuir
    os valores que deixaram de cumprir a capacidade da viagem ou a precedência.
    Estas duas restrições, uma vez falhadas, nunca mais voltam a ser cumpridas.
    (A regra "as viagens abrem-se por ordem" não entra aqui: uma viagem que
    ainda não pode ser usada pode passar a poder mais tarde.)
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
    """Atribui um valor a uma variável de cada vez; quando não há valor
    possível, volta atrás (backtrack). Devolve (atribuição ou None, estatísticas)."""
    estat = {"atribuicoes": 0, "backtracks": 0, "esgotou_tempo": False}
    limite = time.time() + max_segundos
    atribuicao = {}

    def procurar(dominios):
        if time.time() > limite:
            estat["esgotou_tempo"] = True
            return False
        if len(atribuicao) == len(csp.variaveis):
            return True                          # o último valor já foi simulado completo

        X = escolher_variavel(csp, atribuicao, dominios, ordem)
        for valor in valores_possiveis(csp, X, atribuicao, dominios, soft):
            atribuicao[X] = valor
            estat["atribuicoes"] += 1

            novos = forward_checking(csp, atribuicao, dominios) if usar_fc else dominios
            if novos is not None and procurar(novos):
                return True

            del atribuicao[X]                    # não deu: voltar atrás
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
    Constrói o CSP e resolve-o.
    Devolve (rotas ou None, atribuição ou None, estatísticas).
    As rotas estão no formato do model.py: {veículo: [(operação, item), ...]}
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
