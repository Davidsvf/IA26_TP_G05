# Projeto Prático de Inteligência Artificial
## Planeamento de Rotas para Distribuição de Produtos Locais

**Instituição:** Instituto Politécnico do Cávado e do Ave (IPCA)  
**Unidade Curricular:** Inteligência Artificial  
**Ano Letivo:** 2026/2027  
**Grupo:** G05  

### Elementos do Grupo
- David Faria - 31517
- Francisco Campinho - 31079
- Rafael Costa - 31524
- Miguel Marques - 33221

---

## 1. Descrição do Projeto

O projeto planeia as rotas de veículos que **recolhem** produtos alimentares em produtores locais, os levam ao **armazém** (centro de recolha) e os **entregam** aos clientes.

**Objetivo:** encontrar rotas válidas que **minimizem os km percorridos**.

O problema é resolvido e comparado com quatro técnicas de Inteligência Artificial:

- **Greedy Best-First Search** (pesquisa informada)
- **A\*** (pesquisa informada)
- **Hill-Climbing** (pesquisa local)
- **CSP** — Problema de Satisfação de Restrições (backtracking, forward checking e AC-3)

## 2. Definição do Problema

Percurso obrigatório de cada produto:

**Produtor → Armazém → Local de entrega**

Restrições:

- **Janelas de tempo:** a recolha e a entrega começam dentro da janela de cada item.
- **Capacidade:** a carga de um veículo nunca ultrapassa a sua capacidade.
- **Encomendas inteiras:** cada encomenda é carregada e entregue de uma só vez.
- **Armazém:** um produto não pode ficar mais de **24 horas** no armazém.
- **Horizonte:** 2 dias (30 e 31 de outubro de 2026); os veículos começam e acabam no armazém.
- **Tempos de serviço:** 10 min em produtores e clientes, 15 min para descarregar ou carregar no armazém.

**Função de custo:** km totais percorridos, incluindo o regresso ao armazém.

## 3. Datasets

| | tiny | small |
|---|---:|---:|
| Produtores | 3 | 6 |
| Locais de entrega | 5 | 7 |
| Encomendas | 12 | 25 |
| Itens | 40 | 80 |
| Veículos (capacidade) | 2 (150 e 200 kg) | 2 (250 e 300 kg) |
| Localizações | 9 | 14 |

Cada dataset tem 12 ficheiros CSV, incluindo o `rules.csv` com as regras do problema.

## 4. Estrutura do Projeto

```text
IA26_P01_G05/
├── README.md
├── data_tiny/                     # 12 ficheiros CSV do dataset tiny
├── data_small/                    # 12 ficheiros CSV do dataset small
├── src/
│   ├── model.py                   # modelo do problema e verificação das soluções
│   ├── informed.py                # Greedy Best-First Search e A*
│   ├── local_search.py            # Hill-Climbing
│   └── csp.py                     # CSP: backtracking, forward checking e AC-3
└── notebooks/
    ├── 01_Project01_G05.ipynb     # notebook principal: resumo, modelação, resultados e conclusão
    ├── 02_eda_tiny.ipynb          # análise exploratória do tiny
    ├── 02_eda_small.ipynb         # análise exploratória do small
    ├── informed_search.ipynb      # Greedy e A*
    ├── local_search.ipynb         # Hill-Climbing
    └── csp.ipynb                  # CSP
```

### Módulos (`src/`)

- **`model.py`** — descreve o problema e funciona como "juiz" para todos os algoritmos: carrega os CSV (`Problem.from_csv`), aplica as regras de cada paragem (`apply_stop`) e verifica um plano completo (`check_solution`), devolvendo se é válido e os km.
- **`informed.py`** — `greedy_search` (heurísticas H1: operações que faltam; H2: locais que faltam visitar) e `a_star_search` (f = g + h, com heurísticas admissíveis em km).
- **`local_search.py`** — `hill_climbing`, que parte da solução do Greedy e usa três tipos de vizinhos (trocar, mover e passar uma encomenda para o outro veículo).
- **`csp.py`** — `resolver_csp`: variáveis (recolha de cada item e entrega de cada encomenda), domínios (veículo e nº da viagem), restrições, AC-3, forward checking e backtracking, com a heurística de urgência e a restrição soft de menos km.

Uma solução é um dicionário `{veículo: [(operação, item), ...]}`, com as operações `pickup`, `unload`, `load` e `deliver`.

## 5. Como Executar

### Requisitos

- Python 3
- Bibliotecas: pandas, numpy, matplotlib e Jupyter

```bash
python3 -m pip install pandas numpy matplotlib notebook
```

### Executar os notebooks

Na pasta principal do projeto:

```bash
jupyter notebook
```

Abrir os notebooks da pasta `notebooks/` e escolher **Kernel → Restart & Run All**. Os notebooks encontram sozinhos a pasta do projeto (podem ser abertos a partir da raiz ou da pasta `notebooks/`).

Ordem sugerida: `02_eda_tiny` → `informed_search` → `local_search` → `csp` → `01_Project01_G05`.

### Usar os módulos em Python

```python
from src.model import Problem, check_solution, print_solution
from src.informed import greedy_search, heuristica2
from src.local_search import hill_climbing

problem = Problem.from_csv("data_tiny")
solucao, nos = greedy_search(problem, heuristica=heuristica2)
melhor, historico, avaliacoes = hill_climbing(problem, solucao)
print_solution(problem, melhor)
```

## 6. Resultados

Distância (km) das soluções encontradas. Todas as soluções foram confirmadas com o `check_solution`.

| Algoritmo | tiny | small |
|---|---:|---:|
| Greedy H1 | 798,54 | 1070,60 |
| Greedy H2 | 802,24 | 1009,19 |
| A\* H1 / H2 (500 expansões) | sem solução | sem solução |
| Hill-Climbing (a partir do Greedy H1) | 708,96 | 852,38 |
| Hill-Climbing (a partir do Greedy H2) | **636,31** | **832,41** |
| CSP | 700,22 | 891,19 |

- **Greedy:** encontra sempre uma solução válida, em segundos, mas não garante a melhor.
- **A\*:** está implementado corretamente, mas não chega ao fim. O estado inclui horas, cada estado gera dezenas de sucessores e a pesquisa espalha-se em largura (ver `informed_search.ipynb`, secção 8).
- **Hill-Climbing:** dá as melhores soluções, melhorando o Greedy entre 11% e 21%.
- **CSP:** encontra uma solução válida sem precisar de solução inicial, com menos km do que o Greedy.

A análise completa está no notebook `01_Project01_G05.ipynb`.

## 7. Tecnologias Utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python | Modelação do problema e algoritmos |
| Jupyter Notebook | Execução e documentação |
| Git e GitHub | Controlo de versões e colaboração |
