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

O presente projeto tem como objetivo desenvolver um agente inteligente para o planeamento de rotas de recolha e distribuição de produtos alimentares locais.

O problema envolve produtores que disponibilizam produtos, clientes que realizam encomendas e veículos responsáveis pelo transporte entre diferentes localizações.

Pretende-se encontrar rotas válidas e eficientes, respeitando as restrições logísticas e temporais estabelecidas no enunciado.

Para a resolução do problema, serão utilizadas e comparadas diferentes técnicas de Inteligência Artificial:

- **Greedy Best-First Search**
- **A\* Search**
- **Hill-Climbing**
- **Constraint Satisfaction Problem (CSP)**

## 2. Definição do Problema

O planeamento das rotas deve respeitar o seguinte percurso obrigatório:

**Produtor → Centro de Recolha → Local de Entrega**

As principais restrições do problema incluem:

- **Janelas temporais:** as operações de recolha e entrega devem começar dentro dos intervalos definidos.
- **Capacidade dos veículos:** a carga transportada não pode ultrapassar a capacidade máxima permitida.
- **Ordem das operações:** os produtos devem ser recolhidos e passar pelo centro de recolha antes da entrega.
- **Tempos de serviço:** devem ser considerados os tempos necessários para recolha, entrega, carga e descarga.
- **Armazenamento:** os produtos não podem permanecer no centro de recolha durante mais de 24 horas.
- **Distâncias e tempos de viagem:** devem ser considerados os valores presentes na matriz de distâncias.

O objetivo será encontrar soluções viáveis e eficientes, avaliadas através de uma função de custo a definir na fase de modelação.

## 3. Dataset

O projeto utiliza um dataset sintético fornecido pelo docente, organizado em ficheiros CSV.

Nesta fase é utilizado o dataset `small`, composto por:

| Entidade | Quantidade |
|---|---:|
| Produtores | 6 |
| Clientes | 21 |
| Produtos | 31 |
| Encomendas | 25 |
| Itens de encomenda | 80 |
| Centros de recolha | 1 |
| Veículos | 2 |
| Localizações | 14 |
| Pares origem-destino | 182 |

O dataset inclui também o ficheiro `rules.csv`, que contém as regras operacionais do problema.

### Análise Exploratória dos Dados (EDA)

Foi realizada uma análise exploratória utilizando Python, Pandas, NumPy e Matplotlib, com os seguintes objetivos:

- Analisar a estrutura e as dimensões dos dados;
- Verificar valores em falta e duplicados;
- Validar as relações entre as tabelas;
- Verificar a consistência dos pesos;
- Analisar as janelas temporais;
- Comparar os pesos das encomendas com as capacidades dos veículos;
- Representar geograficamente as localizações;
- Validar a matriz de distâncias e tempos de viagem.

Os resultados e respetivas conclusões encontram-se no notebook `01_eda_tiny.ipynb`.

## 4. Estrutura do Repositório

```text
IA2026_TP_G05/
│
├── Proj01.ipynb
├── EDA.ipynb
├── README.md
│
└── data/
    ├── collection_centers.csv
    ├── customers.csv
    ├── delivery_locations.csv
    ├── distance_matrix.csv
    ├── localities.csv
    ├── locations.csv
    ├── order_items.csv
    ├── orders.csv
    ├── producers.csv
    ├── products.csv
    ├── rules.csv
    └── vehicles.csv
```

**Descrição dos ficheiros principais:**

- `Proj01.ipynb` — apresentação do projeto, definição do problema, resumo da EDA e abordagem proposta.
- `EDA.ipynb` — análise exploratória do dataset `small`.
- `data/` — dados disponibilizados pelo docente.
- `README.md` — documentação e instruções de execução do projeto.

A estrutura será expandida nas próximas fases para incluir os módulos de implementação dos algoritmos.

## 5. Tecnologias Utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python | Desenvolvimento e modelação do problema |
| Jupyter Notebook | Execução e documentação |
| Pandas | Manipulação e análise dos dados |
| NumPy | Operações numéricas |
| Matplotlib | Visualização gráfica |
| Git e GitHub | Controlo de versões e colaboração |

## 6. Como Executar o Projeto

### Requisitos

É necessário ter o Python 3 instalado.

### Instalação das dependências

No terminal, dentro da pasta do projeto, executar:

```bash
python3 -m pip install pandas numpy matplotlib notebook
```

### Executar os notebooks

Iniciar o Jupyter Notebook:

```bash
jupyter notebook
```

Após iniciar o Jupyter, abrir um dos seguintes ficheiros:

- `main.ipynb` — apresentação geral e documentação do projeto.
- `01_eda_tiny.ipynb` — análise exploratória dos dados.

Para executar a EDA, selecionar **Kernel → Restart & Run All**.

**Nota:** os notebooks devem ser executados a partir da pasta principal do projeto, mantendo os CSV em `data/small/`.

## 7. Metodologia e Abordagem Proposta

O desenvolvimento será realizado em diferentes fases.

### Fase 1 — Análise Exploratória dos Dados

Exploração do dataset, validação da integridade dos dados e identificação das principais restrições.

### Fase 2 — Modelação do Problema

Definição das estruturas `Problem`, `Request`, `Vehicle` e `State`, juntamente com mecanismos de validação de rotas e cálculo dos respetivos custos.

### Fase 3 — Pesquisa Informada

Implementação dos algoritmos Greedy Best-First Search e A*, utilizando heurísticas para orientar a pesquisa.

### Fase 4 — Pesquisa Local

Implementação de Hill-Climbing para melhorar uma solução inicialmente obtida pelo Greedy.

### Fase 5 — Satisfação de Restrições

Formulação do problema como CSP, através da definição de variáveis, domínios e restrições.

### Fase 6 — Comparação de Resultados

Avaliação das diferentes abordagens relativamente à viabilidade das soluções, aos custos das rotas e ao desempenho computacional.

## 8. Estado Atual do Projeto

**Entrega Preliminar**

Nesta fase foram realizadas as seguintes tarefas:

- [x] Estudo e compreensão do enunciado.
- [x] Organização inicial do repositório.
- [x] Análise exploratória do dataset `small`.
- [x] Verificação da qualidade e integridade dos dados.
- [x] Identificação das principais restrições do problema.
- [x] Definição da abordagem proposta.
- [x] Documentação inicial do projeto.

**Próximas etapas:** modelação do problema, implementação dos algoritmos, realização de testes e comparação dos resultados.

---

*Instituto Politécnico do Cávado e do Ave — Inteligência Artificial — 2026/2027*