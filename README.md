# Estudo Causal: Representação Parlamentar, Emendas e Desigualdade nos Estados Brasileiros

## Pergunta Causal

> **Mais cadeiras parlamentares → mais emendas orçamentárias → redução da desigualdade estadual?**

Testa-se um modelo de mediação causal onde a **representação política** afeta a **desigualdade regional** *por meio* da alocação de recursos federais via emendas parlamentares.

---

## Contexto

### Sistema eleitoral brasileiro (Câmara dos Deputados)

| Item | Regra |
|---|---|
| Distribuição de cadeiras | Proporcional à população |
| Mínimo por estado | 8 deputados |
| Máximo por estado | 70 deputados |
| Total de cadeiras | 513 deputados |

A distorção é estrutural: São Paulo tem ~22% da população mas apenas ~13,6% das cadeiras. Já o Amapá tem ~0,35% da população e ~1,56% das cadeiras. O voto de um paulista vale, em média, **~4,5x menos** do que o de um amapaense na Câmara.

### Emendas impositivas — linha do tempo

| Ano | Marco |
|---|---|
| 2015 | EC 86: emendas individuais impositivas (saúde) |
| 2019 | EC 100: expansão das emendas impositivas |
| 2019 | EC 105: emendas de bancada estadual impositivas |
| 2021–22 | "Orçamento secreto" — emendas de relator (declaradas inconstitucionais pelo STF) |
| 2023+ | "Emendas Pix" — transferências diretas a municípios |
| 2024 | ~50% do orçamento discricionário federal sob controle parlamentar |

O crescimento das emendas impositivas transferiu poder de alocação do Executivo para o Legislativo. Como o Legislativo tem uma lógica de representação distorcida (caps min/max), os estados sobre-representados captam mais emendas per capita.

---

## Framework Causal

### DAG (Grafo Acíclico Dirigido)

```
População (P) ──────────────────────────────────────────► Gini (Y)
     │                                                        ▲
     ▼                                                        │
Cadeiras Constitucionais                                      │
 (com cap min=8, max=70)                                      │
     │                                                        │
     ▼                                                        │
Representação per capita (T) ──► Emendas per capita (M) ──────┘
     │                               ▲
     │                               │
     └── Partido dominante ──────────┘
              │
              ▼
         Coalizão com
         governo federal
```

**Formalmente:**

- `T` = Cadeiras / População do estado (representação per capita; tratamento)
- `M` = Emendas federais recebidas / População do estado (mediador)
- `Y` = Variação do coeficiente de Gini estadual (outcome)
- `X` = Confundidores: PIB per capita, % população urbana, partido do governador, alinhamento com governo federal, região geográfica, Gini inicial

### Hipóteses causais

**H1 (T → M):** Estados com maior representação per capita captam mais emendas per capita.  
**H2 (M → Y):** Maior volume de emendas recebidas reduz a desigualdade (Gini) do estado.  
**H3 (T → M → Y):** O efeito causal de representação sobre desigualdade é *mediado* pelas emendas.

### Efeitos de interesse

| Efeito | Notação | Descrição |
|---|---|---|
| Total (ATE) | `T → Y` (total) | Efeito total de mais representação sobre Gini |
| Direto (NDE) | `T → Y` (direto) | Efeito que não passa pelas emendas |
| Indireto (NIE) | `T → M → Y` | Efeito mediado pelas emendas |
| % mediado | NIE / ATE | Quanto do efeito total passa pelas emendas |

---

## Estratégia de Identificação

### Problema central
A alocação de cadeiras **não é aleatória** — é função da população (com truncamento). Precisamos de variação exógena.

### Abordagem principal: Quasi-experimento por truncamento constitucional

O cap constitucional (mínimo 8, máximo 70) cria **descontinuidades** na curva de representação:
- Estados **no limite superior** (SP: 70 cadeiras) são *sub-representados* exogenamente
- Estados **no limite inferior** (8 cadeiras: AC, AP, RO, RR, TO, AM, MS, MT, AL, SE, PI, PB...) são *sobre-representados* exogenamente

Isso permite uma **Variável Instrumental (IV)**:
> Z = "distância ao limite constitucional (over/under-representation)"

A lógica: o truncamento é determinístico e constitucional — não foi escolhido pelo estado para maximizar emendas, portanto satisfaz a condição de *exclusão* do IV.

### Abordagem complementar: Diferenças-em-diferenças (DiD)

A EC 86/2015 e EC 100/2019 são choques exógenos que *amplificaram* o canal emendas.
- Comparar evolução do Gini antes/depois das reformas
- Interagir com nível de representação per capita do estado

### Análise de mediação causal

Utilizando a decomposição de Imai et al. (2010) via `mediation` (R) ou `causalml`/`dowhy` (Python).

---

## Dados

| Fonte | Variável | URL |
|---|---|---|
| TSE | Cadeiras por estado por eleição | https://dadosabertos.tse.jus.br |
| IBGE/PNAD | Gini estadual (PNAD Contínua) | https://sidra.ibge.gov.br |
| IBGE | População estadual (Censo + estimativas) | https://ibge.gov.br |
| SIGA Brasil (Senado) | Emendas parlamentares executadas por estado | https://www12.senado.leg.br/orcamento/sigabrasil |
| Portal da Transparência | Transferências federais por município/estado | https://portaldatransparencia.gov.br |
| IPEA | PIB per capita estadual, IDH municipal | https://www.ipeadata.gov.br |
| Câmara — Dados Abertos | Emendas por deputado e UF | https://dadosabertos.camara.leg.br |

---

## Estrutura do Projeto

```
causal_study/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/              ← dados originais, nunca modificar
│   └── processed/        ← dados limpos para análise
├── src/
│   ├── __init__.py
│   ├── collect/
│   │   ├── tse.py        ← cadeiras por estado/ano
│   │   ├── ibge.py       ← Gini, população, PIB
│   │   └── siga.py       ← emendas por estado/ano
│   ├── build_panel.py    ← junta tudo num painel estado × ano
│   └── analysis/
│       ├── dag.py        ← define e valida o DAG
│       ├── iv.py         ← estimação IV (truncamento constitucional)
│       ├── did.py        ← diff-in-diff pré/pós EC86/EC100
│       └── mediation.py  ← análise de mediação causal
└── notebooks/
    ├── 01_eda.ipynb           ← exploratória: cadeiras × emendas × Gini
    ├── 02_dag_validation.ipynb ← testes de hipóteses do DAG
    ├── 03_iv_estimation.ipynb  ← estimação via IV
    └── 04_mediation.ipynb      ← decomposição efeito direto/indireto
```

---

## Referências metodológicas

- Imai, K., Keele, L., & Tingley, D. (2010). A general approach to causal mediation analysis. *Psychological Methods*.
- Pearl, J. (2001). Direct and indirect effects. *UAI*.
- Imbens, G. & Angrist, J. (1994). Identification and estimation of local average treatment effects. *Econometrica*.
- Cattaneo, M., Idrobo, N., & Titiunik, R. (2020). *A Practical Introduction to Regression Discontinuity Designs*.
- DoWhy documentation: https://py-why.github.io/dowhy/
