"""
src/analysis/dag.py

Define o DAG causal do estudo usando a biblioteca DoWhy / networkx.
Permite:
  - Visualizar o grafo
  - Listar conjuntos de ajuste (backdoor, frontdoor)
  - Testar implicações do DAG (testes de independência condicional)

Referência: https://py-why.github.io/dowhy/
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import networkx as nx

try:
    import dowhy
    from dowhy import CausalModel
    HAS_DOWHY = True
except ImportError:
    HAS_DOWHY = False
    warnings.warn("DoWhy não instalado. Instale com: pip install dowhy")


# ── Definição do DAG em notação GML (string) ──────────────────────────────────
# Variáveis do modelo:
#   T  = representacao_relativa  (tratamento: cadeiras per capita relativo)
#   M  = emendas_per_cap         (mediador)
#   Y  = gini                    (outcome: desigualdade)
#   X1 = log_pib_per_cap         (confundidor: riqueza)
#   X2 = regiao                  (confundidor: região geográfica)
#   X3 = gini_lag1               (confundidor: desigualdade inicial / baseline)
#   X4 = alinhamento_gov         (confundidor: partido governador alinhado ao federal)
#   Z  = distorcao_cadeiras      (instrumento: afasta da proporcionalidade constitucional)
#   P  = populacao               (causa comum de T e Y, não confundidor direto)

DAG_STRING = """
digraph {
    /* Instrumento */
    distorcao_cadeiras -> representacao_relativa;

    /* Tratamento → Mediador → Outcome */
    representacao_relativa -> emendas_per_cap;
    emendas_per_cap -> gini;

    /* Efeito direto T → Y (se houver) */
    representacao_relativa -> gini;

    /* Confundidores */
    log_pib_per_cap -> representacao_relativa;
    log_pib_per_cap -> emendas_per_cap;
    log_pib_per_cap -> gini;

    regiao -> representacao_relativa;
    regiao -> gini;

    gini_lag1 -> gini;

    alinhamento_gov -> emendas_per_cap;
    alinhamento_gov -> gini;

    /* Populacao afeta T (via distribuição de cadeiras) e Y diretamente */
    populacao -> representacao_relativa;
    populacao -> gini;
}
"""


def build_nx_dag() -> nx.DiGraph:
    """Retorna o DAG como DiGraph do networkx."""
    G = nx.DiGraph()

    edges = [
        ("distorcao_cadeiras", "representacao_relativa"),
        ("representacao_relativa", "emendas_per_cap"),
        ("emendas_per_cap", "gini"),
        ("representacao_relativa", "gini"),
        ("log_pib_per_cap", "representacao_relativa"),
        ("log_pib_per_cap", "emendas_per_cap"),
        ("log_pib_per_cap", "gini"),
        ("regiao", "representacao_relativa"),
        ("regiao", "gini"),
        ("gini_lag1", "gini"),
        ("alinhamento_gov", "emendas_per_cap"),
        ("alinhamento_gov", "gini"),
        ("populacao", "representacao_relativa"),
        ("populacao", "gini"),
    ]
    G.add_edges_from(edges)

    node_roles = {
        "distorcao_cadeiras": "instrument",
        "representacao_relativa": "treatment",
        "emendas_per_cap": "mediator",
        "gini": "outcome",
        "log_pib_per_cap": "confounder",
        "regiao": "confounder",
        "gini_lag1": "confounder",
        "alinhamento_gov": "confounder",
        "populacao": "confounder",
    }
    nx.set_node_attributes(G, node_roles, "role")
    return G


def plot_dag(ax=None, figsize=(14, 9)) -> plt.Figure:
    """Plota o DAG com cores por papel de cada variável."""
    G = build_nx_dag()

    COLOR_MAP = {
        "instrument":  "#4CAF50",  # verde
        "treatment":   "#2196F3",  # azul
        "mediator":    "#FF9800",  # laranja
        "outcome":     "#F44336",  # vermelho
        "confounder":  "#9E9E9E",  # cinza
    }

    roles = nx.get_node_attributes(G, "role")
    node_colors = [COLOR_MAP.get(roles.get(n, "confounder"), "#9E9E9E") for n in G.nodes()]

    pos = {
        "distorcao_cadeiras":      (-2.5, 0),
        "representacao_relativa":  (-1,   0),
        "emendas_per_cap":         (0.5,  0),
        "gini":                    (2,    0),
        "log_pib_per_cap":         (-1,   1.5),
        "populacao":               (-2.5, -1.5),
        "regiao":                  (0.5,  1.5),
        "gini_lag1":               (2,    1.5),
        "alinhamento_gov":         (0.5, -1.5),
    }

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors,
        node_size=2500,
        font_size=9,
        font_color="white",
        font_weight="bold",
        edge_color="#555555",
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.1",
    )

    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color="#4CAF50", label="Instrumento (Z)"),
        Patch(color="#2196F3", label="Tratamento (T)"),
        Patch(color="#FF9800", label="Mediador (M)"),
        Patch(color="#F44336", label="Outcome (Y)"),
        Patch(color="#9E9E9E", label="Confundidor (X)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title("DAG Causal: Representação → Emendas → Desigualdade", fontsize=13)
    ax.axis("off")
    return fig


def get_backdoor_adjustment_set() -> list[str]:
    """
    Retorna o conjunto de ajuste backdoor para o efeito T → Y.
    (a ser chamado via DoWhy para confirmação formal)
    """
    # Por inspecção do DAG: caminhos backdoor de T para Y passam por:
    # T ← log_pib_per_cap → Y
    # T ← regiao → Y
    # T ← populacao → Y
    # → Conjunto de ajuste mínimo: {log_pib_per_cap, regiao, populacao, gini_lag1}
    return ["log_pib_per_cap", "regiao", "populacao", "gini_lag1"]


def get_frontdoor_path() -> list[tuple[str, str]]:
    """
    Retorna o caminho frontdoor: T → M → Y.
    """
    return [
        ("representacao_relativa", "emendas_per_cap"),
        ("emendas_per_cap", "gini"),
    ]


def build_dowhy_model(df) -> "CausalModel":
    """
    Instancia modelo DoWhy com o painel de dados.
    Requer: pip install dowhy
    """
    if not HAS_DOWHY:
        raise ImportError("DoWhy não instalado. pip install dowhy")

    model = CausalModel(
        data=df,
        treatment="representacao_relativa",
        outcome="gini",
        graph=DAG_STRING,
    )
    return model


if __name__ == "__main__":
    fig = plot_dag()
    fig.tight_layout()
    plt.savefig("dag_causal.png", dpi=150, bbox_inches="tight")
    print("DAG salvo em dag_causal.png")
    print("\nConjunto de ajuste backdoor (T→Y):", get_backdoor_adjustment_set())
    print("Caminho frontdoor (T→M→Y):", get_frontdoor_path())
