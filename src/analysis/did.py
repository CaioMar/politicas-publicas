"""
src/analysis/did.py

Diferenças-em-Diferenças (DiD) para avaliar o impacto das reformas de
emendas impositivas sobre a desigualdade estadual.

A lógica DiD aqui é um staggered/interacted DiD:
  - Shock exógeno: EC 86/2015 e EC 100/2019 (emendas impositivas)
  - Heterogeneidade de tratamento: estados mais ou menos representados
    beneficiaram diferentemente do choque externo

Especificação principal (interacted DiD):

  Gini_it = α_i + λ_t + β₁·pos_ec86_t + β₂·pos_ec100_t
            + β₃·(RepresentacaoRelativa_i × pos_ec86_t)
            + β₄·(RepresentacaoRelativa_i × pos_ec100_t)
            + γ·X_it + ε_it

Onde:
  α_i = efeito fixo de UF (absorve todas as diferenças permanentes entre estados)
  λ_t = efeito fixo de ano (absorve tendências nacionais)
  β₃, β₄ = efeito heterogêneo: estados com mais representação per capita
             reagiram diferentemente ao choque dos emendas impositivas?

Referência:
  Callaway & Sant'Anna (2021) — Difference-in-Differences with Multiple Time Periods
  Cengiz et al. (2019) — stacked DiD
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    from linearmodels.panel import PanelOLS, PooledOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    warnings.warn("linearmodels não instalado. pip install linearmodels")


# ── Especificação DiD principal ───────────────────────────────────────────────

def run_interacted_did(df: pd.DataFrame, outcome: str = "gini") -> object:
    """
    DiD interado: como o choque das emendas impositivas afetou estados
    com diferentes níveis de representação parlamentar?

    Estima β₃ e β₄ — o efeito heterogêneo por representação.
    """
    df = df.copy().dropna(subset=[outcome, "representacao_relativa",
                                   "log_pib_per_cap", "gini_lag1"])

    # Interações centralizadas (representação em torno da média para interpretabilidade)
    mean_rep = df["representacao_relativa"].mean()
    df["rep_c"] = df["representacao_relativa"] - mean_rep
    df["rep_x_ec86"] = df["rep_c"] * df["pos_ec86"]
    df["rep_x_ec100"] = df["rep_c"] * df["pos_ec100"]

    if HAS_LINEARMODELS:
        return _did_panel(df, outcome)
    else:
        return _did_pooled(df, outcome)


def _did_panel(df: pd.DataFrame, outcome: str) -> object:
    """DiD com efeitos fixos de UF e ano via linearmodels."""
    from linearmodels.panel import PanelOLS

    df_panel = df.set_index(["uf", "ano"])
    dep = df_panel[outcome]
    exog = df_panel[["rep_x_ec86", "rep_x_ec100", "log_pib_per_cap", "gini_lag1"]]

    model = PanelOLS(dep, exog, entity_effects=True, time_effects=True,
                     drop_absorbed=True)
    result = model.fit(cov_type="clustered", cluster_entity=True)

    print("=" * 65)
    print(f"DiD INTERADO (Painel EF UF+ano) — outcome: {outcome}")
    print("=" * 65)
    print(result.summary.tables[1])
    print(f"\nNota: rep_x_ec86 > 0 significa que estados mais representados")
    print(f"tiveram maior redução de Gini após EC 86/2015.")
    return result


def _did_pooled(df: pd.DataFrame, outcome: str) -> object:
    """DiD com dummies de UF e ano via statsmodels (fallback)."""
    formula = (
        f"{outcome} ~ rep_x_ec86 + rep_x_ec100 + log_pib_per_cap + "
        f"gini_lag1 + C(uf) + C(ano)"
    )
    model = smf.ols(formula, data=df).fit(cov_type="cluster",
                                           cov_kwds={"groups": df["uf"]})
    print("=" * 65)
    print(f"DiD POOLED OLS (dummies UF+ano) — outcome: {outcome}")
    print("=" * 65)
    interested = model.summary2().tables[1]
    interested = interested[interested.index.str.contains("rep_x|pib")]
    print(interested[["Coef.", "Std.Err.", "t", "P>|t|"]])
    return model


# ── Parallel trends (pré-tendências) ─────────────────────────────────────────

def plot_parallel_trends(
    df: pd.DataFrame,
    outcome: str = "gini",
    reform_year: int = 2015,
    figsize=(12, 5),
) -> plt.Figure:
    """
    Plota evolução temporal do outcome separando estados no teto máximo (SP)
    e estados no mínimo constitucional vs. demais.

    Verifica visualmente a hipótese de tendências paralelas pré-reforma.
    """
    df = df.copy()
    df["grupo"] = "Meio (livre)"
    df.loc[df["status_cap"] == "maximo", "grupo"] = "Teto máximo (SP)"
    df.loc[df["status_cap"] == "minimo", "grupo"] = "Piso mínimo (8 cadeiras)"

    trend = df.groupby(["ano", "grupo"])[outcome].mean().reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    for grupo, sub in trend.groupby("grupo"):
        ax.plot(sub["ano"], sub[outcome], marker="o", label=grupo, linewidth=2)

    ax.axvline(reform_year, color="red", linestyle="--", linewidth=1.5,
               label=f"EC 86/{reform_year} (emendas impositivas)")
    ax.axvline(2019, color="orange", linestyle="--", linewidth=1.5,
               label="EC 100/2019")
    ax.set_xlabel("Ano")
    ax.set_ylabel(f"Média {outcome}")
    ax.set_title("Tendências paralelas por grupo de representação")
    ax.legend()
    fig.tight_layout()
    return fig


# ── Evento study (coeficientes dinâmicos) ─────────────────────────────────────

def run_event_study(
    df: pd.DataFrame,
    outcome: str = "gini",
    reform_year: int = 2015,
    base_year: int = 2014,
) -> pd.DataFrame:
    """
    Estima coeficientes de evento (event study) para verificar:
      1. Ausência de efeitos pré-reforma (validação)
      2. Efeitos dinâmicos pós-reforma

    Modela: Gini_it = α_i + λ_t + Σ_k β_k · (rep_c × I[ano=k]) + γ·X_it + ε_it
    """
    df = df.copy()
    mean_rep = df["representacao_relativa"].mean()
    df["rep_c"] = df["representacao_relativa"] - mean_rep

    anos_unique = sorted(df["ano"].unique())
    anos_excl = [base_year]  # ano base omitido

    coefs = []
    for ano in anos_unique:
        if ano in anos_excl:
            continue
        col = f"rep_x_y{ano}"
        df[col] = df["rep_c"] * (df["ano"] == ano).astype(int)

    event_cols = [c for c in df.columns if c.startswith("rep_x_y")]
    controls = "log_pib_per_cap + gini_lag1 + C(uf) + C(ano)"
    formula = f"{outcome} ~ {' + '.join(event_cols)} + {controls}"

    model = smf.ols(formula, data=df.dropna(subset=[outcome, "log_pib_per_cap", "gini_lag1"])).fit(
        cov_type="cluster", cov_kwds={"groups": df["uf"]}
    )

    for col in event_cols:
        ano_val = int(col.replace("rep_x_y", ""))
        coefs.append({
            "ano": ano_val,
            "coef": model.params.get(col, np.nan),
            "se": model.bse.get(col, np.nan),
            "pre_reforma": ano_val < reform_year,
        })

    coefs.append({"ano": base_year, "coef": 0.0, "se": 0.0, "pre_reforma": True})
    result = pd.DataFrame(coefs).sort_values("ano").reset_index(drop=True)
    result["ci_low"] = result["coef"] - 1.96 * result["se"]
    result["ci_high"] = result["coef"] + 1.96 * result["se"]
    return result


def plot_event_study(event_df: pd.DataFrame, reform_year: int = 2015) -> plt.Figure:
    """Plota gráfico de evento com intervalos de confiança."""
    fig, ax = plt.subplots(figsize=(12, 5))

    pre = event_df[event_df["pre_reforma"]]
    pos = event_df[~event_df["pre_reforma"]]

    for sub, color, label in [(pre, "#2196F3", "Pré-reforma"), (pos, "#F44336", "Pós-reforma")]:
        ax.errorbar(sub["ano"], sub["coef"],
                    yerr=1.96 * sub["se"], fmt="o",
                    color=color, capsize=5, label=label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(reform_year - 0.5, color="red", linestyle="--", linewidth=1.5,
               label=f"Reforma EC 86/{reform_year}")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Coeficiente de evento (interação rep × ano)")
    ax.set_title("Event Study: efeito heterogêneo da representação por ano")
    ax.legend()
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.build_panel import build_panel
    from src.collect.tse import CADEIRAS_POR_UF
    import numpy as np

    rng = np.random.default_rng(42)
    ufs = list(CADEIRAS_POR_UF.keys())
    anos = list(range(2012, 2024))
    n = len(ufs) * len(anos)

    gini_df = pd.DataFrame({"uf": ufs * len(anos), "ano": sorted(anos * len(ufs)),
                             "gini": rng.uniform(0.35, 0.65, n)})
    em_df = pd.DataFrame({"uf": ufs * len(anos), "ano": sorted(anos * len(ufs)),
                           "valor_pago": rng.uniform(1e6, 5e9, n)})
    pib_df = pd.DataFrame({"uf": ufs * len(anos), "ano": sorted(anos * len(ufs)),
                            "pib_per_capita": rng.uniform(10_000, 90_000, n)})

    panel = build_panel(gini_df, em_df, pib_df)
    run_interacted_did(panel)
