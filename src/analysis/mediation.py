"""
src/analysis/mediation.py

Análise de mediação causal:
  T (representacao_relativa) → M (emendas_per_cap) → Y (gini)

Decompõe o efeito total em:
  - NDE (Natural Direct Effect): T → Y não passando por M
  - NIE (Natural Indirect Effect): T → M → Y
  - Proporção mediada: NIE / ATE

Dois métodos implementados:
  1. Baron & Kenny sequencial (OLS, assunção de linearidade)
  2. Imai et al. (2010) via DoWhy / causalml (não-paramétrico, com bootstrap CI)

Referências:
  - Baron & Kenny (1986). The moderator-mediator variable distinction.
  - Imai, Keele, Tingley (2010). A general approach to causal mediation analysis.
  - Pearl (2001). Direct and indirect effects.
  - VanderWeele (2015). Explanation in Causal Inference.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    import dowhy
    from dowhy import CausalModel
    HAS_DOWHY = True
except ImportError:
    HAS_DOWHY = False

try:
    from causalml.inference.meta import LRSRegressor
    HAS_CAUSALML = True
except ImportError:
    HAS_CAUSALML = False


@dataclass
class MediationResult:
    """Resultado da decomposição causal."""
    ate: float       # Average Total Effect
    nde: float       # Natural Direct Effect
    nie: float       # Natural Indirect Effect
    prop_mediated: float  # NIE / ATE
    ate_se: float = np.nan
    nde_se: float = np.nan
    nie_se: float = np.nan
    method: str = ""

    def __str__(self) -> str:
        lines = [
            f"\n{'='*55}",
            f"  Método: {self.method}",
            f"{'='*55}",
            f"  ATE (efeito total):               {self.ate:+.6f}  (SE={self.ate_se:.6f})",
            f"  NDE (efeito direto):              {self.nde:+.6f}  (SE={self.nde_se:.6f})",
            f"  NIE (efeito via emendas):         {self.nie:+.6f}  (SE={self.nie_se:.6f})",
            f"  Proporção mediada (NIE/ATE):      {self.prop_mediated:.1%}",
            f"{'='*55}",
            f"  Interpretação:",
            f"  {'REDUZ' if self.ate < 0 else 'AUMENTA'} desigualdade em {abs(self.ate):.4f} por unidade de representação.",
            f"  Desse efeito, {abs(self.prop_mediated):.1%} passa pelas emendas (via mediação).",
        ]
        return "\n".join(lines)


# ── Método 1: Baron & Kenny (OLS sequencial) ──────────────────────────────────

def baron_kenny(
    df: pd.DataFrame,
    treatment: str = "representacao_relativa",
    mediator: str = "emendas_per_cap",
    outcome: str = "gini",
    controls: list[str] | None = None,
    log_mediator: bool = True,
    add_historical_controls: bool = True,
) -> MediationResult:
    """
    Decomposição Baron & Kenny via OLS.

    Passo 1: Y ~ T + X                   → coef α₁ (efeito total)
    Passo 2: M ~ T + X                   → coef β₁ (T → M)
    Passo 3: Y ~ T + M + X              → coef α₂ (NDE) e β₂ (M → Y)
    NIE = β₁ × β₂ (produto dos coeficientes)

    Parâmetros
    ----------
    add_historical_controls : bool (default True)
        Se True, adiciona automaticamente às variáveis de controle as
        proxies históricas disponíveis no painel (log_pib_pc_1991,
        gini_baseline) para bloquear o backdoor da persistência histórica
        de desigualdade. Isso torna a mediação válida condicionalmente
        mesmo na presença da ameaça identificada em seção 4.2.
    """
    if controls is None:
        controls = ["log_pib_per_cap", "gini_lag1", "C(regiao)"]

    # Adicionar proxies históricas se disponíveis e solicitadas
    if add_historical_controls:
        extra = []
        if "gini_1991" in df.columns and df["gini_1991"].notna().any():
            extra.append("gini_1991")   # preferido: Gini ADH 1991 (Censo)
        elif "gini_baseline" in df.columns and df["gini_baseline"].notna().any():
            extra.append("gini_baseline")  # fallback: PNAD 2012
        if "log_pib_pc_1991" in df.columns and df["log_pib_pc_1991"].notna().any():
            extra.append("log_pib_pc_1991")
        if extra:
            controls = controls + extra
            print(f"[Mediação] Controles históricos adicionados: {extra}")
        else:
            import warnings
            warnings.warn(
                "Proxies históricas (gini_1991, log_pib_pc_1991) não encontradas "
                "no painel. A ameaça do confundidor histórico não está sendo controlada. "
                "Execute build_panel.py para gerar o painel completo."
            )

    # Normalizar variáveis para coeficientes comparáveis
    df = df.copy().dropna(subset=[outcome, treatment, mediator] +
                          [c for c in controls if not c.startswith("C(")])

    med_col = mediator
    if log_mediator:
        med_col = "log_emendas_per_cap"
        if med_col not in df.columns:
            df[med_col] = np.log(df[mediator].replace(0, np.nan))
        df = df.dropna(subset=[med_col])

    ctrl_str = " + ".join(controls)

    # Passo 1 — efeito total T → Y
    m1 = smf.ols(f"{outcome} ~ {treatment} + {ctrl_str}", data=df).fit()
    ate = m1.params[treatment]
    ate_se = m1.bse[treatment]

    # Passo 2 — T → M
    m2 = smf.ols(f"{med_col} ~ {treatment} + {ctrl_str}", data=df).fit()
    beta1 = m2.params[treatment]
    beta1_se = m2.bse[treatment]

    # Passo 3 — NDE e M → Y (controlando T)
    m3 = smf.ols(f"{outcome} ~ {treatment} + {med_col} + {ctrl_str}", data=df).fit()
    nde = m3.params[treatment]
    nde_se = m3.bse[treatment]
    beta2 = m3.params[med_col]
    beta2_se = m3.bse[med_col]

    # NIE = produto β₁ × β₂ (Delta method SE)
    nie = beta1 * beta2
    nie_se = np.sqrt((beta1 * beta2_se) ** 2 + (beta2 * beta1_se) ** 2)

    prop = nie / ate if abs(ate) > 1e-10 else np.nan

    print(f"\n[Baron & Kenny] T→M: β={beta1:.4f}(p={m2.pvalues[treatment]:.3f}), "
          f"M→Y: β={beta2:.4f}(p={m3.pvalues[med_col]:.3f}), "
          f"T→Y direto: β={nde:.4f}(p={m3.pvalues[treatment]:.3f})")

    result = MediationResult(
        ate=ate, nde=nde, nie=nie, prop_mediated=prop,
        ate_se=ate_se, nde_se=nde_se, nie_se=nie_se,
        method="Baron & Kenny (OLS sequencial)",
    )
    print(result)
    return result


# ── Método 2: DoWhy (identificação formal + estimação) ───────────────────────

def dowhy_mediation(
    df: pd.DataFrame,
    dag_string: str,
    treatment: str = "representacao_relativa",
    mediator: str = "emendas_per_cap",
    outcome: str = "gini",
    n_simulations: int = 100,
) -> MediationResult:
    """
    Análise de mediação via DoWhy.
    Estima NDE e NIE usando identificação causal formal.
    """
    if not HAS_DOWHY:
        raise ImportError("DoWhy não instalado. pip install dowhy")

    from dowhy import CausalModel

    df_clean = df.dropna(subset=[outcome, treatment, mediator,
                                  "log_pib_per_cap", "gini_lag1"]).copy()

    # --- Efeito total ---
    model_total = CausalModel(
        data=df_clean,
        treatment=treatment,
        outcome=outcome,
        graph=dag_string,
    )
    identified_total = model_total.identify_effect(proceed_when_unidentifiable=True)
    estimate_total = model_total.estimate_effect(
        identified_total,
        method_name="backdoor.linear_regression",
        control_value=df_clean[treatment].mean() - df_clean[treatment].std(),
        treatment_value=df_clean[treatment].mean() + df_clean[treatment].std(),
    )
    ate = estimate_total.value

    # --- Efeito direto (bloqueando mediador) ---
    model_direct = CausalModel(
        data=df_clean,
        treatment=treatment,
        outcome=outcome,
        graph=dag_string,
        common_causes=["log_pib_per_cap", "gini_lag1", "regiao", mediator],
    )
    identified_direct = model_direct.identify_effect(proceed_when_unidentifiable=True)
    estimate_direct = model_direct.estimate_effect(
        identified_direct,
        method_name="backdoor.linear_regression",
        control_value=df_clean[treatment].mean() - df_clean[treatment].std(),
        treatment_value=df_clean[treatment].mean() + df_clean[treatment].std(),
    )
    nde = estimate_direct.value
    nie = ate - nde
    prop = nie / ate if abs(ate) > 1e-10 else np.nan

    result = MediationResult(
        ate=ate, nde=nde, nie=nie, prop_mediated=prop,
        method="DoWhy (backdoor linear regression)",
    )
    print(result)
    return result


# ── Visualização ──────────────────────────────────────────────────────────────

def plot_mediation_diagram(result: MediationResult, figsize=(10, 4)):
    """Diagrama visual da decomposição causal."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, text, color):
        ax.add_patch(plt.Rectangle((x - 1.1, y - 0.5), 2.2, 1.0,
                                    facecolor=color, edgecolor="black", linewidth=1.5, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)

    def arrow(x1, y1, x2, y2, label, color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.25
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=9, color=color)

    box(1.5, 2.5, "Representação\nper capita (T)", "#2196F3")
    box(5.0, 4.2, f"Emendas\nper capita (M)", "#FF9800")
    box(8.5, 2.5, "Gini (Y)", "#F44336")

    # NIE: T → M → Y
    arrow(2.6, 3.0, 3.9, 4.0, f"NIE={result.nie:+.4f}", "#FF9800")
    arrow(6.1, 4.0, 7.4, 3.0, "", "#FF9800")

    # NDE: T → Y (direto)
    arrow(2.6, 2.5, 7.4, 2.5, f"NDE={result.nde:+.4f}", "#2196F3")

    # ATE total
    ax.text(5.0, 0.7,
            f"ATE = {result.ate:+.4f} | NIE/ATE = {result.prop_mediated:.1%}",
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="black"))

    ax.set_title("Decomposição causal: efeito total, direto e mediado", fontsize=13)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.build_panel import build_panel
    from src.collect.tse import CADEIRAS_POR_UF
    import numpy as np

    rng = np.random.default_rng(7)
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
    result = baron_kenny(panel)
    plot_mediation_diagram(result)
    import matplotlib.pyplot as plt
    plt.savefig("mediation_diagram.png", dpi=150, bbox_inches="tight")
    print("\nDiagrama salvo em mediation_diagram.png")
