"""
src/analysis/iv.py

Estratégia de Variável Instrumental (IV) / 2SLS para identificar
o efeito causal da representação parlamentar sobre emendas e Gini.

Instrumento: distorcao_cadeiras
  = cadeiras_camara - cadeiras_justas (proporcional pura)

Intuição: a distância em relação à proporcionalidade pura é
  determinada pela Constituição (caps min/max), logo é exógena —
  não foi escolhida pelo estado para maximizar captação de emendas
  ou minimizar desigualdade.

Condições do instrumento:
  (i)  Relevância:   distorcao_cadeiras → representacao_relativa  (F > 10)
  (ii) Exclusão:     distorcao_cadeiras afeta Gini APENAS via representação
  (iii) Exogeneidade: não correlacionado com confundidores não observados

Referência: Angrist & Pischke (2009) — Mostly Harmless Econometrics
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS

try:
    from linearmodels.iv import IV2SLS as PanelIV2SLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False
    warnings.warn("linearmodels não instalado. Instale com: pip install linearmodels")


def run_first_stage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Primeiro estágio do 2SLS: regressão de T (representacao_relativa) em Z.

    Verifica condição de relevância: F-statistic deve ser > 10.
    """
    controls = "log_pib_per_cap + C(regiao) + gini_lag1"
    formula = f"representacao_relativa ~ distorcao_cadeiras + {controls}"

    model = smf.ols(formula, data=df.dropna(subset=["representacao_relativa",
                                                       "distorcao_cadeiras",
                                                       "log_pib_per_cap",
                                                       "gini_lag1"])).fit()

    print("=" * 60)
    print("PRIMEIRO ESTÁGIO: representacao_relativa ~ distorcao_cadeiras")
    print("=" * 60)
    print(model.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]])
    print(f"\nF-statistic (instrumento): {model.fvalue:.2f}")
    print(f"R² ajustado:               {model.rsquared_adj:.4f}")
    if model.fvalue < 10:
        warnings.warn("F-stat < 10: instrumento fraco! Interpretar resultados com cautela.")

    return model


def run_reduced_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forma reduzida: regressão do outcome Y em Z diretamente.
    Estimativa indireta do efeito causal: β_rf = β_iv × β_fs
    """
    controls = "log_pib_per_cap + C(regiao) + gini_lag1"
    formula = f"gini ~ distorcao_cadeiras + {controls}"

    model = smf.ols(formula, data=df.dropna(subset=["gini", "distorcao_cadeiras",
                                                      "log_pib_per_cap", "gini_lag1"])).fit()
    print("\n" + "=" * 60)
    print("FORMA REDUZIDA: gini ~ distorcao_cadeiras")
    print("=" * 60)
    print(model.summary2().tables[1][["Coef.", "Std.Err.", "t", "P>|t|"]])
    return model


def run_conditional_iv(df: pd.DataFrame) -> dict:
    """
    IV Condicional: testa robustez da exclusion restriction incluindo proxies
    da situação pré-constitucional (1988) como controles adicionais.

    Motivação
    ---------
    Ameaça identificada: o número de cadeiras pode ter sido determinado (em parte)
    pelo nível de desigualdade / desenvolvimento dos estados em 1988.
    Se Desig_1988 → Cadeiras  E  Desig_1988 → Gini_atual, a exclusion restriction
    do IV fica comprometida (há caminho direto instrumento → outcome).

    Estratégia
    ----------
    Controlando explicitamente para Desig_1988 (ou proxy) bloqueamos esse backdoor.
    O instrumento passa a ser válido condicionalmente.

    Proxies utilizadas (em ordem de preferência):
      1. log_pib_pc_1991 : log PIB per capita estadual 1991 (IPEADATA/PIBPCE)
         → correlacionado com desigualdade histórica; exógeno à política atual
      2. gini_baseline   : primeiro Gini disponível no painel (≈2012)
         → proxy da persistência inercial da desigualdade

    Retorna
    -------
    dict com chaves:
      'baseline'    : resultado OLS sem proxy histórica (referência)
      'cond_pib91'  : IV controlando por log_pib_pc_1991
      'cond_gini0'  : IV controlando por gini_baseline
      'cond_both'   : IV controlando por ambas
      'sensitivity' : DataFrame comparativo de coeficientes e p-values
    """
    results = {}
    base_controls = "log_pib_per_cap + C(regiao) + gini_lag1"

    # ── Verificar disponibilidade das proxies ─────────────────────────────────
    has_pib91  = "log_pib_pc_1991" in df.columns and df["log_pib_pc_1991"].notna().any()
    has_gini0  = "gini_baseline"   in df.columns and df["gini_baseline"].notna().any()

    if not has_pib91 and not has_gini0:
        warnings.warn(
            "Proxies históricas ausentes do painel. "
            "Execute build_panel.py com get_historical_proxy() habilitado.\n"
            "O IV condicional não pode ser estimado sem elas."
        )
        return {}

    def _ols_iv(df_clean: pd.DataFrame, extra: str, tag: str):
        """Helper: OLS da forma reduzida + primeiro estágio com controles extras."""
        controls = f"{base_controls} + {extra}" if extra else base_controls
        fs = smf.ols(
            f"representacao_relativa ~ distorcao_cadeiras + {controls}",
            data=df_clean,
        ).fit()
        rf = smf.ols(
            f"gini ~ distorcao_cadeiras + {controls}",
            data=df_clean,
        ).fit()
        iv_est = rf.params["distorcao_cadeiras"] / fs.params["distorcao_cadeiras"]
        return {
            "tag": tag,
            "fs_coef": fs.params["distorcao_cadeiras"],
            "fs_fstat": fs.fvalue,
            "rf_coef": rf.params["distorcao_cadeiras"],
            "rf_pval": rf.pvalues["distorcao_cadeiras"],
            "iv_wald": iv_est,
            "n": int(fs.nobs),
            "extra_controls": extra or "(none)",
        }

    # ── Especificação base (sem proxy histórica) ──────────────────────────────
    req_cols = ["gini", "representacao_relativa", "distorcao_cadeiras",
                "log_pib_per_cap", "gini_lag1"]
    df0 = df.dropna(subset=req_cols).copy()
    results["baseline"] = _ols_iv(df0, "", "baseline")

    # ── Controlando por PIB per capita 1991 ───────────────────────────────────
    if has_pib91:
        df1 = df.dropna(subset=req_cols + ["log_pib_pc_1991"]).copy()
        results["cond_pib91"] = _ols_iv(df1, "log_pib_pc_1991", "cond_pib91")

    # ── Controlando por Gini baseline (persistência histórica) ───────────────
    if has_gini0:
        df2 = df.dropna(subset=req_cols + ["gini_baseline"]).copy()
        results["cond_gini0"] = _ols_iv(df2, "gini_baseline", "cond_gini0")

    # ── Controlando por ambas ─────────────────────────────────────────────────
    if has_pib91 and has_gini0:
        df3 = df.dropna(subset=req_cols + ["log_pib_pc_1991", "gini_baseline"]).copy()
        results["cond_both"] = _ols_iv(df3, "log_pib_pc_1991 + gini_baseline", "cond_both")

    # ── Tabela comparativa ────────────────────────────────────────────────────
    sens = pd.DataFrame(list(results.values()))
    results["sensitivity"] = sens

    print("\n" + "=" * 70)
    print("IV CONDICIONAL — Sensibilidade à ameaça de confundidor histórico")
    print("=" * 70)
    print(sens[["tag", "fs_fstat", "rf_coef", "rf_pval", "iv_wald",
                "n", "extra_controls"]].to_string(index=False))
    print("\nInterpretação:")
    print("  Se rf_coef e iv_wald NÃO mudarem muito ao adicionar proxies históricas,")
    print("  a exclusion restriction é robusta ao confundidor potencial.")
    print("  Se mudarem significativamente → o instrumento estava contaminado.")

    return results


def run_2sls_total_effect(df: pd.DataFrame):
    """
    2SLS: efeito total de representacao_relativa sobre gini.
    T: representacao_relativa
    Z: distorcao_cadeiras
    Y: gini
    X: log_pib_per_cap, regiao, gini_lag1

    Se HAS_LINEARMODELS, usa PanelOLS com efeitos fixos de UF e ano.
    Senão, usa 2SLS cross-sectional via statsmodels.
    """
    df_clean = df.dropna(subset=[
        "gini", "representacao_relativa", "distorcao_cadeiras",
        "log_pib_per_cap", "gini_lag1",
    ]).copy()

    if HAS_LINEARMODELS:
        return _run_panel_2sls(df_clean)
    else:
        return _run_simple_2sls(df_clean)


def _run_panel_2sls(df: pd.DataFrame):
    """IV2SLS com efeitos fixos de UF e ano (dados em painel)."""
    from linearmodels.iv import IV2SLS as LMiv2sls
    import pandas as pd

    df = df.set_index(["uf", "ano"])
    exog = df[["log_pib_per_cap", "gini_lag1"]]
    endog = df[["representacao_relativa"]]
    instruments = df[["distorcao_cadeiras"]]
    dep = df["gini"]

    # Absorb efeitos fixos de UF e ano via dummies
    uf_dummies = pd.get_dummies(df.index.get_level_values("uf"), drop_first=True, prefix="uf")
    ano_dummies = pd.get_dummies(df.index.get_level_values("ano"), drop_first=True, prefix="ano")
    exog_full = pd.concat([exog, uf_dummies, ano_dummies], axis=1).astype(float)

    model = LMiv2sls(dep, exog_full, endog, instruments).fit(cov_type="clustered",
                                                               clusters=df.index.get_level_values("uf"))
    print("\n" + "=" * 60)
    print("2SLS PAINEL (EF UF + ano, SE clusterizado por UF)")
    print("Efeito total: representacao_relativa → gini")
    print("=" * 60)
    print(model.summary.tables[1])
    return model


def _run_simple_2sls(df: pd.DataFrame):
    """2SLS cross-sectional simples via statsmodels (fallback)."""
    # Média por UF para cross-section
    df_cs = df.groupby("uf").mean(numeric_only=True).reset_index()

    y = df_cs["gini"].values
    X = np.column_stack([
        np.ones(len(df_cs)),
        df_cs["representacao_relativa"].values,
        df_cs["log_pib_per_cap"].values,
    ])
    Z = np.column_stack([
        np.ones(len(df_cs)),
        df_cs["distorcao_cadeiras"].values,
        df_cs["log_pib_per_cap"].values,
    ])

    model = IV2SLS(y, X, Z).fit()
    print("\n" + "=" * 60)
    print("2SLS Cross-Section (fallback sem linearmodels)")
    print("Efeito total: representacao_relativa → gini")
    print("=" * 60)
    print(pd.DataFrame({
        "coef": model.params,
        "se": model.bse,
        "t": model.tvalues,
        "p": model.pvalues,
    }, index=["const", "representacao_relativa", "log_pib_per_cap"]))
    return model


if __name__ == "__main__":
    # Teste com dados sintéticos
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from src.build_panel import build_panel, REGIAO_MAP
    from src.collect.tse import CADEIRAS_POR_UF
    import numpy as np

    rng = np.random.default_rng(0)
    ufs = list(CADEIRAS_POR_UF.keys())
    anos = list(range(2012, 2024))

    gini_m = pd.DataFrame({"uf": ufs * len(anos), "ano": sorted(anos * len(ufs)),
                            "gini": rng.uniform(0.35, 0.65, len(ufs) * len(anos))})
    em_m = pd.DataFrame({"uf": ufs * len(anos), "ano": sorted(anos * len(ufs)),
                          "valor_pago": rng.uniform(1e6, 5e9, len(ufs) * len(anos))})
    pib_m = pd.DataFrame({"uf": ufs * len(anos), "ano": sorted(anos * len(ufs)),
                           "pib_per_capita": rng.uniform(10_000, 90_000, len(ufs) * len(anos))})

    panel = build_panel(gini_m, em_m, pib_m)
    run_first_stage(panel)
    run_reduced_form(panel)
