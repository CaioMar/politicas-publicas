"""
src/build_panel.py

Constrói o painel estado × ano que é a base de toda a análise.

Colunas do painel final:
  uf                       : sigla do estado
  ano                      : ano de referência
  cadeiras_camara          : deputados federais (fixo por legislatura)
  cadeiras_total           : camara + senado
  populacao                : estimativa IBGE
  cadeiras_camara_per_cap  : cadeiras / 100k hab
  representacao_relativa   : % cadeiras / % pop nacional
  status_cap               : 'maximo' | 'minimo' | 'livre'
  emendas_per_cap          : R$ emendas pagas / 100k hab (MEDIADOR)
  gini                     : coeficiente de Gini (OUTCOME)
  gini_lag1                : Gini ano anterior (controle de estado inicial)
  pib_per_capita           : PIB per capita estadual (confundidor)
  log_pib_per_cap          : log(pib_per_capita)
  log_emendas_per_cap      : log(emendas_per_cap)
  regiao                   : Norte | Nordeste | Centro-Oeste | Sudeste | Sul
  pos_ec86                 : 1 se ano >= 2015 (EC 86 emendas impositivas saúde)
  pos_ec100                : 1 se ano >= 2019 (EC 100/109 emendas impositivas geral)

  ── Proxies históricas (threat ao IV) ─────────────────────────────────────────
  pib_pc_1991              : PIB per capita estadual em 1991 (R$ 2010, IPEADATA PIBPCE)
  log_pib_pc_1991          : log(pib_pc_1991)
  gini_baseline            : primeiro Gini disponível por estado no painel PNAD (≈2012)

  Nota: Estas colunas são usadas para testar se o instrumento (distorcao_cadeiras)
  permanece válido após controlar pela situação pré-constitucional.
  Ver seção 4.2 do paper e src/analysis/iv.py::run_conditional_iv().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.tse import get_seats_dataframe

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

REGIAO_MAP: dict[str, str] = {
    "AC": "Norte",    "AM": "Norte",    "AP": "Norte",    "PA": "Norte",
    "RO": "Norte",    "RR": "Norte",    "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "ES": "Sudeste",  "MG": "Sudeste",  "RJ": "Sudeste",  "SP": "Sudeste",
    "PR": "Sul",      "RS": "Sul",      "SC": "Sul",
}


def build_panel(
    gini_df: pd.DataFrame,
    emendas_df: pd.DataFrame,
    pib_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Junta todos os DataFrames num painel balanceado estado × ano.

    Parâmetros
    ----------
    gini_df     : [uf, ano, gini]         — de src.collect.ibge.get_gini()
    emendas_df  : [uf, ano, valor_pago]   — de src.collect.siga (agregado por uf+ano)
    pib_df      : [uf, ano, pib_per_capita] — de src.collect.ibge.get_pib_per_capita()

    Retorna
    -------
    pd.DataFrame com o painel completo, salvo em data/processed/panel.parquet
    """
    seats = get_seats_dataframe()

    # Base: produto cartesiano UF × anos disponíveis no Gini
    anos = sorted(gini_df["ano"].dropna().unique())
    ufs = seats["uf"].tolist()

    panel = pd.MultiIndex.from_product([ufs, anos], names=["uf", "ano"])
    df = pd.DataFrame(index=panel).reset_index()

    # Representação (invariante no tempo na legislatura atual)
    seats_cols = [
        "uf", "cadeiras_camara", "cadeiras_senado", "cadeiras_total",
        "populacao_2022", "cadeiras_camara_per_cap", "cadeiras_total_per_cap",
        "representacao_relativa", "status_cap",
    ]
    df = df.merge(seats[seats_cols], on="uf", how="left")
    df = df.rename(columns={"populacao_2022": "populacao"})

    # Indicadores de reforma (tratamento DiD)
    df["pos_ec86"] = (df["ano"] >= 2015).astype(int)
    df["pos_ec100"] = (df["ano"] >= 2019).astype(int)

    # Gini
    df = df.merge(gini_df[["uf", "ano", "gini"]], on=["uf", "ano"], how="left")

    # Gini defasado 1 período (controle causal — baseline)
    df = df.sort_values(["uf", "ano"])
    df["gini_lag1"] = df.groupby("uf")["gini"].shift(1)

    # Emendas per capita (mediador)
    # Usa valor_empenhado (comprometido) em vez de valor_pago para consistência temporal:
    # em 2014 apenas ~2% das emendas foram pagas no mesmo ano, mas >99% foram empenhadas.
    emendas_agg = emendas_df.rename(columns={"valor_empenhado": "emendas_total"})
    df = df.merge(emendas_agg[["uf", "ano", "emendas_total"]], on=["uf", "ano"], how="left")
    df["emendas_per_cap"] = np.where(
        df["populacao"] > 0,
        df["emendas_total"] / df["populacao"] * 100_000,
        np.nan,
    )

    # PIB per capita
    df = df.merge(pib_df[["uf", "ano", "pib_per_capita"]], on=["uf", "ano"], how="left")
    df["log_pib_per_cap"] = np.log(df["pib_per_capita"].replace(0, np.nan))
    df["log_emendas_per_cap"] = np.log(df["emendas_per_cap"].replace(0, np.nan))

    # Região
    df["regiao"] = df["uf"].map(REGIAO_MAP)

    # ── Proxies históricas (controle para ameaça ao IV) ─────────────────────
    # Causalidade potencial: Desig_1988 → Cadeiras E Desig_1988 → Gini_atual
    # Se essa path existir, o instrumento não satisfaz a exclusion restriction.
    # Controlando para o estado pré-constitucional, o IV torna-se válido
    # condicionalmente ("conditional IV"). Ver seção 4.2 do paper.
    try:
        from src.collect.ibge import get_historical_proxy
        hist = get_historical_proxy(cache=True)
        df = df.merge(hist[["uf", "pib_pc_1991", "log_pib_pc_1991", "gini_baseline"]],
                      on="uf", how="left")
    except Exception as _e:
        import warnings
        warnings.warn(f"Proxy histórica não disponível ({_e}). "
                      "As colunas pib_pc_1991/gini_baseline estarão ausentes do painel.")
        df["pib_pc_1991"] = np.nan
        df["log_pib_pc_1991"] = np.nan
        df["gini_baseline"] = np.nan

    # Instrumentos (IV)
    # Z1: dummies de cap (binário: no teto máximo ou mínimo constitucional)
    df["iv_cap_max"] = (df["status_cap"] == "maximo").astype(int)
    df["iv_cap_min"] = (df["status_cap"] == "minimo").astype(int)
    # Z2: diferença entre cadeiras observadas e cadeiras "justas" (proporcional pura)
    total_pop = df.groupby("ano")["populacao"].transform("sum")
    df["cadeiras_justas"] = df["populacao"] / total_pop * 513
    df["distorcao_cadeiras"] = df["cadeiras_camara"] - df["cadeiras_justas"]

    path = PROCESSED_DIR / "panel.parquet"
    df.to_parquet(path, index=False)
    print(f"Painel salvo em {path} — {len(df)} observações, {df['uf'].nunique()} UFs, {df['ano'].nunique()} anos")
    return df


if __name__ == "__main__":
    from src.collect.ibge import get_gini, get_pib_per_capita
    from src.collect.siga import get_todas_emendas_por_uf
    import os

    token = os.environ.get("TRANSPARENCIA_API_TOKEN", "")
    # Anos onde emendas + gini se sobrepõem (emendas: 2014-2024; gini: 2012-2024)
    ANOS = list(range(2014, 2025))

    print("Carregando emendas (cache)...")
    emendas = get_todas_emendas_por_uf(anos=ANOS, api_token=token)

    print("Carregando Gini (cache/IPEADATA)...")
    gini = get_gini(anos=ANOS)

    print("Carregando PIB per capita (cache/IPEADATA)...")
    pib = get_pib_per_capita(anos=ANOS)

    panel = build_panel(gini, emendas, pib)
    print(panel.head(10).to_string(index=False))

    print("\n=== Estatísticas descritivas ===")
    print(panel[["representacao_relativa", "emendas_per_cap", "gini", "pib_per_capita"]].describe().to_string())

    print("\n=== Missings por coluna ===")
    print(panel.isnull().sum().to_string())
