"""
src/collect/ibge.py

Coleta indicadores estaduais via IPEADATA (Ipea Data API):
  - Coeficiente de Gini (PNAD Contínua/A, série PNADCA_GINIUF)
  - PIB per capita estadual (Contas Regionais, série PIBPCE, preços de 2010)

API IPEADATA: http://www.ipeadata.gov.br/api/odata4/
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

IPEADATA_BASE = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{code}')"

# Mapa de código IBGE de UF (2 dígitos) → sigla
UF_CODIGO_PARA_SIGLA = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}


def _fetch_ipeadata_states(serie_code: str, value_col: str) -> pd.DataFrame:
    """
    Baixa uma série do IPEADATA e retorna somente os registros de estados.
    Colunas resultado: [uf, ano, <value_col>]
    """
    url = IPEADATA_BASE.format(code=serie_code)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    vals = r.json().get("value", [])

    df = pd.DataFrame(vals)
    df = df[df["NIVNOME"] == "Estados"].copy()
    df["uf"] = df["TERCODIGO"].astype(str).str.zfill(2).map(UF_CODIGO_PARA_SIGLA)
    df["ano"] = pd.to_datetime(df["VALDATA"], utc=True).dt.year
    df[value_col] = pd.to_numeric(df["VALVALOR"], errors="coerce")

    return (
        df[["uf", "ano", value_col]]
        .dropna()
        .sort_values(["uf", "ano"])
        .reset_index(drop=True)
    )


def get_gini(anos: list[int] | None = None, cache: bool = True) -> pd.DataFrame:
    """
    Retorna o Gini estadual da PNAD Contínua (IPEADATA: PNADCA_GINIUF).

    Colunas: [uf, ano, gini]
    Cobertura: 2012–presente, 27 UFs.
    """
    cache_path = RAW_DIR / "ibge_gini.parquet"

    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = _fetch_ipeadata_states("PNADCA_GINIUF", "gini")
        df.to_parquet(cache_path, index=False)

    if anos:
        df = df[df["ano"].isin(anos)]

    return df


def get_pib_per_capita(anos: list[int] | None = None, cache: bool = True) -> pd.DataFrame:
    """
    Retorna PIB per capita estadual (preços de 2010, R$) das Contas Regionais.
    IPEADATA série PIBPCE.

    Colunas: [uf, ano, pib_per_capita]
    Cobertura: 1985–2023, 27 UFs.
    """
    cache_path = RAW_DIR / "ibge_pib_pc.parquet"

    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = _fetch_ipeadata_states("PIBPCE", "pib_per_capita")
        df.to_parquet(cache_path, index=False)

    if anos:
        df = df[df["ano"].isin(anos)]

    return df


def get_historical_proxy(cache: bool = True) -> pd.DataFrame:
    """
    Retorna proxy de desenvolvimento histórico (pré-Constituição) por UF.

    Estratégia:
    -----------
    Não existe série de Gini estadual contínua antes de 2001 no IPEADATA.
    Usamos dois indicadores como proxy da situação histórica de cada estado
    no momento em que as cadeiras foram fixadas (1988):

      pib_pc_1991   - PIB per capita estadual de 1991 (Contas Regionais,
                      IPEADATA série PIBPCE, preços de 2010 R$).
                      Alta correlação com desenvolvimento/desigualdade inicial.

      gini_baseline - Primeiro Gini disponível por estado no painel PNAD
                      Contínua (normalmente 2012 ou 2013). Servem como
                      estado inicial, pois a correlação serial do Gini
                      estadual brasileiro é > 0.95 (Lustig et al. 2013).

    Limitação explicitamente documentada na Seção 4.2 do paper:
    a Gini exata de 1988 não está disponível no IPEADATA/PNADCA.
    O PIB 1991 é proxy plausível mas imperfeito para a desigualdade no
    momento da promulgação constitucional.

    Retorna
    -------
    pd.DataFrame com colunas: [uf, pib_pc_1991, gini_baseline]
    """
    cache_path = RAW_DIR / "ibge_hist_proxy.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    # ── PIB per capita 1991 ───────────────────────────────────────────────────
    pib_all = _fetch_ipeadata_states("PIBPCE", "pib_per_capita")
    pib_1991 = (
        pib_all[pib_all["ano"] == 1991][["uf", "pib_per_capita"]]
        .rename(columns={"pib_per_capita": "pib_pc_1991"})
    )
    if pib_1991.empty:
        # Fallback: usar 1995 se 1991 não disponível
        pib_1991 = (
            pib_all[pib_all["ano"] == 1995][["uf", "pib_per_capita"]]
            .rename(columns={"pib_per_capita": "pib_pc_1991"})
        )

    # ── Gini baseline (primeiro ano disponível por UF) ────────────────────────
    gini_all = _fetch_ipeadata_states("PNADCA_GINIUF", "gini")
    gini_baseline = (
        gini_all.sort_values("ano")
        .groupby("uf", as_index=False)
        .first()[["uf", "gini"]]
        .rename(columns={"gini": "gini_baseline"})
    )

    proxy = pib_1991.merge(gini_baseline, on="uf", how="outer")
    proxy["log_pib_pc_1991"] = np.log(proxy["pib_pc_1991"].replace(0, np.nan))

    proxy.to_parquet(cache_path, index=False)
    print(f"Proxy histórico salvo: {len(proxy)} UFs")
    return proxy


if __name__ == "__main__":
    print("=== Gini Estadual (últimos 3 anos) ===")
    gini = get_gini()
    recent = gini[gini["ano"] >= gini["ano"].max() - 2]
    print(recent.pivot(index="uf", columns="ano", values="gini").to_string())

    print("\n=== PIB per capita Estadual (últimos 3 anos disponíveis) ===")
    pib = get_pib_per_capita()
    recent_pib = pib[pib["ano"] >= pib["ano"].max() - 2]
    print(recent_pib.pivot(index="uf", columns="ano", values="pib_per_capita").to_string())
