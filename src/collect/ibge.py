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

    Fontes
    ------
    1. gini_1991  — Gini estadual de 1991 do Atlas do Desenvolvimento Humano
                    (IPEADATA série ADH_GINI, calculado pelo PNUD/IPEA com base
                    no Censo Demográfico 1991). Cobre 27 UFs, valores em [0,1].
                    É o proxy mais próximo do Gini vigente quando a Constituição
                    de 1988 fixou os limites de cadeiras.
                    Anos disponíveis: 1991, 2000, 2010.

    2. pib_pc_1991 — PIB per capita estadual de 1991 (Contas Regionais,
                     IPEADATA série PIBPCE, preços de 2010 R$).

    3. gini_baseline — primeiro Gini disponível no painel PNAD Contínua
                       (≈ 2012), retido como controle adicional de persistência.

    Retorna
    -------
    pd.DataFrame com colunas:
        [uf, gini_1991, log_pib_pc_1991, pib_pc_1991, gini_baseline]
    """
    cache_path = RAW_DIR / "ibge_hist_proxy.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    # ── Gini 1991 via Atlas do Desenvolvimento Humano ─────────────────────────
    adh_url = ("http://www.ipeadata.gov.br/api/odata4/"
               "ValoresSerie(SERCODIGO='ADH_GINI')")
    r = requests.get(adh_url, timeout=30)
    r.raise_for_status()
    adh_raw = pd.DataFrame(r.json().get("value", []))
    adh_est = adh_raw[adh_raw["NIVNOME"] == "Estados"].copy()
    adh_est["uf"] = (
        adh_est["TERCODIGO"].astype(str).str.zfill(2).map(UF_CODIGO_PARA_SIGLA)
    )
    adh_est["ano"] = pd.to_datetime(adh_est["VALDATA"], utc=True).dt.year
    adh_est["gini_adh"] = pd.to_numeric(adh_est["VALVALOR"], errors="coerce")
    # Use 1991 as primary baseline; fall back to 2000 if missing
    gini_1991 = (
        adh_est[adh_est["ano"] == 1991][["uf", "gini_adh"]]
        .rename(columns={"gini_adh": "gini_1991"})
        .dropna()
    )
    if gini_1991.empty:
        gini_1991 = (
            adh_est[adh_est["ano"] == 2000][["uf", "gini_adh"]]
            .rename(columns={"gini_adh": "gini_1991"})
            .dropna()
        )

    # ── PIB per capita 1991 ───────────────────────────────────────────────────
    pib_all = _fetch_ipeadata_states("PIBPCE", "pib_per_capita")
    pib_1991 = (
        pib_all[pib_all["ano"] == 1991][["uf", "pib_per_capita"]]
        .rename(columns={"pib_per_capita": "pib_pc_1991"})
    )
    if pib_1991.empty:
        pib_1991 = (
            pib_all[pib_all["ano"] == 1995][["uf", "pib_per_capita"]]
            .rename(columns={"pib_per_capita": "pib_pc_1991"})
        )

    # ── Gini baseline (PNAD Contínua, ≈2012) ─────────────────────────────────
    gini_all = _fetch_ipeadata_states("PNADCA_GINIUF", "gini")
    gini_baseline = (
        gini_all.sort_values("ano")
        .groupby("uf", as_index=False)
        .first()[["uf", "gini"]]
        .rename(columns={"gini": "gini_baseline"})
    )

    # ── Juntar tudo ───────────────────────────────────────────────────────────
    proxy = gini_1991.merge(pib_1991, on="uf", how="outer")
    proxy = proxy.merge(gini_baseline, on="uf", how="outer")
    proxy["log_pib_pc_1991"] = np.log(proxy["pib_pc_1991"].replace(0, np.nan))

    proxy.to_parquet(cache_path, index=False)
    print(f"Proxy histórico salvo: {len(proxy)} UFs | "
          f"gini_1991 disponível: {proxy['gini_1991'].notna().sum()} UFs")
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
