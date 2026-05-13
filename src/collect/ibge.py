"""
src/collect/ibge.py

Coleta indicadores estaduais via IPEADATA (Ipea Data API):
  - Coeficiente de Gini (PNAD Contínua/A, série PNADCA_GINIUF)
  - PIB per capita estadual (Contas Regionais, série PIBPCE, preços de 2010)

API IPEADATA: http://www.ipeadata.gov.br/api/odata4/
"""

from __future__ import annotations

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


if __name__ == "__main__":
    print("=== Gini Estadual (últimos 3 anos) ===")
    gini = get_gini()
    recent = gini[gini["ano"] >= gini["ano"].max() - 2]
    print(recent.pivot(index="uf", columns="ano", values="gini").to_string())

    print("\n=== PIB per capita Estadual (últimos 3 anos disponíveis) ===")
    pib = get_pib_per_capita()
    recent_pib = pib[pib["ano"] >= pib["ano"].max() - 2]
    print(recent_pib.pivot(index="uf", columns="ano", values="pib_per_capita").to_string())
