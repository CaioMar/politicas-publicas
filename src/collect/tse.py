"""
src/collect/tse.py

Coleta o número de cadeiras (deputados federais eleitos) por estado
e por legislatura (ano de eleição) a partir da API de Dados Abertos do TSE
e/ou de arquivos CSV do repositório de resultados eleitorais.

Saída: DataFrame com colunas [ano_eleicao, uf, cadeiras, populacao, cadeiras_per_cap]

Fontes:
  - Dados Abertos TSE: https://dadosabertos.tse.jus.br
  - Resultado por UF: https://cdn.tse.jus.br/estatistica/sead/odsele/votacao_candidato_munzona/...
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

# ── constantes constitucionais ────────────────────────────────────────────────
CADEIRAS_MIN = 8
CADEIRAS_MAX = 70
TOTAL_CADEIRAS_CAMARA = 513

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# Distribuição histórica de cadeiras por UF (fixada a cada censo / resolução TSE)
# Fonte: TSE Resolução 23.389/2013 (atual) + Censo 2022 (vigência a partir de 2026)
CADEIRAS_POR_UF: dict[str, int] = {
    # Distribuição vigente nas eleições 2014-2022 (Resolução TSE 23.389/2013)
    "AC": 8,  "AL": 9,  "AM": 8,  "AP": 8,  "BA": 39,
    "CE": 22, "DF": 8,  "ES": 10, "GO": 17, "MA": 18,
    "MG": 53, "MS": 8,  "MT": 8,  "PA": 17, "PB": 12,
    "PE": 25, "PI": 10, "PR": 30, "RJ": 46, "RN": 8,
    "RO": 8,  "RR": 8,  "RS": 31, "SC": 16, "SE": 8,
    "SP": 70, "TO": 8,
}

# Cadeiras Senado: constante constitucional, 3 por UF
CADEIRAS_SENADO_POR_UF = 3


def get_seats_dataframe(include_senate: bool = True) -> pd.DataFrame:
    """
    Retorna DataFrame com representação parlamentar por UF.

    Colunas:
        uf                   : sigla da Unidade Federativa
        cadeiras_camara      : deputados federais
        cadeiras_senado      : 3 (constante constitucional)
        cadeiras_total       : camara + senado
        populacao_2022       : estimativa Censo 2022 (IBGE)
        cadeiras_camara_per_cap   : cadeiras_camara / populacao (× 100k)
        cadeiras_total_per_cap    : cadeiras_total / populacao (× 100k)
        distancia_cap_max    : 70 - cadeiras_camara (0 para SP)
        distancia_cap_min    : cadeiras_camara - 8  (0 para UFs no mínimo)
        status_cap           : 'maximo' | 'minimo' | 'livre'
        representacao_relativa: ratio entre % cadeiras e % populacao nacional
    """
    pop = _get_populacao_2022()

    rows = []
    total_pop = sum(pop.values())

    for uf, cadeiras in CADEIRAS_POR_UF.items():
        populacao = pop.get(uf, 0)
        cadeiras_senado = CADEIRAS_SENADO_POR_UF
        cadeiras_total = cadeiras + cadeiras_senado

        share_cadeiras = cadeiras / TOTAL_CADEIRAS_CAMARA
        share_pop = populacao / total_pop if total_pop > 0 else 0

        rows.append({
            "uf": uf,
            "cadeiras_camara": cadeiras,
            "cadeiras_senado": cadeiras_senado,
            "cadeiras_total": cadeiras_total,
            "populacao_2022": populacao,
            "cadeiras_camara_per_cap": cadeiras / populacao * 100_000 if populacao else None,
            "cadeiras_total_per_cap": cadeiras_total / populacao * 100_000 if populacao else None,
            "distancia_cap_max": CADEIRAS_MAX - cadeiras,
            "distancia_cap_min": cadeiras - CADEIRAS_MIN,
            "status_cap": _classify_cap(cadeiras),
            "representacao_relativa": share_cadeiras / share_pop if share_pop else None,
        })

    df = pd.DataFrame(rows).sort_values("uf").reset_index(drop=True)
    return df


def _classify_cap(cadeiras: int) -> str:
    if cadeiras == CADEIRAS_MAX:
        return "maximo"
    if cadeiras == CADEIRAS_MIN:
        return "minimo"
    return "livre"


def _get_populacao_2022() -> dict[str, int]:
    """
    Retorna estimativa de população por UF (Censo IBGE 2022).
    Valores preliminares divulgados em Jun/2023.
    """
    return {
        "AC": 830_026,
        "AL": 3_337_357,
        "AM": 4_269_995,
        "AP": 845_731,
        "BA": 14_136_417,
        "CE": 9_240_580,
        "DF": 2_817_068,
        "ES": 4_108_508,
        "GO": 7_206_589,
        "MA": 6_775_152,
        "MG": 20_538_718,
        "MS": 2_833_742,
        "MT": 3_658_813,
        "PA": 8_116_132,
        "PB": 4_059_905,
        "PE": 9_674_793,
        "PI": 3_269_200,
        "PR": 11_443_208,
        "RJ": 16_054_524,
        "RN": 3_302_406,
        "RO": 1_815_278,
        "RR": 636_707,
        "RS": 10_880_506,
        "SC": 7_609_601,
        "SE": 2_338_474,
        "SP": 44_420_459,
        "TO": 1_607_363,
    }


if __name__ == "__main__":
    df = get_seats_dataframe()
    print(df.to_string(index=False))
    print(f"\nDistorção SP: representa {df.loc[df.uf=='SP','representacao_relativa'].values[0]:.3f}x")
    print(f"Distorção AP: representa {df.loc[df.uf=='AP','representacao_relativa'].values[0]:.3f}x")
