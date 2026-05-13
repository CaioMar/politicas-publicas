"""
src/collect/siga.py

Coleta dados de emendas parlamentares executadas por UF a partir de:
  1. Portal da Transparência (API REST)
  2. SIGA Brasil - Senado Federal (arquivos CSV públicos)

As emendas são o MEDIADOR do modelo causal: medem o quanto de recurso
federal foi direcionado para cada estado via poder parlamentar.

Fonte SIGA Brasil:
  https://www12.senado.leg.br/orcamento/sigabrasil
  Arquivo: Execução - Emendas Parlamentares (download anual)

Fonte Portal Transparência (API v2):
  https://portaldatransparencia.gov.br/api-de-dados/emendas
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Portal da Transparência ────────────────────────────────────────────────────
# URL migrada em 2025 para api.portaldatransparencia.gov.br
# NOTA: a API ignora tamanhoDaPagina e retorna exatamente 15 registros por página.
TRANSPARENCIA_BASE = "https://api.portaldatransparencia.gov.br/api-de-dados/emendas"
_PAGE_SIZE = 15  # tamanho real retornado pela API (ignora o parâmetro)


def _fetch_page(ano: int, pagina: int, api_token: str) -> list[dict]:
    """Busca uma página de emendas e retorna lista de dicts brutos."""
    headers = {"chave-api-dados": api_token, "Accept": "application/json"}
    params = {"ano": ano, "pagina": pagina, "tamanhoDaPagina": _PAGE_SIZE}
    resp = requests.get(TRANSPARENCIA_BASE, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_items(items: list[dict], ano: int) -> pd.DataFrame:
    """Converte lista de dicts da API em DataFrame normalizado."""
    rows = []
    for item in items:
        localidade = item.get("localidadeDoGasto", "")
        uf = _extract_uf_from_localidade(localidade)
        rows.append({
            "ano": ano,
            "uf": uf,
            "codigo_emenda": item.get("codigoEmenda", ""),
            "tipo_emenda": item.get("tipoEmenda", ""),
            "autor": item.get("nomeAutor", item.get("autor", "")),
            "funcao": item.get("funcao", ""),
            "subfuncao": item.get("subfuncao", ""),
            "valor_empenhado": _to_float(item.get("valorEmpenhado")),
            "valor_liquidado": _to_float(item.get("valorLiquidado")),
            "valor_pago": _to_float(item.get("valorPago")),
            "valor_resto_pago": _to_float(item.get("valorRestoPago")),
        })
    return pd.DataFrame(rows)


def get_emendas_ano(
    ano: int,
    api_token: str,
    delay: float = 0.12,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Coleta TODOS os registros de emendas de um ano, paginando completamente.

    A API retorna exatamente 15 registros por página (ignora tamanhoDaPagina).
    Salva cache em data/raw/emendas_{ano}.parquet para evitar re-coleta.

    Parâmetros
    ----------
    ano       : ano de referência (ex: 2023)
    api_token : chave da API do Portal da Transparência
    delay     : segundos entre requisições (padrão 0.12 ≈ 8 req/s, dentro do limite)
    cache     : se True, usa cache em disco quando disponível
    """
    cache_path = RAW_DIR / f"emendas_{ano}.parquet"
    if cache and cache_path.exists():
        print(f"  [{ano}] carregando cache: {cache_path.name}")
        return pd.read_parquet(cache_path)

    all_frames: list[pd.DataFrame] = []
    pagina = 1
    while True:
        items = _fetch_page(ano, pagina, api_token)
        if not items:
            break
        all_frames.append(_parse_items(items, ano))
        if pagina % 100 == 0:
            acum = pagina * _PAGE_SIZE
            print(f"  [{ano}] pg {pagina} — ~{acum} registros coletados...", flush=True)
        pagina += 1
        time.sleep(delay)

    if not all_frames:
        return pd.DataFrame()

    df = pd.concat(all_frames, ignore_index=True)
    df.to_parquet(cache_path, index=False)
    print(f"  [{ano}] ✓ {len(df)} registros salvos em {cache_path.name}")
    return df


def get_todas_emendas_por_uf(
    anos: list[int],
    api_token: str,
    delay: float = 0.12,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Coleta emendas de todos os anos e agrega por [ano, uf].

    Retorna DataFrame com colunas:
      ano, uf, valor_empenhado, valor_liquidado, valor_pago, valor_resto_pago
    """
    all_frames = []
    for ano in anos:
        print(f"Coletando emendas {ano}...")
        df_ano = get_emendas_ano(ano, api_token, delay=delay, cache=cache)
        all_frames.append(df_ano)

    if not all_frames:
        return pd.DataFrame()

    df_all = pd.concat(all_frames, ignore_index=True)

    df_agg = (
        df_all[df_all["uf"].str.len() == 2]  # remove registros sem UF válida (e.g. "Nacional")
        .groupby(["ano", "uf"])[["valor_empenhado", "valor_liquidado", "valor_pago", "valor_resto_pago"]]
        .sum()
        .reset_index()
    )
    return df_agg


# ── SIGA Brasil (CSV público, sem autenticação) ────────────────────────────────
# URL dos arquivos de execução de emendas no SIGA Brasil
# Formato: planilha de execução anual baixada manualmente ou via scraping
# Ver: https://www12.senado.leg.br/orcamento/sigabrasil
# Os arquivos CSV podem ser baixados em:
# Consultar → Execução → Emendas Parlamentares → Exportar CSV

def load_siga_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Carrega e normaliza um CSV exportado do SIGA Brasil.

    O CSV do SIGA tem colunas como:
      'Autor', 'Tipo', 'UF do autor', 'Função', 'Subfunção',
      'Dotação Inicial', 'Empenhado', 'Liquidado', 'Pago',
      'RP não processado', 'Ano', ...

    Retorna DataFrame com [ano, uf_autor, tipo_emenda, empenhado, liquidado, pago].
    """
    df = pd.read_csv(
        filepath,
        sep=";",
        encoding="latin-1",
        decimal=",",
        thousands=".",
        dtype=str,
    )
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_rename = {
        "ano": "ano",
        "uf_do_autor": "uf_autor",
        "tipo": "tipo_emenda",
        "empenhado": "empenhado",
        "liquidado": "liquidado",
        "pago": "pago",
    }
    df = df.rename(columns={k: v for k, v in col_rename.items() if k in df.columns})

    for col in ["empenhado", "liquidado", "pago"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ano" in df.columns:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce")

    return df


# Mapa nome por extenso da UF → sigla
_NOME_UF_SIGLA: dict[str, str] = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAZONAS": "AM",
    "BAHIA": "BA", "CEARA": "CE", "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES", "GOIAS": "GO", "MARANHAO": "MA",
    "MATO GROSSO DO SUL": "MS", "MATO GROSSO": "MT", "MINAS GERAIS": "MG",
    "PARA": "PA", "PARAIBA": "PB", "PARANA": "PR", "PERNAMBUCO": "PE",
    "PIAUI": "PI", "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS", "RONDONIA": "RO", "RORAIMA": "RR",
    "SANTA CATARINA": "SC", "SAO PAULO": "SP", "SERGIPE": "SE",
    "TOCANTINS": "TO",
}


def _extract_uf_from_localidade(localidade: str) -> str:
    """
    Extrai sigla da UF da string retornada pela API.
    Exemplos: "PERNAMBUCO (UF)" → "PE", "SAO PAULO (UF)" → "SP"
    """
    import unicodedata
    nome = localidade.upper().replace(" (UF)", "").strip()
    # Normaliza acentos
    nome_norm = unicodedata.normalize("NFD", nome)
    nome_norm = "".join(c for c in nome_norm if not unicodedata.combining(c))
    return _NOME_UF_SIGLA.get(nome_norm, "")


def _to_float(value) -> float:
    if value is None:
        return 0.0
    try:
        # API retorna valores formatados com ponto de milhar e vírgula decimal
        # ex: "6.467,00" → 6467.0
        s = str(value).replace(".", "").replace(",", ".")
        return float(s)
    except (ValueError, TypeError):
        return 0.0


if __name__ == "__main__":
    import os
    from pathlib import Path as _Path

    # Carrega .env se existir
    _env = _Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    token = os.environ.get("TRANSPARENCIA_API_TOKEN", "")
    if not token:
        print("Defina TRANSPARENCIA_API_TOKEN no arquivo .env ou como variável de ambiente.")
        print("Cadastro em: https://portaldatransparencia.gov.br/api-de-dados/cadastrar-email")
    else:
        print("Token encontrado. Coletando emendas de 2022 e 2023 (teste)...")
        df = get_todas_emendas_por_uf([2022, 2023], api_token=token)
        print(df.sort_values("valor_pago", ascending=False).to_string(index=False))
