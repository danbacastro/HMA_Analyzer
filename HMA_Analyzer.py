import io
import sys
import zipfile
from datetime import datetime
from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
import streamlit as st

# Sugestões inteligentes (opcional)
try:
    from rapidfuzz import process, fuzz
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False

import re
import unicodedata
import math

# ==== AUTH (colar após os imports, antes de qualquer st.set_page_config) ====
import streamlit as st
import os

AUTH_USERNAME = "hma-scih"  # fixo

def _get_secret_password() -> str:
    # tenta ler de secrets TOML: [app] password_hma="xxx"
    try:
        return st.secrets["app"]["password_hma"]
    except Exception:
        # fallback opcional (útil localmente): variável de ambiente
        return os.environ.get("PASSWORD_HMA", "")

def require_login():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    # se já autenticado, só oferece logout na sidebar
    if st.session_state.auth_ok:
        with st.sidebar:
            st.markdown("---")
            if st.button("Sair (logout)", use_container_width=True):
                st.session_state.auth_ok = False
                st.experimental_rerun()
        return  # segue para o app

    # caso não autenticado: mostra tela de login e bloqueia o resto
    st.title("🔐 HMA Analyzer — Login")
    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Usuário", value="", placeholder="hma-scih")
        pw   = st.text_input("Senha", value="", type="password")
        ok   = st.form_submit_button("Entrar")

    if ok:
        secret_pw = _get_secret_password()
        if user.strip() == AUTH_USERNAME and pw == str(secret_pw):
            st.session_state.auth_ok = True
            st.success("Login efetuado!")
            st.experimental_rerun()
        else:
            st.error("Usuário ou senha inválidos.")

    st.stop()  # impede o resto do app enquanto não logar

# **Chame isso antes do resto do app**
require_login()
# ==== FIM AUTH ====

# =========================
# Helpers
# =========================

EMPTY_LABEL = "(sem informação)"

def _strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFKD", text)
    return "".join([c for c in text if not unicodedata.combining(c)])

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = _strip_accents(text).lower().strip()
    t = re.sub(r"[\\/|]+", " ", t)
    t = re.sub(r"[-_]+", " ", t)
    t = re.sub(r"[^a-z0-9\.\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def dynamic_gaps(n_groups: int, n_bars_per_group: int) -> Tuple[float, float]:
    if n_groups <= 0: n_groups = 1
    if n_bars_per_group <= 0: n_bars_per_group = 1
    base = 0.30 - 0.012 * n_groups - 0.008 * n_bars_per_group
    bargap = max(0.05, min(0.40, base))
    bargroupgap = max(0.05, min(0.40, base * 0.8))
    return float(bargap), float(bargroupgap)

def round_up_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return int(math.ceil(value / multiple) * multiple)

# Inferir ano pelo nome do arquivo
def infer_year_from_filename(name: str) -> Optional[int]:
    if not name:
        return None
    m = re.search(r'((?:19|20)\d{2})', name)
    return int(m.group(1)) if m else None

# Parse PT-BR "dia mês" + fallback dd/mm
MESES_MAP = {
    "janeiro":1, "fevereiro":2, "marco":3, "março":3, "abril":4, "maio":5, "junho":6,
    "julho":7, "agosto":8, "setembro":9, "outubro":10, "novembro":11, "dezembro":12
}
MESES_PT = {
    1:"janeiro",2:"fevereiro",3:"março",4:"abril",5:"maio",6:"junho",
    7:"julho",8:"agosto",9:"setembro",10:"outubro",11:"novembro",12:"dezembro"
}

def parse_pt_dates(col: pd.Series, default_year: Optional[int]) -> pd.DataFrame:
    s = col.astype(str).fillna("").str.lower().map(_strip_accents)
    m = s.str.extract(r'(?P<dia>\d{1,2})\s*(?P<mes>janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)')
    m2 = s.str.extract(r'(?P<dia2>\d{1,2})[/-](?P<mes2>\d{1,2})(?:[/-](?P<ano2>\d{2,4}))?')

    dia = pd.to_numeric(m["dia"], errors="coerce")
    mes = m["mes"].map(MESES_MAP)

    dia = dia.fillna(pd.to_numeric(m2["dia2"], errors="coerce"))
    mes = mes.fillna(pd.to_numeric(m2["mes2"], errors="coerce")).clip(1,12)

    ano = s.str.extract(r'((?:19|20)\d{2})')[0]
    ano = pd.to_numeric(ano, errors="coerce")
    if default_year is not None:
        ano = ano.fillna(default_year)

    dia = dia.where(dia.between(1,31), np.nan)
    mes = mes.where(mes.between(1,12), np.nan)

    out = pd.DataFrame({"dia_pars": dia, "mes_pars": mes, "ano_pars": ano})
    return out

def sync_multiselect_state(key: str, options: List, select_all_on_change: bool = True):
    opts_key = f"{key}__opts"
    cur_opts = list(options)
    prev_opts = st.session_state.get(opts_key, None)
    if prev_opts != cur_opts:
        st.session_state[opts_key] = cur_opts
        st.session_state[key] = cur_opts.copy() if select_all_on_change else []
    else:
        sel = st.session_state.get(key, cur_opts.copy())
        st.session_state[key] = [x for x in sel if x in cur_opts]

# strings seguras para contagens/agrupamentos
def safe_series_strings(s: pd.Series, empty_label=EMPTY_LABEL) -> pd.Series:
    out = s.astype(str)
    out = out.replace(["nan", "NaN", "None", "NONE"], "").str.strip()
    out = out.mask(out.eq(""), other=empty_label)
    return out

# =========================
# Paletas consistentes (cores fixas por categoria)
# =========================

def _get_palette_colors(n: int) -> List[str]:
    try:
        from plotly.colors import qualitative as q
        bank = []
        # concatenamos várias paletas para garantir bastante cores
        bank += list(getattr(q, "Set3", []))
        bank += list(getattr(q, "Plotly", []))
        bank += list(getattr(q, "Safe", []))
        bank += list(getattr(q, "Pastel", []))
        bank += list(getattr(q, "Bold", []))
        bank += list(getattr(q, "D3", []))
        if len(bank) < n:
            bank = (bank * ((n // len(bank)) + 1))[:n]
        else:
            bank = bank[:n]
        return bank
    except Exception:
        # fallback genérico
        base = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        if len(base) < n:
            base = (base * ((n // len(base)) + 1))[:n]
        return base[:n]

def _ensure_color_map(state_key: str, categories: List[str]) -> Dict[str, str]:
    """Garante um dict estável categoria->cor em session_state[state_key]."""
    if state_key not in st.session_state:
        st.session_state[state_key] = {}
    cmap = st.session_state[state_key]
    # mantém cores já atribuídas
    missing = [c for c in categories if c not in cmap]
    if missing:
        colors = _get_palette_colors(len(categories) + 5)  # margem
        # evita reassinar cores existentes
        used = set(cmap.values())
        for cat in missing:
            # pega a primeira cor ainda não usada
            color = next((col for col in colors if col not in used), None)
            if color is None:
                # se esgotou, apenas cíclica
                color = colors[len(used) % len(colors)]
            cmap[cat] = color
            used.add(color)
        st.session_state[state_key] = cmap
    # se alguma categoria saiu, mantemos o mapeamento (para estabilidade visual)
    return cmap

def color_map_for_series(series: pd.Series, state_key: str) -> Dict[str, str]:
    cats = sorted([c for c in series.dropna().astype(str).unique().tolist()])
    return _ensure_color_map(state_key, cats)

# =========================
# Streamlit – UI
# =========================

st.set_page_config(page_title="SCIH HMA — Analytics & Report", layout="wide")
st.title("🧫 SCIH HMA  — Analytics & Report")

st.markdown(
    """
1. Carregue a **planilha bruta** (CSV/XLSX)
2. Carregue a **planilha de tradução** (CSV)

OBS: Lembre-se de não alterar o estilo padrão das planilhas importadas no programa
    """
)

with st.sidebar:
    st.header("⚙️ Arquivos")
    data_file = st.file_uploader(
        "Planilha Bruta",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
    )
    trans_file = st.file_uploader(
        "Planilha de Padronização de Nomes",
        type=["csv"],
        accept_multiple_files=False,
    )

    st.markdown("---")
    st.header("🗓️ Filtro por Data")
    st.caption("Baseado na coluna de datas da planilha bruta.")
    date_filters_placeholder = st.container()

    st.markdown("---")
    st.header("📊 Controles de gráficos")
    min_count = st.number_input(
        "Gráfico de Prevalência - filtrar resultados com pelo menos N ocorrências",
        min_value=1, max_value=1000, value=1, step=1, key="min_count"
    )
    top_n = st.slider("N", 5, 50, 15, step=1, key="top_n")

    st.subheader("Eixo Y dos Gráficos de Barras)")
    y_multiple = st.number_input("Arredondar Y para múltiplos de:", min_value=1, value=2, step=1, key="y_mult")
    y_manual_on = st.checkbox("Definir Y máximo manualmente", value=False, key="y_manual_on")
    y_manual_val = st.number_input("Y máximo manual", min_value=1, value=10, step=1, key="y_manual_val")

    st.subheader("Pizzas")
    show_pcts = st.checkbox("Mostrar % nos Gráficos de Pizza", value=True, key="show_pcts")

    # Controlar exibição da categoria vazia
    show_empty = st.checkbox(f"Incluir '{EMPTY_LABEL}' nas análises", value=False, key="show_empty")

    st.markdown("---")
    st.subheader("Excluir Categorias")
    st.caption("Remova resultados que não deseja contabilizar (ex.: 'negativo', 'contaminante').")
    exclude_placeholder = st.empty()

# =========================
# Leitura de dados
# =========================

@st.cache_data(show_spinner=False)
def read_raw(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", "")
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(file, header=None, skiprows=2)
    return pd.read_csv(file, header=None, skiprows=2)

@st.cache_data(show_spinner=False)
def read_translation(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame(columns=["resultado","padronizado","tipo_micro","resultado_norm"])
    tdf = pd.read_csv(file)
    def pick(colnames: List[str], default_idx: int):
        for cand in colnames:
            for c in tdf.columns:
                if cand == c or normalize_text(cand) == normalize_text(c):
                    return c
        return tdf.columns[min(default_idx, tdf.shape[1]-1)]
    c_res = pick(["resultado","original","res","termo","from"], 0)
    c_pad = pick(["padronização","padronizacao","padroniza","correto","to"], 1)
    c_tip = pick(["tipo do micro-organismo","tipo_micro","tipo","classe"], 2 if tdf.shape[1] >= 3 else 1)
    out = tdf[[c_res, c_pad]].copy(); out.columns = ["resultado","padronizado"]
    out["tipo_micro"] = tdf[c_tip] if c_tip in tdf.columns else ""
    out = out.dropna(subset=["resultado","padronizado"])
    out["resultado_norm"] = out["resultado"].map(normalize_text)
    out = out.drop_duplicates(subset=["resultado_norm"], keep="last").reset_index(drop=True)
    return out

raw_df = read_raw(data_file)
if raw_df.empty:
    st.info("⬅️ Carregue a planilha *bruta* para começar.")
    st.stop()

# Check: quantas linhas de dados foram analisadas
st.caption(f"✅ Número de Dados Lidos: **{len(raw_df)}**.")

# Índices fixos: B=1, D=3, E=4, F=5
try:
    raw = pd.DataFrame({
        "data": pd.to_datetime(raw_df.iloc[:, 1], errors="coerce", dayfirst=True, infer_datetime_format=True),
        "setor": raw_df.iloc[:, 3].astype(str),
        "tipo_amostra": raw_df.iloc[:, 4].astype(str),
        "resultado_raw": raw_df.iloc[:, 5].astype(str),
    })
except Exception as e:
    st.error(f"Erro ao mapear colunas por índice (B,D,E,F): {e}")
    st.stop()

# Ano/Mês a partir de 'data' OU do texto PT-BR da coluna B
default_year = infer_year_from_filename(getattr(data_file, "name", ""))
parsed = parse_pt_dates(raw_df.iloc[:, 1], default_year)

raw["ano"] = raw["data"].dt.year
raw["mes_num"] = raw["data"].dt.month
raw["ano"] = raw["ano"].fillna(parsed["ano_pars"])
raw["mes_num"] = raw["mes_num"].fillna(parsed["mes_pars"])

need_fill = raw["data"].isna()
if need_fill.any():
    # Fallbacks seguros
    yr_fallback = default_year if default_year is not None else pd.Timestamp.today().year

    ano_fill = raw["ano"].astype("Int64")
    mes_fill = raw["mes_num"].astype("Int64")
    dia_fill = parsed["dia_pars"].astype("Int64")

    # Preenche NAs antes de converter para int nativo
    ano_fill = ano_fill.fillna(yr_fallback)
    mes_fill = mes_fill.fillna(1)
    dia_fill = dia_fill.fillna(1)

    parts = pd.DataFrame({
        "year":  ano_fill.astype(int),
        "month": mes_fill.astype(int),
        "day":   dia_fill.astype(int),
    })

    synth = pd.to_datetime(parts, errors="coerce")
    raw.loc[need_fill, "data"] = synth[need_fill]

raw["mes"] = raw["mes_num"].map(MESES_PT)

with st.expander("Prévia da planilha bruta (20 linhas)"):
    st.dataframe(raw_df.head(20), use_container_width=True)
with st.expander("Prévia já nomeada (20 linhas)"):
    st.dataframe(raw.head(20), use_container_width=True)

# =========================
# Tradução (CSV base) + preparação
# =========================

trans_tbl = read_translation(trans_file)
map_pad_base  = dict(zip(trans_tbl["resultado_norm"], trans_tbl["padronizado"].astype(str))) if not trans_tbl.empty else {}
map_tipo_base = dict(zip(trans_tbl["resultado_norm"], trans_tbl["tipo_micro"].astype(str))) if not trans_tbl.empty else {}

# Normaliza os textos da planilha bruta (vamos usar para padronizar DEPOIS do card de não mapeados)
res_norm = raw["resultado_raw"].map(normalize_text)

# =========================
# Filtros – Data e Exclusão
# =========================

with date_filters_placeholder:
    anos_disp = sorted(raw["ano"].dropna().astype(int).unique().tolist())
    meses_disp = sorted([m for m in raw["mes_num"].dropna().unique().tolist() if 1 <= m <= 12])
    meses_rotulos = [MESES_PT[m] for m in meses_disp]
    rot2num = {v: k for k, v in MESES_PT.items()}

    sync_multiselect_state("f_anos", anos_disp, select_all_on_change=True)
    sync_multiselect_state("f_meses", meses_rotulos, select_all_on_change=True)

    c1, c2 = st.columns(2)
    if c1.button("Selecionar todos (Ano)"):
        st.session_state["f_anos"] = anos_disp.copy()
    if c2.button("Limpar (Ano)"):
        st.session_state["f_anos"] = []

    sel_anos = st.multiselect("Ano(s)", options=anos_disp, key="f_anos")

    c3, c4 = st.columns(2)
    if c3.button("Selecionar todos (Mês)"):
        st.session_state["f_meses"] = meses_rotulos.copy()
    if c4.button("Limpar (Mês)"):
        st.session_state["f_meses"] = []

    sel_meses_rot = st.multiselect("Mês(es)", options=meses_rotulos, key="f_meses")

sel_meses_num = [rot2num[r] for r in sel_meses_rot] if sel_meses_rot else []
mask_year  = raw["ano"].isin(sel_anos) if len(sel_anos) > 0 else pd.Series([True]*len(raw), index=raw.index)
mask_month = raw["mes_num"].isin(sel_meses_num) if len(sel_meses_num) > 0 else pd.Series([True]*len(raw), index=raw.index)
mask_date  = mask_year & mask_month

# =========================
# Card: NÃO Mapeados (antes da padronização final)
# =========================

unmatched = (
    pd.DataFrame({"resultado": raw["resultado_raw"], "resultado_norm": res_norm})
    [~res_norm.isin(set(map_pad_base.keys()))]
    .drop_duplicates(subset=["resultado_norm"])
    .reset_index(drop=True)
)

def suggest_matches(unmatched_df: pd.DataFrame, mapping_dict: Dict[str, str], limit: int = 3, score_cutoff: int = 86) -> pd.DataFrame:
    out = unmatched_df.copy()
    out["sugestoes"] = ""
    if unmatched_df.empty or not mapping_dict or not HAVE_RAPIDFUZZ:
        return out
    targets = sorted(set(mapping_dict.values()))
    if not targets: return out
    sug = []
    for term in unmatched_df["resultado_norm"].astype(str):
        res = process.extract(term, targets, scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff, limit=limit)
        sug.append("; ".join(f"{cand} ({score})" for cand, score, _ in res))
    out["sugestoes"] = sug
    return out

unmatched_suggested = suggest_matches(unmatched, map_pad_base)

st.header("🔎 Resultados não Mapeados")
if unmatched_suggested.empty:
    st.success("Todos os resultados foram mapeados ✅")
    new_map = pd.DataFrame(columns=["resultado","padronizado","tipo_micro","resultado_norm"])
else:
    st.warning("Existem resultados sem padronização. Atualize o Arquivo de Padronização de Nomes.")
    editable = unmatched_suggested.copy()
    editable["padronizado"] = ""
    editable["tipo_micro"] = ""
    st.caption("Sugestões automáticas (se houver) aparecem em *sugestoes*.")
    edited = st.data_editor(
        editable[["resultado", "resultado_norm", "sugestoes", "padronizado", "tipo_micro"]],
        use_container_width=True,
        num_rows="dynamic",
        key="editor_unmatched",
    )

    new_map = edited.copy()
    new_map["padronizado"] = new_map["padronizado"].astype(str)
    new_map["tipo_micro"]  = new_map["tipo_micro"].astype(str) if "tipo_micro" in new_map.columns else ""
    new_map = new_map[new_map["padronizado"].str.strip() != ""]
    if not new_map.empty:
        new_map = new_map[["resultado", "padronizado", "tipo_micro"]]
        new_map["resultado_norm"] = new_map["resultado"].map(normalize_text)
        new_map = new_map.drop_duplicates(subset=["resultado_norm"], keep="last")
    else:
        new_map = pd.DataFrame(columns=["resultado","padronizado","tipo_micro","resultado_norm"])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if not new_map.empty:
            buffer = io.StringIO()
            new_map[["resultado", "padronizado", "tipo_micro"]].to_csv(buffer, index=False)
            st.download_button(
                "⬇️ Baixar novos mapeamentos (CSV)",
                data=buffer.getvalue(),
                file_name=f"novos_mapeamentos_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
    with col_b:
        template = unmatched_suggested[["resultado"]].copy()
        template.insert(1, "padronizado", "")
        template.insert(2, "tipo_micro", "")
        buf2 = io.StringIO()
        template.to_csv(buf2, index=False)
        st.download_button(
                "⬇️ Baixar template para tradução (CSV)",
                data=buf2.getvalue(),
                file_name=f"template_traducao_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
    with col_c:
        if trans_tbl is None or trans_tbl.empty:
            base_trad = pd.DataFrame(columns=["resultado","padronizado","tipo_micro"])
        else:
            base_trad = trans_tbl[["resultado","padronizado","tipo_micro"]].copy().fillna("")
        novos_trad = new_map[["resultado","padronizado","tipo_micro"]].copy().fillna("") if not new_map.empty else pd.DataFrame(columns=["resultado","padronizado","tipo_micro"])
        merged_trad = pd.concat([base_trad, novos_trad], ignore_index=True)
        if not merged_trad.empty:
            merged_trad["resultado_norm"] = merged_trad["resultado"].map(normalize_text)
            merged_trad = merged_trad.drop_duplicates(subset=["resultado_norm"], keep="last").reset_index(drop=True)
        else:
            merged_trad["resultado_norm"] = []
        buf_merged = io.StringIO()
        merged_trad[["resultado","padronizado","tipo_micro"]].to_csv(buf_merged, index=False)
        st.download_button(
            "⬇️ Baixar Planilha de Padronização de Nomes ATUALIZADO",
            data=buf_merged.getvalue(),
            file_name=f"traducao_atualizada_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="btn_download_traducao_atualizada"
        )

# =========================
# Padronização final (CSV base + runtime)
# =========================

rt_pad  = st.session_state.get("runtime_map_pad",  {})
rt_tipo = st.session_state.get("runtime_map_tipo", {})

if not new_map.empty:
    if "runtime_map_pad" not in st.session_state:
        st.session_state["runtime_map_pad"] = {}
    if "runtime_map_tipo" not in st.session_state:
        st.session_state["runtime_map_tipo"] = {}
    for _, r in new_map.iterrows():
        k = normalize_text(str(r["resultado"]))
        st.session_state["runtime_map_pad"][k] = str(r["padronizado"])
        tm = str(r.get("tipo_micro", "")).strip()
        if tm:
            st.session_state["runtime_map_tipo"][k] = tm
    rt_pad  = st.session_state["runtime_map_pad"]
    rt_tipo = st.session_state["runtime_map_tipo"]

map_pad_combined  = {**map_pad_base,  **rt_pad}
map_tipo_combined = {**map_tipo_base, **rt_tipo}

# Não mapeados viram "(sem informação)"
std_series = res_norm.map(map_pad_combined)
raw["resultado_std"] = std_series.where(
    std_series.notna() & (std_series.astype(str).str.strip() != ""),
    other=EMPTY_LABEL
)
raw["tipo_micro"] = res_norm.map(map_tipo_combined).fillna("")

# Recalcula bases filtradas
present_vals = raw.loc[mask_date, "resultado_std"].astype(str).fillna("").tolist()
suggest_exclude = sorted({v for v in present_vals if any(k in normalize_text(v) for k in ["negativ", "contamin"])})

options_ex = sorted(set(safe_series_strings(raw["resultado_std"]).tolist()))
if not show_empty and EMPTY_LABEL in options_ex and EMPTY_LABEL not in suggest_exclude:
    suggest_exclude.append(EMPTY_LABEL)
default_ex = [x for x in suggest_exclude if x in options_ex]

exclude_list = exclude_placeholder.multiselect(
    "Excluir resultados (aplicado nos gráficos e tabelas)",
    options=options_ex,
    default=default_ex,
    key="select_exclude"
)

base    = raw[mask_date].copy()
df_plot = base[~safe_series_strings(base["resultado_std"]).isin(exclude_list)] if exclude_list else base

# Badge do período
def _format_period(sel_anos_list, sel_meses_nums):
    anos_all = sorted(raw["ano"].dropna().astype(int).unique().tolist())
    meses_all = sorted([m for m in raw["mes_num"].dropna().unique().tolist() if 1 <= m <= 12])
    anos = sel_anos_list if sel_anos_list else anos_all
    meses = sel_meses_nums if sel_meses_nums else meses_all
    meses_lbl = ", ".join(MESES_PT.get(m, str(m)) for m in meses)
    anos_lbl = ", ".join(str(a) for a in anos)
    if meses_lbl and anos_lbl:
        return f"{meses_lbl} de {anos_lbl}"
    return anos_lbl or meses_lbl or "todos os períodos"

periodo_lbl = _format_period(st.session_state.get("f_anos", []), [MESES_MAP.get(x, x) if isinstance(x, str) else x for x in st.session_state.get("f_meses", [])])
st.markdown(
    f"""
    <div style="padding:.5rem .75rem;border-radius:8px;
               background:#F1F5F9;display:inline-block;
               font-size:0.9rem;">
      <b>Período ativo:</b> {periodo_lbl} • <b>Registros:</b> {len(base)}
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# GRÁFICOS – ORDEM
# =========================

st.header("📈 Resumos & Gráficos")

def apply_yaxis(fig, y_max_current: int):
    y_max = y_manual_val if y_manual_on else round_up_multiple(max(1, y_max_current), max(1, y_multiple))
    fig.update_layout(yaxis=dict(range=[0, y_max]))
    return fig

# Paletas consistentes (c-maps)
cmap_resultado = color_map_for_series(safe_series_strings(df_plot["resultado_std"]), "cmap_resultado_std")
cmap_setor     = color_map_for_series(df_plot["setor"], "cmap_setor")

# 1) TOP RESULTADOS (sem negativos/contaminantes)
present_vals_base = safe_series_strings(base["resultado_std"])
auto_exclude_top = sorted({v for v in present_vals_base if any(k in normalize_text(v) for k in ["negativ", "contamin"])})
top_df = base[~safe_series_strings(base["resultado_std"]).isin(auto_exclude_top)].copy()

vals_top = safe_series_strings(top_df["resultado_std"])
if not show_empty:
    vals_top = vals_top[vals_top != EMPTY_LABEL]
counts_top = vals_top.value_counts().reset_index()
counts_top.columns = ["resultado_padronizado", "n"]
counts_top_f = counts_top[counts_top["n"] >= min_count]

col1, col2 = st.columns([1,2])
with col1:
    st.caption("OBS: Categorias negativo e contaminante NÃO são contabilizadas")
    st.subheader("Micro-organismos Prevalentes")
    st.dataframe(counts_top_f.head(top_n), use_container_width=True)
with col2:
    try:
        import plotly.express as px
        # aplicar paleta dos resultados
        fig = px.bar(
            counts_top_f.head(top_n),
            x="resultado_padronizado", y="n",
            title=f"{top_n} Micro-organismos Prevalentes",
            color="resultado_padronizado",
            color_discrete_map=cmap_resultado
        )
        fig.update_layout(showlegend=False)
        n_groups, n_bars_per_group = 1, max(1, counts_top_f.head(top_n).shape[0])
        bargap, bargroupgap = dynamic_gaps(n_groups, n_bars_per_group)
        fig.update_layout(bargap=bargap, bargroupgap=bargroupgap)
        y_max_cur = int(counts_top_f["n"].max() or 0)
        fig = apply_yaxis(fig, y_max_cur)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    except Exception:
        st.info("Instale plotly para ver os gráficos (pip install plotly)")

# 2) POR SETOR – BARRAS
st.subheader("Gráfico de Barras por Setor")
if df_plot.empty:
    st.info("Nenhum dado disponível para os setores/período/exclusões selecionados.")
else:
    setores_opts_barras = sorted(df_plot["setor"].dropna().unique().tolist())
    setores_ui = ["(Todos)"] + setores_opts_barras
    if "barras_setor_focus" not in st.session_state:
        st.session_state["barras_setor_focus"] = ["(Todos)"]
    prev_opts = st.session_state.get("barras_setor_focus__opts")
    if prev_opts != setores_opts_barras:
        st.session_state["barras_setor_focus__opts"] = setores_opts_barras
        st.session_state["barras_setor_focus"] = ["(Todos)"]
    sel_focus = st.multiselect("Selecionar setor(es)", options=setores_ui, key="barras_setor_focus")
    if "(Todos)" in sel_focus and len(sel_focus) > 1:
        sel_focus = [x for x in sel_focus if x != "(Todos)"]
        st.session_state["barras_setor_focus"] = sel_focus

    if "(Todos)" in st.session_state["barras_setor_focus"] or len(st.session_state["barras_setor_focus"]) == 0:
        df_barras = df_plot.copy()
        color_kw = {"color": "setor", "color_discrete_map": cmap_setor}
    else:
        df_barras = df_plot[df_plot["setor"].isin(st.session_state["barras_setor_focus"])]
        color_kw = {"color": "resultado_std", "color_discrete_map": cmap_resultado}

    grp = df_barras.copy()
    grp["resultado_std"] = safe_series_strings(grp["resultado_std"])
    if not show_empty:
        grp = grp[grp["resultado_std"] != EMPTY_LABEL]
    grp = grp.groupby(["setor", "resultado_std"]).size().reset_index(name="n").sort_values("n", ascending=False)

    try:
        import plotly.express as px
        fig4 = px.bar(grp, x="resultado_std", y="n", barmode="group", **color_kw)
        n_groups = max(1, len(grp["setor"].unique()))
        n_bars_per_group = max(1, grp["resultado_std"].nunique())
        bargap, bargroupgap = dynamic_gaps(n_groups, n_bars_per_group)
        fig4.update_layout(bargap=bargap, bargroupgap=bargroupgap)
        fig4.update_xaxes(categoryorder="total descending")
        y_max_cur = int(grp["n"].max() or 0)
        fig4 = apply_yaxis(fig4, y_max_cur)
        st.plotly_chart(fig4, use_container_width=True, theme="streamlit")
    except Exception:
        st.dataframe(grp)

# 3) PIZZA – Distribuição de Resultados (multi seleção de setor)
st.subheader("Gráfico de Distribuição de Micro-organismos por Setores Resultados")
if df_plot.empty:
    st.info("Sem dados para distribuição de resultados.")
else:
    setores_opts = sorted(df_plot["setor"].dropna().unique().tolist())
    sel_setores_pie_res = st.multiselect(
        "Selecionar setor(es) (opcional).",
        options=["(Todos)"] + setores_opts,
        default=["(Todos)"],
        key="pizza_res_setor_multisel"
    )
    if "(Todos)" in sel_setores_pie_res or len(sel_setores_pie_res) == 0:
        df_pie_res = df_plot
    else:
        df_pie_res = df_plot[df_plot["setor"].isin(sel_setores_pie_res)]
    vals_res = safe_series_strings(df_pie_res["resultado_std"])
    if not show_empty:
        vals_res = vals_res[vals_res != EMPTY_LABEL]
    pie_res = vals_res.value_counts().reset_index()
    pie_res.columns = ["resultado_padronizado", "n"]
    try:
        import plotly.express as px
        fig5 = px.pie(
            pie_res, names="resultado_padronizado", values="n", hole=0.4,
            color="resultado_padronizado", color_discrete_map=cmap_resultado
        )
        fig5.update_traces(textposition="inside", textinfo=("percent+label" if show_pcts else "label+value"))
        st.plotly_chart(fig5, use_container_width=True, theme="streamlit")
    except Exception:
        st.dataframe(pie_res)

# =========================
# NOVOS GRÁFICOS (multi)
# =========================

st.subheader("Gráfico de Distribuição de Micro-organismos por Origem de Amostra")
tipos_amostra = sorted(df_plot["tipo_amostra"].dropna().unique().tolist())
sel_tipos_amostra = st.multiselect(
    "Selecionar origem de amostra (opcional)",
    options=["(Todos)"] + tipos_amostra,
    default=["(Todos)"],
    key="pizza_tipos_amostra_multisel"
)
if "(Todos)" in sel_tipos_amostra or len(sel_tipos_amostra) == 0:
    df_ma = df_plot
else:
    df_ma = df_plot[df_plot["tipo_amostra"].isin(sel_tipos_amostra)]
vals_ma = safe_series_strings(df_ma["tipo_micro"])
if not show_empty:
    vals_ma = vals_ma[vals_ma != EMPTY_LABEL]
pie_micro_amostra = vals_ma.value_counts().reset_index()
pie_micro_amostra.columns = ["tipo_micro", "n"]
if pie_micro_amostra.empty:
    st.caption("Sem dados suficientes para esta combinação.")
else:
    try:
        import plotly.express as px
        fig_ma = px.pie(pie_micro_amostra, names="tipo_micro", values="n", hole=0.4)
        fig_ma.update_traces(textposition="inside", textinfo=("percent+label" if show_pcts else "label+value"))
        st.plotly_chart(fig_ma, use_container_width=True, theme="streamlit")
    except Exception:
        st.dataframe(pie_micro_amostra)

st.subheader("Gráfico de Distribuição de Classe de Micro-organismos por Setor")
setores_opts2 = sorted(df_plot["setor"].dropna().unique().tolist())
sel_setores_pie = st.multiselect(
    "Selecionar setor(es) (opcional)",
    options=["(Todos)"] + setores_opts2,
    default=["(Todos)"],
    key="pizza_setor_multisel"
)
if "(Todos)" in sel_setores_pie or len(sel_setores_pie) == 0:
    df_ms = df_plot
else:
    df_ms = df_plot[df_plot["setor"].isin(sel_setores_pie)]
vals_ms = safe_series_strings(df_ms["tipo_micro"])
if not show_empty:
    vals_ms = vals_ms[vals_ms != EMPTY_LABEL]
pie_micro_setor = vals_ms.value_counts().reset_index()
pie_micro_setor.columns = ["tipo_micro", "n"]
if pie_micro_setor.empty:
    st.caption("Sem dados suficientes para esta combinação.")
else:
    try:
        import plotly.express as px
        fig_ms = px.pie(pie_micro_setor, names="tipo_micro", values="n", hole=0.4)
        fig_ms.update_traces(textposition="inside", textinfo=("percent+label" if show_pcts else "label+value"))
        st.plotly_chart(fig_ms, use_container_width=True, theme="streamlit")
    except Exception:
        st.dataframe(pie_micro_setor)

# =========================
# COMPARAÇÃO MENSAL/ANUAL
# =========================
st.header("🗓️ Comparação Mensal/Anual")

if df_plot.empty:
    st.info("Sem dados após os filtros atuais para comparar por mês/ano.")
else:
    cmp = df_plot.copy()
    if "ano" not in cmp.columns:
        cmp["ano"] = cmp["data"].dt.year
    if "mes_num" not in cmp.columns:
        cmp["mes_num"] = cmp["data"].dt.month
    cmp = cmp.dropna(subset=["ano","mes_num"])

    cmp["mes_rot"] = cmp["mes_num"].map(MESES_PT)
    cmp["mes_ano"] = cmp["mes_rot"].astype(str) + "/" + cmp["ano"].astype(int).astype(str)

    colA, colB, colC = st.columns([1,1,1])

    setores_opts_cmp = sorted(cmp["setor"].dropna().unique().tolist())
    setores_ui_cmp = ["(Todos)"] + setores_opts_cmp
    if "cmp_setores" not in st.session_state:
        st.session_state["cmp_setores"] = ["(Todos)"]
    prev_opts_cmp_set = st.session_state.get("cmp_setores__opts")
    if prev_opts_cmp_set != setores_opts_cmp:
        st.session_state["cmp_setores__opts"] = setores_opts_cmp
        st.session_state["cmp_setores"] = ["(Todos)"]
    with colA:
        sel_setores_cmp = st.multiselect("Setor(es)", options=setores_ui_cmp, key="cmp_setores")
        if "(Todos)" in sel_setores_cmp and len(sel_setores_cmp) > 1:
            sel_setores_cmp = [x for x in sel_setores_cmp if x != "(Todos)"]
            st.session_state["cmp_setores"] = sel_setores_cmp

    org_opts_cmp = sorted(cmp["resultado_std"].dropna().unique().tolist())
    org_ui_cmp = ["(Todos)"] + org_opts_cmp
    if "cmp_orgs" not in st.session_state:
        st.session_state["cmp_orgs"] = ["(Todos)"]
    prev_opts_cmp_org = st.session_state.get("cmp_orgs__opts")
    if prev_opts_cmp_org != org_opts_cmp:
        st.session_state["cmp_orgs__opts"] = org_opts_cmp
        st.session_state["cmp_orgs"] = ["(Todos)"]
    with colB:
        sel_org_cmp = st.multiselect("Micro-organismo(s)", options=org_ui_cmp, key="cmp_orgs")
        if "(Todos)" in sel_org_cmp and len(sel_org_cmp) > 1:
            sel_org_cmp = [x for x in sel_org_cmp if x != "(Todos)"]
            st.session_state["cmp_orgs"] = sel_org_cmp

    with colC:
        agrupar_por = st.radio("Agrupar por", ["micro-organismo", "setor"], index=0, horizontal=True, key="cmp_groupby")
        barmode_stack = st.toggle("Empilhar (stack)", value=False, key="cmp_stack")

    if "(Todos)" not in st.session_state["cmp_setores"] and len(st.session_state["cmp_setores"]) > 0:
        cmp = cmp[cmp["setor"].isin(st.session_state["cmp_setores"])]
    if "(Todos)" not in st.session_state["cmp_orgs"] and len(st.session_state["cmp_orgs"]) > 0:
        cmp = cmp[cmp["resultado_std"].isin(st.session_state["cmp_orgs"])]

    if cmp.empty:
        st.caption("Sem dados após aplicar os filtros deste card.")
    else:
        if agrupar_por == "micro-organismo":
            grp = cmp.groupby(["ano","mes_num","mes_ano","resultado_std"]).size().reset_index(name="n")
            pivot = grp.pivot_table(index=["ano","mes_num","mes_ano"], columns="resultado_std", values="n", aggfunc="sum", fill_value=0)
            color_field = "resultado_std"
            c_map = cmap_resultado
        else:
            grp = cmp.groupby(["ano","mes_num","mes_ano","setor"]).size().reset_index(name="n")
            pivot = grp.pivot_table(index=["ano","mes_num","mes_ano"], columns="setor", values="n", aggfunc="sum", fill_value=0)
            color_field = "setor"
            c_map = cmap_setor

        pivot = pivot.sort_values(by=["ano","mes_num"])
        plot_df = pivot.reset_index()

        try:
            import plotly.express as px
            long_df = plot_df.melt(id_vars=["ano","mes_num","mes_ano"], var_name=color_field, value_name="n")
            long_df = long_df[long_df["n"] > 0]
            if long_df.empty:
                st.caption("Sem contagens > 0 para plotar.")
            else:
                fig_cmp = px.bar(
                    long_df,
                    x="mes_ano", y="n", color=color_field,
                    color_discrete_map=c_map,
                    barmode=("stack" if barmode_stack else "group"),
                    title=f"Comparação mensal — agrupado por {color_field}"
                )
                y_max_cur = int(long_df["n"].max() or 0)
                fig_cmp = apply_yaxis(fig_cmp, y_max_cur)
                fig_cmp.update_xaxes(categoryorder="array", categoryarray=plot_df["mes_ano"].tolist())
                st.plotly_chart(fig_cmp, use_container_width=True, theme="streamlit")
        except Exception:
            st.dataframe(plot_df)

        st.caption("Tabela (linhas: mês/ano; colunas: categorias selecionadas)")
        st.dataframe(pivot, use_container_width=True)

# =========================
# NOVA SEÇÃO — Rank de mudança mês a mês por micro-organismo
# =========================
st.subheader("Rank de Mudança Mensal por Micro-organismo (Δ último vs anterior)")
try:
    # usa df_plot (com filtros/exclusões) para refletir a visão atual
    tmp = df_plot.copy()
    tmp = tmp.dropna(subset=["ano","mes_num"])
    if tmp.empty:
        st.caption("Sem dados suficientes após filtros.")
    else:
        tmp["mes_ano_key"] = tmp["ano"].astype(int).astype(str) + "-" + tmp["mes_num"].astype(int).astype(str).str.zfill(2)
        # ordena cronologicamente
        ordered_keys = sorted(tmp["mes_ano_key"].unique().tolist())
        if len(ordered_keys) < 2:
            st.caption("É necessário pelo menos 2 meses para calcular o Δ.")
        else:
            last_key = ordered_keys[-1]
            prev_key = ordered_keys[-2]
            g = tmp.groupby(["mes_ano_key","resultado_std"]).size().reset_index(name="n")
            cur = g[g["mes_ano_key"] == last_key].set_index("resultado_std")["n"]
            prv = g[g["mes_ano_key"] == prev_key].set_index("resultado_std")["n"]
            # garante presença de todas as categorias
            all_res = sorted(set(cur.index).union(set(prv.index)))
            cur = cur.reindex(all_res, fill_value=0)
            prv = prv.reindex(all_res, fill_value=0)
            delta = cur - prv
            arrow = delta.apply(lambda x: "▲" if x>0 else ("▼" if x<0 else "➖"))
            df_rank = pd.DataFrame({
                "resultado_padronizado": all_res,
                f"{prev_key}": prv.values,
                f"{last_key}": cur.values,
                "Δ": delta.values,
                "tendência": arrow.values
            }).sort_values(["Δ", f"{last_key}"], ascending=[False, False])
            st.dataframe(df_rank.reset_index(drop=True), use_container_width=True)
except Exception as e:
    st.caption(f"Não foi possível gerar o rank de mudança: {e}")

# =========================
# NOVA SEÇÃO — Heatmap mês × setor
# =========================
st.subheader("Heatmap Mês × Setor")
try:
    hm = df_plot.copy()
    hm = hm.dropna(subset=["ano","mes_num"])
    if hm.empty:
        st.caption("Sem dados suficientes após filtros.")
    else:
        hm["mes_rot"] = hm["mes_num"].map(MESES_PT)
        hm["mes_ano"] = hm["mes_rot"].astype(str) + "/" + hm["ano"].astype(int).astype(str)
        table = hm.groupby(["mes_ano","setor"]).size().reset_index(name="n")
        # pivot com meses como linhas em ordem cronológica
        # ordenação cronológica:
        order_df = hm[["mes_ano","ano","mes_num"]].drop_duplicates().sort_values(["ano","mes_num"])
        cat_order = order_df["mes_ano"].tolist()
        pv = table.pivot_table(index="mes_ano", columns="setor", values="n", aggfunc="sum", fill_value=0)
        pv = pv.reindex(cat_order)
        try:
            import plotly.express as px
            fig_hm = px.imshow(
                pv,
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(color="Contagem"),
            )
            fig_hm.update_layout(margin=dict(l=40,r=20,t=30,b=40))
            st.plotly_chart(fig_hm, use_container_width=True, theme="streamlit")
        except Exception:
            st.dataframe(pv, use_container_width=True)
except Exception as e:
    st.caption(f"Não foi possível gerar o heatmap: {e}")

# =========================
# CARDS – Tabelas com filtros por coluna
# =========================

def df_with_column_filters(df: pd.DataFrame, label: str, cols_filter: List[str], key_prefix: str) -> pd.DataFrame:
    with st.expander(label, expanded=True):
        filters = {}
        for c in cols_filter:
            options = sorted(df[c].dropna().unique().tolist())
            selected = st.multiselect(
                f"Filtrar {c}",
                options=options,
                default=options,
                key=f"{key_prefix}_{c}"
            )
            filters[c] = selected
        mask = pd.Series([True]*len(df), index=df.index)
        for c, selected in filters.items():
            mask &= df[c].isin(selected)
        st.dataframe(df[mask], use_container_width=True)
        return df[mask]

st.subheader("Distribuições por Segmento")
seg1 = base.copy()
seg1["resultado_std"] = safe_series_strings(seg1["resultado_std"])
if not show_empty:
    seg1 = seg1[seg1["resultado_std"] != EMPTY_LABEL]
seg1 = seg1.groupby(["setor","resultado_std"]).size().reset_index(name="n").sort_values("n", ascending=False)
_ = df_with_column_filters(seg1, "Tabela: Setor × Resultado", ["setor","resultado_std"], key_prefix="seg1")

st.subheader("Distribuições por Origem da Amostra")
seg2 = base.copy()
seg2["resultado_std"] = safe_series_strings(seg2["resultado_std"])
if not show_empty:
    seg2 = seg2[seg2["resultado_std"] != EMPTY_LABEL]
seg2 = seg2.groupby(["tipo_amostra","resultado_std"]).size().reset_index(name="n").sort_values("n", ascending=False)
_ = df_with_column_filters(seg2, "Tabela: Tipo de amostra × Resultado (com filtros por coluna)", ["tipo_amostra","resultado_std"], key_prefix="seg2")

st.subheader("Distribuições por Classe de Micro-organismo")
if base["tipo_micro"].replace("", np.nan).notna().any():
    seg3 = base.copy()
    seg3["resultado_std"] = safe_series_strings(seg3["resultado_std"])
    if not show_empty:
        seg3 = seg3[seg3["resultado_std"] != EMPTY_LABEL]
    seg3 = seg3.groupby(["tipo_micro","resultado_std"]).size().reset_index(name="n").sort_values("n", ascending=False)
    _ = df_with_column_filters(seg3, "Tabela: Tipo de micro-organismo × Resultado", ["tipo_micro","resultado_std"], key_prefix="seg3")
else:
    st.caption("Sem dados de 'tipo_micro' suficientes para esta distribuição.")

# =========================
# Exportações
# =========================

st.header("📦 Exportações")
ordem = ["data","setor","tipo_amostra","resultado_raw","resultado_std","tipo_micro"]
export_base = df_plot if not df_plot.empty else base
export_df = export_base[ordem]

csv_buf = io.StringIO()
export_df.to_csv(csv_buf, index=False)
st.download_button(
    "⬇️ Baixar planilha padronizada (CSV) — dados filtrados",
    data=csv_buf.getvalue(),
    file_name=f"HMA_padronizado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv",
)

# Template de tradução (bruto)
if not unmatched.empty:
    tmp = unmatched[["resultado"]].copy()
    tmp.insert(1, "padronizado", "")
    tmp.insert(2, "tipo_micro", "")
    st.download_button(
        "⬇️ Baixar template de tradução (A:resultado, B:padronizado, C:tipo)",
        data=tmp.to_csv(index=False),
        file_name=f"template_traducao_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
)

st.caption("Planilha bruta **sem cabeçalho** (pulando 4 linhas iniciais). Índices fixos: B=Data, D=Setor, E=Tipo de amostra, F=Resultado.")
