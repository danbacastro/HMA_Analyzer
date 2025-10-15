import os, io, re, math, base64, unicodedata
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import dash_auth

# =========================
# ---- Config / Auth ----
# =========================
AUTH_USERNAME = "hma-scih"
PASSWORD = os.environ.get("PASSWORD_HMA", "")
if not PASSWORD:
    raise SystemExit("Defina a variável de ambiente PASSWORD_HMA.")

# Tema bonito
external_stylesheets = [dbc.themes.FLATLY, dbc.icons.BOOTSTRAP]
app: Dash = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    title="SCIH HMA — Dashboard"
)
server = app.server
auth = dash_auth.BasicAuth(server, {AUTH_USERNAME: PASSWORD})

# =========================
# ---- Helpers (do seu app) ----
# =========================
EMPTY_LABEL = "(sem informação)"
MESES_MAP = {
    "janeiro":1, "fevereiro":2, "marco":3, "março":3, "abril":4, "maio":5, "junho":6,
    "julho":7, "agosto":8, "setembro":9, "outubro":10, "novembro":11, "dezembro":12
}
MESES_PT = {
    1:"janeiro",2:"fevereiro",3:"março",4:"abril",5:"maio",6:"junho",
    7:"julho",8:"agosto",9:"setembro",10:"outubro",11:"novembro",12:"dezembro"
}

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

def infer_year_from_filename(name: str) -> Optional[int]:
    if not name:
        return None
    m = re.search(r'((?:19|20)\d{2})', name)
    return int(m.group(1)) if m else None

def parse_pt_dates(series: pd.Series, default_year: Optional[int]) -> pd.DataFrame:
    s = series.astype(str).fillna("").str.lower().map(_strip_accents)
    m  = s.str.extract(r'(?P<dia>\d{1,2})\s*(?P<mes>janeiro|fevereiro|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)')
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
    return pd.DataFrame({"dia_pars": dia, "mes_pars": mes, "ano_pars": ano})

def safe_series_strings(s: pd.Series, empty_label=EMPTY_LABEL) -> pd.Series:
    out = s.astype(str).replace(["nan","NaN","None","NONE"], "").str.strip()
    return out.mask(out.eq(""), other=empty_label)

def decode_contents(contents: str, filename: str) -> pd.DataFrame:
    """
    Lê CSV/XLSX enviados pelo dcc.Upload (base64).
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(decoded), header=None, skiprows=2, engine="openpyxl")
    return pd.read_csv(io.BytesIO(decoded), header=None, skiprows=2)

def read_translation_df(contents: str, filename: str) -> pd.DataFrame:
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.BytesIO(decoded))

def build_export(data_df: pd.DataFrame, include_empty: bool) -> pd.DataFrame:
    out = data_df[["data","setor","tipo_amostra","resultado_raw","resultado_std","tipo_micro"]].copy()
    return out if include_empty else out[out["resultado_std"] != EMPTY_LABEL]

def gen_download(df: pd.DataFrame, name: str) -> dict:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return dict(content=buf.getvalue(), filename=name, type="text/csv")

def zscore_alerts(df_plot: pd.DataFrame, z_thr: float, min_hist: int) -> pd.DataFrame:
    if df_plot.empty:
        return pd.DataFrame()
    anom = df_plot.copy()
    if "ano" not in anom.columns: anom["ano"] = anom["data"].dt.year
    if "mes_num" not in anom.columns: anom["mes_num"] = anom["data"].dt.month
    anom = anom.dropna(subset=["ano","mes_num"])
    if anom.empty:
        return pd.DataFrame()
    anom["mkey"] = anom["ano"].astype(int).astype(str) + "-" + anom["mes_num"].astype(int).astype(str).zfill(2)
    order_m = anom[["mkey","ano","mes_num"]].drop_duplicates().sort_values(["ano","mes_num"])
    mkeys = order_m["mkey"].tolist()
    if len(mkeys) < 2:
        return pd.DataFrame()
    last_k = mkeys[-1]
    g = anom.groupby(["resultado_std","mkey"]).size().reset_index(name="n")
    hist = g[g["mkey"] != last_k].groupby("resultado_std")["n"].agg(["mean","std","count"]).reset_index()
    hist = hist.rename(columns={"mean":"media_hist","std":"std_hist","count":"num_meses_hist"})
    cur  = g[g["mkey"] == last_k][["resultado_std","n"]].rename(columns={"n":"n_cur"})
    alerts = pd.merge(cur, hist, on="resultado_std", how="left").fillna({"media_hist":0.0,"std_hist":0.0,"num_meses_hist":0})
    alerts["z"] = (alerts["n_cur"] - alerts["media_hist"]) / alerts["std_hist"].replace(0, np.nan)
    alerts["z"] = alerts["z"].fillna(0.0)
    alerts = alerts[(alerts["num_meses_hist"] >= int(min_hist)) & (alerts["z"] >= float(z_thr))].sort_values("z", ascending=False)
    return alerts

# =========================
# ---- Layout ----
# =========================
header = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.Span("🧫", style={"fontSize":"1.5rem", "marginRight":".4rem"}),
            dbc.NavbarBrand("SCIH HMA — Analytics & Report", className="fw-bold")
        ]),
        dbc.Badge("Dash • Plotly • Bootstrap", color="primary", className="ms-auto")
    ]),
    color="white", className="shadow-sm"
)

upload_card = dbc.Card([
    dbc.CardHeader("Upload das Planilhas"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Planilha Bruta (CSV/XLSX)", className="fw-semibold"),
                dcc.Upload(
                    id="up-raw",
                    children=html.Div(["Arraste e solte, ou ", html.A("selecione o arquivo")]),
                    multiple=False,
                    className="border rounded p-3 text-center",
                    accept=".csv,.xlsx,.xls",
                ),
                html.Div(id="up-raw-name", className="text-muted small mt-1")
            ], md=6),
            dbc.Col([
                html.Label("Planilha de Tradução (CSV, opcional)", className="fw-semibold"),
                dcc.Upload(
                    id="up-trans",
                    children=html.Div(["Arraste e solte, ou ", html.A("selecione o arquivo")]),
                    multiple=False,
                    className="border rounded p-3 text-center",
                    accept=".csv",
                ),
                html.Div(id="up-trans-name", className="text-muted small mt-1")
            ], md=6),
        ], className="g-3"),
        html.Hr(),
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Top N"),
                dbc.Input(id="top-n", type="number", min=5, max=50, step=1, value=15)
            ]), md=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Mín. ocorrências"),
                dbc.Input(id="min-count", type="number", min=1, max=1000, step=1, value=1)
            ]), md=3),
            dbc.Col(dbc.Checklist(
                options=[{"label":" Incluir (sem informação)", "value":"include"}],
                value=["include"],
                inline=True, id="include-empty"
            ), md=3),
            dbc.Col(dbc.Checklist(
                options=[{"label":" Mostrar % nas pizzas", "value":"pcts"}],
                value=["pcts"],
                inline=True, id="show-pcts"
            ), md=3),
        ], className="g-3"),
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Z-thr alertas"),
                dbc.Input(id="z-thr", type="number", step=0.5, value=2.0)
            ]), md=3),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Histórico mínimo (meses)"),
                dbc.Input(id="min-hist", type="number", min=2, value=3)
            ]), md=3),
        ], className="g-3 mt-2"),
        dbc.Button("Processar", id="btn-process", color="primary", className="mt-3", disabled=False)
    ])
], className="mb-3")

tabs = dbc.Tabs([
    dbc.Tab(label="Visão Geral", tab_id="tab-overview"),
    dbc.Tab(label="Não Mapeados", tab_id="tab-unmatched"),
    dbc.Tab(label="Gráficos", tab_id="tab-charts"),
    dbc.Tab(label="Alertas", tab_id="tab-alerts"),
    dbc.Tab(label="Exportações", tab_id="tab-exports"),
], id="main-tabs", active_tab="tab-overview", className="mb-3")

content = html.Div(id="tab-content")

stores = html.Div([
    dcc.Store(id="store-raw"),        # DataFrame processado (named)
    dcc.Store(id="store-export"),     # DataFrame export (filtrado include_empty)
    dcc.Store(id="store-unmatched"),  # DF não mapeados + sugestões
    dcc.Download(id="dl-padronizado"),
    dcc.Download(id="dl-template"),
    dcc.Download(id="dl-trad-base"),
])

app.layout = dbc.Container([
    header,
    html.Div(className="my-3"),
    upload_card,
    tabs,
    content,
    stores
], fluid=True)

# =========================
# ---- Callbacks ----
# =========================

@app.callback(
    Output("up-raw-name", "children"),
    Input("up-raw", "filename")
)
def show_raw_name(name):
    return f"Arquivo: {name}" if name else ""

@app.callback(
    Output("up-trans-name", "children"),
    Input("up-trans", "filename")
)
def show_trans_name(name):
    return f"Arquivo: {name}" if name else "—"

@app.callback(
    Output("store-raw", "data"),
    Output("store-export", "data"),
    Output("store-unmatched", "data"),
    Output("main-tabs", "active_tab"),
    Input("btn-process", "n_clicks"),
    State("up-raw", "contents"),
    State("up-raw", "filename"),
    State("up-trans", "contents"),
    State("up-trans", "filename"),
    State("include-empty", "value"),
    prevent_initial_call=True
)
def process(nc, raw_contents, raw_name, trans_contents, trans_name, include_empty_vals):
    if not raw_contents or not raw_name:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # ---------- leitura bruta ----------
    raw_df = decode_contents(raw_contents, raw_name)

    # mapear colunas fixas
    raw_named = pd.DataFrame({
        "data": pd.to_datetime(raw_df.iloc[:, 1], errors="coerce", dayfirst=True, infer_datetime_format=True),
        "setor": raw_df.iloc[:, 3].astype(str),
        "tipo_amostra": raw_df.iloc[:, 4].astype(str),
        "resultado_raw": raw_df.iloc[:, 5].astype(str),
    })

    default_year = infer_year_from_filename(raw_name or "")
    parsed = parse_pt_dates(raw_df.iloc[:,1], default_year)
    raw_named["ano"]     = raw_named["data"].dt.year.fillna(parsed["ano_pars"])
    raw_named["mes_num"] = raw_named["data"].dt.month.fillna(parsed["mes_pars"])

    need_fill = raw_named["data"].isna()
    if need_fill.any():
        yr_fallback = default_year if default_year is not None else pd.Timestamp.today().year
        ano_fill = raw_named["ano"].astype("Int64").fillna(yr_fallback).astype(int)
        mes_fill = raw_named["mes_num"].astype("Int64").fillna(1).astype(int)
        dia_fill = parsed["dia_pars"].astype("Int64").fillna(1).astype(int)
        parts = pd.DataFrame({"year": ano_fill, "month": mes_fill, "day": dia_fill})
        synth = pd.to_datetime(parts, errors="coerce")
        raw_named.loc[need_fill, "data"] = synth[need_fill]

    raw_named["mes"] = raw_named["mes_num"].map(MESES_PT)

    # ---------- tradução ----------
    map_pad, map_tip, trans_tbl = {}, {}, pd.DataFrame()
    if trans_contents and trans_name:
        tdf = read_translation_df(trans_contents, trans_name)

        def pick(colnames: List[str], default_idx: int):
            for cand in colnames:
                for c in tdf.columns:
                    if cand == c or normalize_text(cand) == normalize_text(c):
                        return c
            return tdf.columns[min(default_idx, tdf.shape[1]-1)]

        c_res = pick(["resultado","original","res","termo","from"], 0)
        c_pad = pick(["padronização","padronizacao","padroniza","correto","to"], 1)
        c_tip = pick(["tipo do micro-organismo","tipo_micro","tipo","classe"], 2 if tdf.shape[1] >= 3 else 1)

        trans_tbl = tdf[[c_res, c_pad]].copy()
        trans_tbl.columns = ["resultado","padronizado"]
        trans_tbl["tipo_micro"] = tdf[c_tip] if c_tip in tdf.columns else ""
        trans_tbl = trans_tbl.dropna(subset=["resultado","padronizado"])
        trans_tbl["resultado_norm"] = trans_tbl["resultado"].map(normalize_text)
        trans_tbl = trans_tbl.drop_duplicates(subset=["resultado_norm"], keep="last")

        map_pad = dict(zip(trans_tbl["resultado_norm"], trans_tbl["padronizado"].astype(str)))
        map_tip = dict(zip(trans_tbl["resultado_norm"], trans_tbl["tipo_micro"].astype(str)))

    res_norm = raw_named["resultado_raw"].map(normalize_text)
    std_series = res_norm.map(map_pad)
    raw_named["resultado_std"] = std_series.where(std_series.notna() & (std_series.astype(str).str.strip() != ""), other=EMPTY_LABEL)
    raw_named["tipo_micro"] = res_norm.map(map_tip).fillna("")

    # ---------- não mapeados + sugestões ----------
    try:
        from rapidfuzz import process as rf_process, fuzz as rf_fuzz
        HAVE_RAPIDFUZZ = True
    except Exception:
        HAVE_RAPIDFUZZ = False

    unmatched = (
        pd.DataFrame({"resultado": raw_named["resultado_raw"], "resultado_norm": res_norm})
        [~res_norm.isin(set(map_pad.keys()))]
        .drop_duplicates(subset=["resultado_norm"])
        .reset_index(drop=True)
    )
    unmatched["sugestoes"] = ""
    if HAVE_RAPIDFUZZ and not unmatched.empty and map_pad:
        targets = sorted(set(map_pad.values()))
        def suggest(term):
            res = rf_process.extract(term, targets, scorer=rf_fuzz.token_sort_ratio, score_cutoff=86, limit=3)
            return "; ".join(f"{cand} ({score})" for cand, score, _ in res)
        unmatched["sugestoes"] = unmatched["resultado_norm"].astype(str).map(suggest)

    include_empty = "include" in (include_empty_vals or [])
    export_df = build_export(raw_named, include_empty)

    # Guardar tudo em JSON "records" (para dcc.Store)
    return (
        raw_named.to_dict("records"),
        export_df.to_dict("records"),
        unmatched.to_dict("records"),
        "tab-overview"
    )

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    State("store-raw", "data"),
    State("store-export", "data"),
    State("store-unmatched", "data"),
    State("top-n", "value"),
    State("min-count", "value"),
    State("show-pcts", "value"),
    State("z-thr", "value"),
    State("min-hist", "value"),
)
def render_tab(active, raw_data, export_data, unmatched_data, top_n, min_count, show_pcts_vals, z_thr, min_hist):
    if not raw_data:
        return dbc.Alert("Carregue e processe os arquivos para visualizar o dashboard.", color="secondary")

    raw = pd.DataFrame(raw_data)
    df_plot = pd.DataFrame(export_data)
    unmatched = pd.DataFrame(unmatched_data) if unmatched_data else pd.DataFrame(columns=["resultado","resultado_norm","sugestoes"])

    # badge de período
    anos = sorted(pd.to_numeric(raw["ano"], errors="coerce").dropna().astype(int).unique().tolist())
    meses = sorted(pd.to_numeric(raw["mes_num"], errors="coerce").dropna().astype(int).unique().tolist())
    meses_lbl = ", ".join(MESES_PT[m] for m in meses) if meses else "todos os meses"
    anos_lbl  = ", ".join(map(str, anos)) if anos else "todos os anos"
    pill = dbc.Badge([html.B("Período: "), f"{meses_lbl} de {anos_lbl}", " • ", html.B("Registros: "), str(len(raw))],
                     color="light", text_color="dark", className="mb-2")

    if active == "tab-overview":
        prev_raw = raw.head(20).to_dict("records")
        cols1 = [{"name": c, "id": c} for c in raw.columns]
        tbl1 = dbc.Card([
            dbc.CardHeader("Prévia — Dados Nomeados (20 linhas)"),
            dbc.CardBody(dbc.Table.from_dataframe(raw.head(20), striped=True, bordered=True, hover=True, size="sm"))
        ])
        return html.Div([pill, tbl1])

    elif active == "tab-unmatched":
        table = (dbc.Table.from_dataframe(unmatched, striped=True, bordered=True, hover=True, size="sm")
                 if not unmatched.empty else html.P("Todos os resultados foram mapeados.", className="text-success"))
        tips = dbc.Alert("Baixe o template na aba Exportações, preencha os mapeamentos faltantes e reenvie.", color="info")
        return html.Div([pill, table, tips])

    elif active == "tab-charts":
        show_pcts = "pcts" in (show_pcts_vals or [])

        # Top N (ignorando “negativ/contamin”)
        present_vals_base = safe_series_strings(raw["resultado_std"])
        auto_exclude_top = sorted({v for v in present_vals_base if any(k in normalize_text(v) for k in ["negativ","contamin"])})
        top_df = raw[~safe_series_strings(raw["resultado_std"]).isin(auto_exclude_top)].copy()
        vals_top = safe_series_strings(top_df["resultado_std"])
        if df_plot is not None and not df_plot.empty:
            # respeita include_empty via df_plot
            pass
        else:
            vals_top = vals_top[vals_top != EMPTY_LABEL]
        counts_top = vals_top.value_counts().reset_index()
        counts_top.columns = ["resultado_padronizado","n"]
        counts_top_f = counts_top[counts_top["n"] >= int(min_count)].head(int(top_n))
        fig_top = px.bar(counts_top_f, x="resultado_padronizado", y="n", title=f"Top {int(top_n)} Micro-organismos")
        fig_top.update_layout(margin=dict(l=20,r=10,t=40,b=40))

        # Barras por setor
        grp = df_plot.copy()
        grp["resultado_std"] = safe_series_strings(grp["resultado_std"])
        grp = grp.groupby(["setor","resultado_std"]).size().reset_index(name="n").sort_values("n", ascending=False)
        fig_barras = px.bar(grp, x="resultado_std", y="n", color="setor", barmode="group", title="Distribuição por Setor")
        fig_barras.update_xaxes(categoryorder="total descending")

        # Pizza por resultado
        pie_res = safe_series_strings(df_plot["resultado_std"])
        pie_res = pie_res.value_counts().reset_index()
        pie_res.columns = ["resultado_padronizado","n"]
        fig_pie_res = px.pie(pie_res, names="resultado_padronizado", values="n", hole=0.4, title="Distribuição de Resultados")
        fig_pie_res.update_traces(textposition="inside", textinfo=("percent+label" if show_pcts else "label+value"))

        # Pizza por classe micro
        pie_tm = safe_series_strings(df_plot["tipo_micro"])
        pie_tm = pie_tm.value_counts().reset_index()
        pie_tm.columns = ["tipo_micro","n"]
        if not pie_tm.empty:
            fig_pie_tm = px.pie(pie_tm, names="tipo_micro", values="n", hole=0.4, title="Distribuição por Classe de Micro-organismo")
            fig_pie_tm.update_traces(textposition="inside", textinfo=("percent+label" if show_pcts else "label+value"))
        else:
            fig_pie_tm = go.Figure().add_annotation(text="Sem dados suficientes para pizza por classe.", showarrow=False)

        # Comparação mensal (por micro-organismo)
        cmp = df_plot.copy()
        if "ano" not in cmp.columns: cmp["ano"] = raw["ano"]
        if "mes_num" not in cmp.columns: cmp["mes_num"] = raw["mes_num"]
        cmp = cmp.dropna(subset=["ano","mes_num"])
        if cmp.empty:
            fig_cmp = go.Figure().add_annotation(text="Sem dados para comparação mensal.", showarrow=False)
        else:
            grp_cmp = cmp.groupby(["ano","mes_num","resultado_std"]).size().reset_index(name="n")
            grp_cmp["mes_ano"] = grp_cmp["mes_num"].map(MESES_PT).astype(str) + "/" + grp_cmp["ano"].astype(int).astype(str)
            fig_cmp = px.bar(grp_cmp, x="mes_ano", y="n", color="resultado_std", barmode="group", title="Comparação Mensal — por Micro-organismo")

        # Heatmap mês × setor
        hm = df_plot.copy()
        if "ano" not in hm.columns: hm["ano"] = raw["ano"]
        if "mes_num" not in hm.columns: hm["mes_num"] = raw["mes_num"]
        hm = hm.dropna(subset=["ano","mes_num"])
        if hm.empty:
            fig_hm = go.Figure().add_annotation(text="Sem dados para heatmap.", showarrow=False)
        else:
            hm["mes_ano"] = hm["mes_num"].map(MESES_PT).astype(str) + "/" + hm["ano"].astype(int).astype(str)
            table = hm.groupby(["mes_ano","setor"]).size().reset_index(name="n")
            pv = table.pivot_table(index="mes_ano", columns="setor", values="n", aggfunc="sum", fill_value=0)
            fig_hm = px.imshow(pv, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Contagem"),
                               title="Heatmap Mês × Setor")

        grid = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_top), md=6),
            dbc.Col(dcc.Graph(figure=fig_barras), md=6),
            dbc.Col(dcc.Graph(figure=fig_pie_res), md=6, className="mt-3"),
            dbc.Col(dcc.Graph(figure=fig_pie_tm), md=6, className="mt-3"),
            dbc.Col(dcc.Graph(figure=fig_cmp), md=12, className="mt-3"),
            dbc.Col(dcc.Graph(figure=fig_hm), md=12, className="mt-3"),
        ], className="g-3")
        return html.Div([pill, grid])

    elif active == "tab-alerts":
        alerts = zscore_alerts(df_plot, float(z_thr or 2.0), int(min_hist or 3))
        if alerts.empty:
            return html.Div([pill, dbc.Alert("Nenhuma anomalia detectada com os parâmetros atuais.", color="success")])
        nice = alerts.copy()
        nice["z"] = nice["z"].map(lambda x: f"{x:+.2f}σ")
        nice["media_hist"] = nice["media_hist"].map(lambda x: f"{x:.1f}")
        nice = nice.rename(columns={"resultado_std":"Micro-organismo","n_cur":"Mês atual","media_hist":"Média hist.","z":"z-score"})
        return html.Div([
            pill,
            dbc.Card([
                dbc.CardHeader("Alertas por Tendência Anômala"),
                dbc.CardBody(dbc.Table.from_dataframe(nice, striped=True, bordered=True, hover=True, size="sm"))
            ])
        ])

    elif active == "tab-exports":
        # 3 botões de download
        buttons = dbc.ButtonGroup([
            dbc.Button("⬇️ Padronizado (CSV)", id="btn-dl-pad", color="primary"),
            dbc.Button("⬇️ Template de Tradução (CSV)", id="btn-dl-template", color="secondary"),
            dbc.Button("⬇️ Tradução atual (CSV)", id="btn-dl-trad", color="secondary"),
        ])
        info = dbc.Alert("Dica: baixe o template, preencha mapeamentos e use como planilha de tradução no próximo upload.", color="info", className="mt-3")
        return html.Div([pill, buttons, info])

    return html.Div([pill])

@app.callback(
    Output("dl-padronizado", "data"),
    Input("btn-dl-pad", "n_clicks"),
    State("store-export", "data"),
    prevent_initial_call=True
)
def download_pad(nc, export_data):
    if not export_data:
        return dash.no_update
    df = pd.DataFrame(export_data)
    name = f"HMA_padronizado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return gen_download(df, name)

@app.callback(
    Output("dl-template", "data"),
    Input("btn-dl-template", "n_clicks"),
    State("store-unmatched", "data"),
    prevent_initial_call=True
)
def download_template(nc, unmatched_data):
    if not unmatched_data:
        # sem não mapeados -> template vazio com colunas
        df = pd.DataFrame(columns=["resultado","padronizado","tipo_micro"])
    else:
        df = pd.DataFrame(unmatched_data)[["resultado"]].copy()
        df.insert(1, "padronizado", "")
        df.insert(2, "tipo_micro", "")
    name = f"template_traducao_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return gen_download(df, name)

@app.callback(
    Output("dl-trad-base", "data"),
    Input("btn-dl-trad", "n_clicks"),
    State("store-unmatched", "data"),
    prevent_initial_call=True
)
def download_trad(nc, unmatched_data):
    # Neste MVP, exportamos um "esqueleto/base" (sem merge automático com uma planilha existente)
    df = pd.DataFrame(columns=["resultado","padronizado","tipo_micro"])
    name = f"traducao_atual_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return gen_download(df, name)

# Healthcheck
@app.server.route("/healthz")
def healthz():
    from flask import jsonify
    return jsonify(ok=True, time=datetime.utcnow().isoformat()+"Z")

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.getenv("PORT", "8050")), debug=False)
