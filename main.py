import base64, io, os, re, unicodedata, math
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

# =========================
# AUTH (HTTP Basic)
# =========================
security = HTTPBasic()
AUTH_USERNAME = "hma-scih"

def _get_secret_password() -> str:
    pw = os.environ.get("PASSWORD_HMA", "")
    if not pw:
        raise HTTPException(status_code=500, detail="Defina a variável de ambiente PASSWORD_HMA.")
    return pw

def require_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != AUTH_USERNAME or credentials.password != _get_secret_password():
        raise HTTPException(status_code=401, detail="Credenciais inválidas.")
    return True

# =========================
# Helpers (adaptados do seu app)
# =========================
EMPTY_LABEL = "(sem informação)"
MESES_MAP = {
    "janeiro":1, "fevereiro":2, "marco":3, "março":3, "abril":4, "maio":5, "junho":6,
    "julho":7, "agosto":8, "setembro":9, "outubro":10, "novembro":11, "dezembro":12
}
MESES_PT = {1:"janeiro",2:"fevereiro",3:"março",4:"abril",5:"maio",6:"junho",7:"julho",8:"agosto",9:"setembro",10:"outubro",11:"novembro",12:"dezembro"}

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

def csv_data_uri(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode("utf-8")).decode("utf-8")
    return f'data:text/csv;base64,{b64}', filename

def plot_html(fig):
    # Gera um <div> Plotly pronto (JS via CDN)
    import plotly.io as pio
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

# =========================
# App
# =========================
app = FastAPI(title="HMA Analyzer — Web (sem Streamlit)")

# Página inicial (form simples)
INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HMA Analyzer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
   body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }
   .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
   h1 { margin: 0 0 12px; }
   h2 { margin-top: 24px; }
   .btn { display:inline-block; padding:8px 12px; border-radius:8px; border:1px solid #ddd; background:#f8fafc; text-decoration:none; color:#111827; }
   table { border-collapse: collapse; width: 100%; }
   th, td { border: 1px solid #e5e7eb; padding: 6px 8px; font-size: 14px; }
   th { background: #f1f5f9; text-align: left; }
   .grid { display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));}
   .muted { color:#6b7280; font-size: 13px;}
   .ok { color:#065f46;}
   .warn { color:#92400e;}
   .pill { display:inline-block; padding:2px 8px; background:#eef2ff; color:#3730a3; border-radius:999px; font-size:12px; }
  </style>
</head>
<body>
  <h1>🧫 SCIH HMA — Analytics & Report (Render)</h1>
  <div class="card">
    <form action="/analyze" method="post" enctype="multipart/form-data">
      <p><b>Planilha Bruta</b> (CSV/XLSX) — estrutura igual à usada no Streamlit:<br>
      <input type="file" name="data_file" required></p>
      <p><b>Planilha de Tradução</b> (CSV, opcional):<br>
      <input type="file" name="trans_file"></p>
      <details>
        <summary>⚙️ Opções</summary>
        <p>Top N (prevalência): <input type="number" name="top_n" value="15" min="5" max="50"></p>
        <p>Mínimo de ocorrências (prevalência): <input type="number" name="min_count" value="1" min="1" max="1000"></p>
        <p>Incluir “(sem informação)” nas análises? <input type="checkbox" name="include_empty"></p>
        <p>Mostrar % nas pizzas? <input type="checkbox" name="show_pcts" checked></p>
        <p>Limite z-score (alertas): <input type="number" step="0.5" name="z_thr" value="2.0"></p>
        <p>Mínimo de meses históricos: <input type="number" name="min_hist" value="3" min="2"></p>
      </details>
      <p><button class="btn" type="submit">Processar</button></p>
      <p class="muted">Autenticação: HTTP Basic (usuário <code>hma-scih</code>, senha = variável <code>PASSWORD_HMA</code> no Render).</p>
    </form>
  </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index(_: bool = Depends(require_auth)):
    return HTMLResponse(INDEX_HTML)

@app.post("/analyze", response_class=HTMLResponse)
def analyze(
    _: bool = Depends(require_auth),
    data_file: UploadFile = File(...),
    trans_file: UploadFile = File(None),
    top_n: int = Form(15),
    min_count: int = Form(1),
    include_empty: Optional[bool] = Form(False),
    show_pcts: Optional[bool] = Form(True),
    z_thr: float = Form(2.0),
    min_hist: int = Form(3),
):
    # ========= Leitura =========
    name = (data_file.filename or "").lower()
    if name.endswith((".xlsx", ".xls")):
        raw_df = pd.read_excel(data_file.file, header=None, skiprows=2, engine="openpyxl")
    else:
        raw_df = pd.read_csv(data_file.file, header=None, skiprows=2)

    if raw_df.shape[1] < 6:
        raise HTTPException(status_code=400, detail="Planilha bruta com colunas insuficientes (espera B,D,E,F ao menos).")

    raw = pd.DataFrame({
        "data": pd.to_datetime(raw_df.iloc[:, 1], errors="coerce", dayfirst=True, infer_datetime_format=True),
        "setor": raw_df.iloc[:, 3].astype(str),
        "tipo_amostra": raw_df.iloc[:, 4].astype(str),
        "resultado_raw": raw_df.iloc[:, 5].astype(str),
    })

    # ========= Datas PT-BR fallback =========
    default_year = infer_year_from_filename(data_file.filename or "")
    parsed = parse_pt_dates(raw_df.iloc[:, 1], default_year)
    raw["ano"] = raw["data"].dt.year.fillna(parsed["ano_pars"])
    raw["mes_num"] = raw["data"].dt.month.fillna(parsed["mes_pars"])
    need_fill = raw["data"].isna()
    if need_fill.any():
        yr_fallback = default_year if default_year is not None else pd.Timestamp.today().year
        ano_fill = raw["ano"].astype("Int64").fillna(yr_fallback).astype(int)
        mes_fill = raw["mes_num"].astype("Int64").fillna(1).astype(int)
        dia_fill = parsed["dia_pars"].astype("Int64").fillna(1).astype(int)
        parts = pd.DataFrame({"year": ano_fill, "month": mes_fill, "day": dia_fill})
        synth = pd.to_datetime(parts, errors="coerce")
        raw.loc[need_fill, "data"] = synth[need_fill]
    raw["mes"] = raw["mes_num"].map(MESES_PT)

    # ========= Tradução (opcional) =========
    map_pad, map_tip = {}, {}
    if trans_file is not None:
        tdf = pd.read_csv(trans_file.file)
        def pick(colnames: List[str], default_idx: int):
            for cand in colnames:
                for c in tdf.columns:
                    if cand == c or normalize_text(cand) == normalize_text(c):
                        return c
            return tdf.columns[min(default_idx, tdf.shape[1]-1)]
        c_res = pick(["resultado","original","res","termo","from"], 0)
        c_pad = pick(["padronização","padronizacao","padroniza","correto","to"], 1)
        c_tip = pick(["tipo do micro-organismo","tipo_micro","tipo","classe"], 2 if tdf.shape[1] >= 3 else 1)
        trans_tbl = tdf[[c_res, c_pad]].copy(); trans_tbl.columns = ["resultado","padronizado"]
        trans_tbl["tipo_micro"] = tdf[c_tip] if c_tip in tdf.columns else ""
        trans_tbl = trans_tbl.dropna(subset=["resultado","padronizado"]).reset_index(drop=True)
        trans_tbl["resultado_norm"] = trans_tbl["resultado"].map(normalize_text)
        trans_tbl = trans_tbl.drop_duplicates(subset=["resultado_norm"], keep="last")
        map_pad = dict(zip(trans_tbl["resultado_norm"], trans_tbl["padronizado"].astype(str)))
        map_tip = dict(zip(trans_tbl["resultado_norm"], trans_tbl["tipo_micro"].astype(str)))
    else:
        trans_tbl = pd.DataFrame(columns=["resultado","padronizado","tipo_micro","resultado_norm"])

    # ========= Padronização =========
    res_norm = raw["resultado_raw"].map(normalize_text)
    std_series = res_norm.map(map_pad)
    raw["resultado_std"] = std_series.where(std_series.notna() & (std_series.astype(str).str.strip() != ""), other=EMPTY_LABEL)
    raw["tipo_micro"] = res_norm.map(map_tip).fillna("")

    # ========= Não mapeados + sugestões =========
    try:
        from rapidfuzz import process, fuzz
        HAVE_RAPIDFUZZ = True
    except Exception:
        HAVE_RAPIDFUZZ = False

    unmatched = (
        pd.DataFrame({"resultado": raw["resultado_raw"], "resultado_norm": res_norm})
        [~res_norm.isin(set(map_pad.keys()))]
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
    unmatched_suggested = suggest_matches(unmatched, map_pad)

    # ========= Export base =========
    export_df = raw[["data","setor","tipo_amostra","resultado_raw","resultado_std","tipo_micro"]].copy()
    if not include_empty:
        export_df = export_df[export_df["resultado_std"] != EMPTY_LABEL]
    dl_pad_uri, dl_pad_name = csv_data_uri(export_df, f"HMA_padronizado_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

    # Template de tradução
    template_trad = None
    if not unmatched.empty:
        tmp = unmatched[["resultado"]].copy()
        tmp.insert(1, "padronizado", "")
        tmp.insert(2, "tipo_micro", "")
        template_trad = tmp
    dl_template_uri, dl_template_name = ("", "")
    if template_trad is not None:
        dl_template_uri, dl_template_name = csv_data_uri(template_trad, f"template_traducao_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

    # Tradução atualizada (merge base + novos)
    merged_trad_uri, merged_trad_name = ("","")
    if trans_tbl is not None and not trans_tbl.empty:
        base_trad = trans_tbl[["resultado","padronizado","tipo_micro"]].copy().fillna("")
    else:
        base_trad = pd.DataFrame(columns=["resultado","padronizado","tipo_micro"])
    # (novos serão preenchidos pelo usuário offline; aqui geramos o arquivo “base + novos mapeáveis” apenas se o user editar; mantemos base)
    dl_trad_uri, dl_trad_name = csv_data_uri(base_trad, f"traducao_atual_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

    # ========= Gráficos =========
    import plotly.express as px

    # Top N (ignorando “negativ/contamin”)
    present_vals_base = safe_series_strings(raw["resultado_std"])
    auto_exclude_top = sorted({v for v in present_vals_base if any(k in normalize_text(v) for k in ["negativ", "contamin"])})
    top_df = raw[~safe_series_strings(raw["resultado_std"]).isin(auto_exclude_top)].copy()
    vals_top = safe_series_strings(top_df["resultado_std"])
    if not include_empty:
        vals_top = vals_top[vals_top != EMPTY_LABEL]
    counts_top = vals_top.value_counts().reset_index()
    counts_top.columns = ["resultado_padronizado", "n"]
    counts_top_f = counts_top[counts_top["n"] >= int(min_count)].head(int(top_n))
    fig_top = px.bar(counts_top_f, x="resultado_padronizado", y="n", title=f"Top {int(top_n)} Micro-organismos Prevalentes")
    html_top = plot_html(fig_top)

    # Barras por Setor
    df_plot = export_df.copy()
    grp = df_plot.copy()
    grp["resultado_std"] = safe_series_strings(grp["resultado_std"])
    if not include_empty:
        grp = grp[grp["resultado_std"] != EMPTY_LABEL]
    grp = grp.groupby(["setor", "resultado_std"]).size().reset_index(name="n").sort_values("n", ascending=False)
    fig_barras = px.bar(grp, x="resultado_std", y="n", color="setor", barmode="group", title="Distribuição por Setor")
    html_barras = plot_html(fig_barras)

    # Pizza — por resultado
    pie_res = safe_series_strings(df_plot["resultado_std"])
    if not include_empty:
        pie_res = pie_res[pie_res != EMPTY_LABEL]
    pie_res = pie_res.value_counts().reset_index()
    pie_res.columns = ["resultado_padronizado", "n"]
    fig_pie_res = px.pie(pie_res, names="resultado_padronizado", values="n", hole=0.4, title="Distribuição de Resultados")
    fig_pie_res.update_traces(textposition="inside", textinfo=("percent+label" if show_pcts else "label+value"))
    html_pie_res = plot_html(fig_pie_res)

    # Pizza — por tipo_micro
    pie_tm = safe_series_strings(df_plot["tipo_micro"])
    if not include_empty:
        pie_tm = pie_tm[pie_tm != EMPTY_LABEL]
    pie_tm = pie_tm.value_counts().reset_index()
    pie_tm.columns = ["tipo_micro", "n"]
    if not pie_tm.empty:
        fig_pie_tm = px.pie(pie_tm, names="tipo_micro", values="n", hole=0.4, title="Distribuição por Classe de Micro-organismo")
        fig_pie_tm.update_traces(textposition="inside", textinfo=("percent+label" if show_pcts else "label+value"))
        html_pie_tm = plot_html(fig_pie_tm)
    else:
        html_pie_tm = "<p class='muted'>Sem dados suficientes para pizza por classe.</p>"

    # Comparação mensal (agrupado por resultado_std)
    cmp = df_plot.copy()
    if "ano" not in cmp.columns: cmp["ano"] = raw["ano"]
    if "mes_num" not in cmp.columns: cmp["mes_num"] = raw["mes_num"]
    cmp = cmp.dropna(subset=["ano","mes_num"])
    if cmp.empty:
        html_cmp = "<p class='muted'>Sem dados suficientes para comparação mensal.</p>"
    else:
        grp_cmp = cmp.groupby(["ano","mes_num","resultado_std"]).size().reset_index(name="n")
        grp_cmp["mes_ano"] = grp_cmp["mes_num"].map(MESES_PT).astype(str) + "/" + grp_cmp["ano"].astype(int).astype(str)
        fig_cmp = px.bar(grp_cmp, x="mes_ano", y="n", color="resultado_std", barmode="group", title="Comparação mensal — por micro-organismo")
        html_cmp = plot_html(fig_cmp)

    # Heatmap mês × setor
    hm = df_plot.copy()
    if "ano" not in hm.columns: hm["ano"] = raw["ano"]
    if "mes_num" not in hm.columns: hm["mes_num"] = raw["mes_num"]
    hm = hm.dropna(subset=["ano","mes_num"])
    if hm.empty:
        html_hm = "<p class='muted'>Sem dados suficientes para heatmap.</p>"
    else:
        hm["mes_ano"] = hm["mes_num"].map(MESES_PT).astype(str) + "/" + hm["ano"].astype(int).astype(str)
        table = hm.groupby(["mes_ano","setor"]).size().reset_index(name="n")
        pv = table.pivot_table(index="mes_ano", columns="setor", values="n", aggfunc="sum", fill_value=0)
        fig_hm = px.imshow(pv, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Contagem"), title="Heatmap mês × setor")
        html_hm = plot_html(fig_hm)

    # Alertas por anomalia (z-score)
    anom = df_plot.copy()
    if "ano" not in anom.columns: anom["ano"] = raw["ano"]
    if "mes_num" not in anom.columns: anom["mes_num"] = raw["mes_num"]
    anom = anom.dropna(subset=["ano","mes_num"])
    alert_table_html = "<p class='muted'>Sem dados para anomalias.</p>"
    if not anom.empty:
        anom["mkey"] = anom["ano"].astype(int).astype(str) + "-" + anom["mes_num"].astype(int).astype(str).str.zfill(2)
        order_m = (
            anom[["mkey","ano","mes_num"]].drop_duplicates().sort_values(["ano","mes_num"])
        )
        mkeys = order_m["mkey"].tolist()
        if len(mkeys) >= 2:
            last_k = mkeys[-1]
            g = anom.groupby(["resultado_std","mkey"]).size().reset_index(name="n")
            hist = g[g["mkey"] != last_k].groupby("resultado_std")["n"].agg(["mean","std","count"]).reset_index()
            hist = hist.rename(columns={"mean":"media_hist", "std":"std_hist", "count":"num_meses_hist"})
            cur = g[g["mkey"] == last_k][["resultado_std","n"]].rename(columns={"n":"n_cur"})
            alerts = pd.merge(cur, hist, on="resultado_std", how="left").fillna({"media_hist":0.0, "std_hist":0.0, "num_meses_hist":0})
            alerts["z"] = (alerts["n_cur"] - alerts["media_hist"]) / alerts["std_hist"].replace(0, np.nan)
            alerts["z"] = alerts["z"].fillna(0.0)
            alerts = alerts[(alerts["num_meses_hist"] >= int(min_hist)) & (alerts["z"] >= float(z_thr))]
            alerts = alerts.sort_values("z", ascending=False)
            if alerts.empty:
                alert_table_html = "<p class='ok'>Nenhuma anomalia detectada com os parâmetros atuais.</p>"
            else:
                nice = alerts.copy()
                nice["z"] = nice["z"].map(lambda x: f"{x:+.2f}σ")
                nice["media_hist"] = nice["media_hist"].map(lambda x: f"{x:.1f}")
                nice = nice.rename(columns={"resultado_std":"Micro-organismo","n_cur":"Mês atual","media_hist":"Média hist.","z":"z-score"})
                alert_table_html = nice.to_html(index=False)

    # Prévias (20 linhas)
    prev_raw = raw_df.head(20).to_html(index=False)
    prev_named = raw.head(20).to_html(index=False)

    # Não mapeados tabela
    if unmatched_suggested.empty:
        unmatched_html = "<p class='ok'>Todos os resultados foram mapeados.</p>"
    else:
        editable = unmatched_suggested.copy()
        editable["padronizado"] = ""
        editable["tipo_micro"] = ""
        unmatched_html = editable[["resultado", "resultado_norm", "sugestoes", "padronizado", "tipo_micro"]].to_html(index=False)

    # Badge período
    anos_disp = sorted(raw["ano"].dropna().astype(int).unique().tolist())
    meses_disp = sorted([int(m) for m in raw["mes_num"].dropna().unique().tolist() if 1 <= int(m) <= 12])
    meses_lbl = ", ".join(MESES_PT[m] for m in meses_disp) if meses_disp else "todos os meses"
    anos_lbl = ", ".join(str(a) for a in anos_disp) if anos_disp else "todos os anos"
    periodo_lbl = f"{meses_lbl} de {anos_lbl}"

    # ========= HTML final =========
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>HMA Analyzer — Resultado</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
   body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
   .card {{ border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
   .btn {{ display:inline-block; padding:8px 12px; border-radius:8px; border:1px solid #ddd; background:#f8fafc; text-decoration:none; color:#111827; }}
   .muted {{ color:#6b7280; font-size: 13px;}}
   .pill {{ display:inline-block; padding:2px 8px; background:#eef2ff; color:#3730a3; border-radius:999px; font-size:12px; }}
   h1 {{ margin: 0 0 12px; }}
   h2 {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>🧫 SCIH HMA — Resultado</h1>
  <p class="pill"><b>Período ativo:</b> {periodo_lbl} • <b>Registros carregados:</b> {len(raw)}</p>

  <div class="card">
    <a class="btn" href="/">⟵ Novo processamento</a>
    <a class="btn" href="{dl_pad_uri}" download="{dl_pad_name}">⬇️ Padronizado (CSV)</a>
    <a class="btn" href="{dl_trad_uri}" download="{dl_trad_name}">⬇️ Tradução atual (CSV)</a>
    {"<a class='btn' href=\"%s\" download=\"%s\">⬇️ Template de tradução (CSV)</a>" % (dl_template_uri, dl_template_name) if dl_template_uri else ""}
    <p class="muted">Baixe o template, preencha os mapeamentos faltantes e reenvie na próxima execução.</p>
  </div>

  <div class="card"><h2>Prévia da planilha bruta (20 linhas)</h2>{prev_raw}</div>
  <div class="card"><h2>Prévia já nomeada (20 linhas)</h2>{prev_named}</div>

  <div class="card">
    <h2>🔎 Resultados não mapeados</h2>
    {unmatched_html}
  </div>

  <div class="card"><h2>Micro-organismos Prevalentes (Top {int(top_n)})</h2>{html_top}</div>
  <div class="card"><h2>Gráfico de Barras por Setor</h2>{html_barras}</div>
  <div class="card"><h2>Distribuição de Resultados</h2>{html_pie_res}</div>
  <div class="card"><h2>Distribuição por Classe de Micro-organismo</h2>{html_pie_tm}</div>
  <div class="card"><h2>🗓️ Comparação Mensal</h2>{html_cmp}</div>
  <div class="card"><h2>Heatmap Mês × Setor</h2>{html_hm}</div>
  <div class="card"><h2>🚨 Alertas por Tendência Anômala</h2>{alert_table_html}</div>

  <p class="muted">Observação: esta versão reproduz o núcleo funcional do app Streamlit em uma página web FastAPI/Plotly, adequada ao Render (sem estado persistente).</p>
</body>
</html>
"""
    return HTMLResponse(html)

# Healthcheck opcional
@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()+"Z"}
