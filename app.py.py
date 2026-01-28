import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# ==========================================
# CONFIGURAÇÃO E ESTILO
# ==========================================
st.set_page_config(page_title="Dashboard – Eventos", layout="wide", page_icon="📊")

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    .main { background-color: #f8f9fa; }
    </style>
    """,
    unsafe_allow_html=True,
)

EVENTOS_ALVO = ["VIP Deutsch", "Nuevo_sun", "Winterfall"]

# ==========================================
# FUNÇÕES DE PROCESSAMENTO (Com Cache)
# ==========================================
@st.cache_data
def carregar_e_limpar_dados(file):
    try:
        df = pd.read_csv(file, sep=None, engine="python")
        df.columns = df.columns.str.strip()

        colunas_necessarias = [
            "ID",
            "Data",
            "Descrição",
            "Fornecedor/Cliente",
            "Classificação",
            "Valor",
            "Status",
            "Tipo",
        ]
        for col in colunas_necessarias:
            if col not in df.columns:
                return None, f"Coluna ausente: {col}"

        df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")

        def limpar_valor(val):
            if pd.isna(val):
                return np.nan
            val = str(val).replace("R$", "").replace(" ", "").strip()
            if "." in val and "," in val:  # 1.234,56
                val = val.replace(".", "").replace(",", ".")
            elif "," in val:  # 1234,56
                val = val.replace(",", ".")
            return float(val)

        df["Valor_num"] = df["Valor"].apply(limpar_valor)

        df = df.dropna(subset=["Data", "Valor_num"]).copy()

        df["Tipo"] = df["Tipo"].astype(str).str.strip().str.upper()
        df["Status"] = df["Status"].astype(str).str.strip()
        df["Descrição"] = df["Descrição"].astype(str).str.strip()
        df["Fornecedor/Cliente"] = df["Fornecedor/Cliente"].astype(str).str.strip()
        df["Classificação"] = df["Classificação"].astype(str).str.strip()

        return df, None
    except Exception as e:
        return None, str(e)


def categorizar_lancamento(descricao: str, fornecedor: str, classificacao: str) -> str:
    desc = f"{descricao} {fornecedor} {classificacao}".lower()

    if classificacao in EVENTOS_ALVO:
        return "Direto Evento"

    if re.search(
        r"\brh\b|folha|sal[aá]rio|salario|pr[oó]-?labore|prolabore|benef[ií]cio|beneficio|inss|fgts|13|f[eé]rias|ferias|vale|vr|vt|plano de sa[uú]de|plano saude",
        desc,
    ):
        return "Pessoas (RH/Folha)"

    if re.search(
        r"receita federal|darf|imposto|tributo|pis|cofins|csll|irpj|simples|icms|iss|sefaz|gps|e-social|esocial",
        desc,
    ):
        return "Receita Federal / Impostos"

    if re.search(
        r"administrativ|despesa geral|overhead|contabil|contabilidade|aluguel|energia|internet|telefone|software|licen[cç]a|servi[cç]o|manuten[cç][aã]o|cart[aã]o|banco|tarifa|marketing institucional|coworking|escritorio|limpeza|seguran[cç]a|suporte",
        desc,
    ):
        return "Overhead Operacional"

    return "Overhead Operacional"


def preparar_visao_custos_por_evento(df_filtrado: pd.DataFrame) -> dict:
    d = df_filtrado.copy()

    d["Categoria"] = d.apply(
        lambda r: categorizar_lancamento(
            str(r.get("Descrição", "")),
            str(r.get("Fornecedor/Cliente", "")),
            str(r.get("Classificação", "")),
        ),
        axis=1,
    )

    ev = d[d["Classificação"].isin(EVENTOS_ALVO)].copy()
    resumo_margem_evento = ev.groupby("Classificação")["Valor_num"].agg(
        Receita=lambda s: s[s > 0].sum(),
        Despesa=lambda s: s[s < 0].sum(),
    ).reset_index()

    resumo_margem_evento["Margem R$"] = resumo_margem_evento["Receita"] + resumo_margem_evento["Despesa"]
    resumo_margem_evento["Margem %"] = np.where(
        resumo_margem_evento["Receita"] > 0,
        (resumo_margem_evento["Margem R$"] / resumo_margem_evento["Receita"]) * 100,
        0,
    )

    corporativo = d[d["Categoria"].isin(["Pessoas (RH/Folha)", "Receita Federal / Impostos"])].copy()
    overhead = d[d["Categoria"] == "Overhead Operacional"].copy()

    custos_diretos = (
        ev[ev["Valor_num"] < 0]
        .groupby("Classificação")["Valor_num"]
        .sum()
        .reset_index()
        .rename(columns={"Classificação": "Evento", "Valor_num": "Custos Diretos (neg)"})
    )

    base_rateio = resumo_margem_evento[["Classificação", "Receita"]].copy()
    soma_receita = base_rateio["Receita"].sum()

    if soma_receita <= 0 or base_rateio.empty or overhead.empty:
        custos_evento_total = custos_diretos.copy()
        custos_evento_total["Overhead Operacional Rateado (neg)"] = 0.0
        custos_evento_total["Custos Totais (neg)"] = custos_evento_total["Custos Diretos (neg)"]
        return {
            "resumo_margem_evento": resumo_margem_evento,
            "custos_evento_total": custos_evento_total,
            "corporativo": corporativo,
            "overhead_rateado": pd.DataFrame(),
        }

    base_rateio["Pct_Rateio"] = base_rateio["Receita"] / soma_receita

    overhead_rateado = overhead.merge(base_rateio[["Classificação", "Pct_Rateio"]], how="cross")
    overhead_rateado = overhead_rateado.rename(
        columns={"Classificação_y": "Evento", "Classificação_x": "Classificação"}
    )
    overhead_rateado["Valor_rateado"] = overhead_rateado["Valor_num"] * overhead_rateado["Pct_Rateio"]

    custos_overhead_rateado = (
        overhead_rateado.groupby("Evento")["Valor_rateado"]
        .sum()
        .reset_index()
        .rename(columns={"Valor_rateado": "Overhead Operacional Rateado (neg)"})
    )

    custos_evento_total = custos_diretos.merge(custos_overhead_rateado, on="Evento", how="outer").fillna(0)
    custos_evento_total["Custos Totais (neg)"] = (
        custos_evento_total["Custos Diretos (neg)"] + custos_evento_total["Overhead Operacional Rateado (neg)"]
    )

    return {
        "resumo_margem_evento": resumo_margem_evento,
        "custos_evento_total": custos_evento_total,
        "corporativo": corporativo,
        "overhead_rateado": overhead_rateado,
    }


# ==========================================
# PRIMEIRO PASSO: ANEXAR ARQUIVO
# ==========================================
st.title("📎 Anexe o arquivo aqui")
arquivo = st.file_uploader("Selecione o CSV (MNM)", type=["csv"])

if arquivo is None:
    st.info("Anexe o arquivo CSV para carregar o dashboard.")
    st.stop()

# ==========================================
# CARREGA DADOS
# ==========================================
df, erro = carregar_e_limpar_dados(arquivo)
if erro:
    st.error(f"Erro ao processar arquivo: {erro}")
    st.stop()

# ==========================================
# SIDEBAR: FILTROS (Datas + Status Pago/Agendado)
# ==========================================
with st.sidebar:
    st.header("📂 Filtros")

    data_min, data_max = df["Data"].min().date(), df["Data"].max().date()
    periodo = st.date_input("Período de Análise", value=(data_min, data_max), format="DD/MM/YYYY")

    st.subheader("Status (Pago / Agendado)")
    status_padrao = ["Pago", "Agendado"]
    status_opcoes = sorted(df["Status"].dropna().unique().tolist())

    default_status = [s for s in status_padrao if s in status_opcoes]
    if not default_status:
        default_status = status_opcoes

    status_sel = st.multiselect("Selecione os status", status_opcoes, default=default_status)

# ==========================================
# APLICA FILTROS
# ==========================================
if isinstance(periodo, tuple) and len(periodo) == 2:
    df_f = df[(df["Data"].dt.date >= periodo[0]) & (df["Data"].dt.date <= periodo[1])].copy()
else:
    df_f = df.copy()

df_f = df_f[df_f["Status"].isin(status_sel)].copy()

# ==========================================
# CABEÇALHO
# ==========================================
st.markdown("### 📊 Dashboard Financeiro – Eventos")
st.caption("Gestão de Margem • Fluxo de Caixa • Auditoria de Custos")

# ==========================================
# KPIs PRINCIPAIS (CAIXA com sinal correto)
# ==========================================
entradas = df_f[df_f["Valor_num"] > 0]["Valor_num"].sum()
saidas = df_f[df_f["Valor_num"] < 0]["Valor_num"].sum()  # negativo
saldo_liq = entradas + saidas

fluxo_diario = df_f.groupby("Data")["Valor_num"].sum().sort_index()
saldo_acumulado = fluxo_diario.cumsum()

col1, col2, col3 = st.columns(3)
col1.metric("Entradas", f"R$ {entradas:,.2f}")
col2.metric("Saídas", f"R$ {saidas:,.2f}")
col3.metric("Saldo Líquido", f"R$ {saldo_liq:,.2f}")

st.divider()

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Rentabilidade", "💰 Fluxo de Caixa", "🔍 Auditoria"])

# --- TAB 1
with tab1:
    st.subheader("Margem de Contribuição por Evento (Direto do Evento)")

    df_ev = df_f[df_f["Classificação"].isin(EVENTOS_ALVO)].copy()

    if df_ev.empty:
        st.warning("Nenhum dado encontrado para os eventos selecionados.")
    else:
        resumo = df_ev.groupby("Classificação")["Valor_num"].agg(
            Receita=lambda s: s[s > 0].sum(),
            Despesa=lambda s: s[s < 0].sum(),
        ).reset_index()

        resumo["Margem R$"] = resumo["Receita"] + resumo["Despesa"]
        resumo["Margem %"] = np.where(resumo["Receita"] > 0, (resumo["Margem R$"] / resumo["Receita"]) * 100, 0)

        c1, c2 = st.columns([1, 1])
        fig_bar = px.bar(resumo, x="Classificação", y="Margem R$", title="Margem em Valor (R$)", text_auto=".2s")
        c1.plotly_chart(fig_bar, use_container_width=True)

        fig_pie = px.pie(resumo, values="Receita", names="Classificação", title="Participação na Receita Total", hole=0.4)
        c2.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(
            resumo.style.format(
                {"Receita": "R$ {:,.2f}", "Despesa": "R$ {:,.2f}", "Margem R$": "R$ {:,.2f}", "Margem %": "{:.1f}%"}
            ),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Custos por Evento (Diretos + Overhead Operacional Rateado)")

    visoes = preparar_visao_custos_por_evento(df_f)
    custos_evento_total = visoes["custos_evento_total"]
    corporativo = visoes["corporativo"]
    overhead_rateado = visoes["overhead_rateado"]

    if custos_evento_total is None or custos_evento_total.empty:
        st.info("Sem dados suficientes para custos por evento no período/filtros.")
    else:
        tabela_custos = custos_evento_total.copy()
        tabela_custos["Custos Diretos"] = (-tabela_custos["Custos Diretos (neg)"]).round(2)
        tabela_custos["Overhead Operacional Rateado"] = (-tabela_custos["Overhead Operacional Rateado (neg)"]).round(2)
        tabela_custos["Custos Totais"] = (-tabela_custos["Custos Totais (neg)"]).round(2)
        tabela_custos = tabela_custos[["Evento", "Custos Diretos", "Overhead Operacional Rateado", "Custos Totais"]]

        fig_custos = px.bar(
            tabela_custos,
            x="Evento",
            y=["Custos Diretos", "Overhead Operacional Rateado"],
            title="Composição de Custos por Evento (R$)",
            barmode="stack",
        )
        st.plotly_chart(fig_custos, use_container_width=True)

        st.dataframe(
            tabela_custos.style.format(
                {"Custos Diretos": "R$ {:,.2f}", "Overhead Operacional Rateado": "R$ {:,.2f}", "Custos Totais": "R$ {:,.2f}"}
            ),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Corporativo (não rateado): Pessoas e Receita Federal / Impostos")

    if corporativo is None or corporativo.empty:
        st.success("Nenhum lançamento corporativo no período/filtros.")
    else:
        corp_res = corporativo.groupby("Categoria")["Valor_num"].sum().reset_index()
        corp_res["Total (R$)"] = corp_res["Valor_num"].abs().round(2)

        fig_corp = px.bar(corp_res, x="Categoria", y="Total (R$)", title="Custos Corporativos (R$)")
        st.plotly_chart(fig_corp, use_container_width=True)

        with st.expander("Ver lançamentos corporativos"):
            st.dataframe(
                corporativo[["Data", "Categoria", "Fornecedor/Cliente", "Descrição", "Status", "Tipo", "Valor_num"]].sort_values("Data"),
                use_container_width=True,
            )

    if overhead_rateado is not None and not overhead_rateado.empty:
        with st.expander("Ver memória do rateio (Overhead Operacional -> Eventos)"):
            st.dataframe(
                overhead_rateado[
                    ["Data", "Fornecedor/Cliente", "Descrição", "Status", "Tipo", "Valor_num", "Evento", "Pct_Rateio", "Valor_rateado"]
                ].sort_values(["Evento", "Data"]),
                use_container_width=True,
            )

# --- TAB 2
with tab2:
    st.subheader("Análise de Liquidez Diária (Saldo Acumulado)")
    if not saldo_acumulado.empty:
        fig_area = px.area(
            saldo_acumulado,
            title="Evolução do Saldo Bancário (Acumulado)",
            labels={"value": "Saldo (R$)", "Data": "Dia"},
        )
        fig_area.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_area, use_container_width=True)

        pior_ponto = saldo_acumulado.min()
        st.error(f"📉 O menor saldo foi **R$ {pior_ponto:,.2f}** em **{saldo_acumulado.idxmin().strftime('%d/%m/%Y')}**.")

        with st.expander("Ver fluxo diário (movimento e saldo acumulado)"):
            tabela_fluxo = pd.DataFrame({"Movimento do Dia": fluxo_diario, "Saldo Acumulado": saldo_acumulado}).reset_index()
            tabela_fluxo = tabela_fluxo.rename(columns={"index": "Data"})
            st.dataframe(tabela_fluxo, use_container_width=True)
    else:
        st.info("Aguardando dados para gerar gráfico...")

# --- TAB 3
with tab3:
    st.subheader("Auditoria Operacional: VIP Deutsch")

    df_vip = df_f[(df_f["Classificação"] == "VIP Deutsch") & (df_f["Valor_num"] < 0)].copy()

    if df_vip.empty:
        st.success("Nenhuma irregularidade ou custo encontrado para VIP Deutsch.")
    else:
        df_vip["Valor_Abs"] = df_vip["Valor_num"].abs()
        duplicados = df_vip[df_vip.duplicated(subset=["Data", "Fornecedor/Cliente", "Valor_Abs"], keep=False)]

        if not duplicados.empty:
            st.warning(f"⚠️ {len(duplicados)} lançamentos com suspeita de duplicidade (mesmo dia, valor e fornecedor).")
            st.dataframe(
                duplicados[["Data", "Fornecedor/Cliente", "Descrição", "Valor_num"]].sort_values("Data"),
                use_container_width=True,
            )
            st.caption(f"Impacto potencial (soma dos duplicados): R$ {duplicados['Valor_Abs'].sum():,.2f}")
        else:
            st.success("Nenhuma duplicidade exata encontrada pelos critérios automáticos.")

        with st.expander("Ver todos os custos VIP Deutsch"):
            st.dataframe(
                df_vip[["Data", "Fornecedor/Cliente", "Descrição", "Status", "Tipo", "Valor_num"]].sort_values("Valor_num"),
                use_container_width=True,
            )

# ==========================================
# RODAPÉ
# ==========================================
st.markdown("---")
st.caption(f"Dashboard gerado em {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
