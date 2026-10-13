import os
import json
from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import yfinance as yf
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

# ==============================================================================
# 1. CONFIGURAÇÃO INICIAL E GESTÃO DO HISTÓRICO LOCAL (30 DIAS)
# ==============================================================================

st.set_page_config(page_title="Agente Financeiro", layout="wide", initial_sidebar_state="expanded")

def verificar_autenticacao():
    """
    Bloqueia o acesso à aplicação até que a palavra-passe correta seja inserida.
    A palavra-passe real NUNCA está aqui no código: é lida do st.secrets.
    """
    if st.session_state.get("utilizador_autenticado", False):
        return True

    st.title("🔒 Acesso Protegido")
    st.write("Por favor, introduz a palavra-passe para aceder à plataforma.")

    input_password = st.text_input("Palavra-passe:", type="password", key="login_pass_input")

    if st.button("Entrar", type="primary", key="login_btn"):
        
        password_correta = st.secrets.get("APP_PASSWORD")

        if password_correta and input_password == password_correta:
            st.session_state["utilizador_autenticado"] = True
            st.rerun()  
        else:
            st.error("Palavra-passe incorreta. Tenta novamente.")

    return False

# ==========================================================
# BARREIRA DE SEGURANÇA (Portão de entrada)
# ==========================================================
if not verificar_autenticacao():
    st.stop()  
    # O Streamlit para AQUI. Nada do código abaixo é executado ou mostrado!
# ==========================================================

api_key = st.secrets.get("GEMINI_API_KEY")

HISTORICO_FILE = "historico_tickers.json"

def carregar_e_limpar_historico():
    """Carrega o histórico e elimina automaticamente tickers com mais de 30 dias."""
    if not os.path.exists(HISTORICO_FILE):
        return {}
    try:
        with open(HISTORICO_FILE, "r") as f:
            data = json.load(f)
        
        agora = datetime.now()
        historico_valido = {}
        for ticker, fecha_iso in data.items():
            fecha_dt = datetime.fromisoformat(fecha_iso)
            if agora - fecha_dt <= timedelta(days=30):
                historico_valido[ticker] = fecha_iso
                
        if len(historico_valido) != len(data):
            guardar_historico(historico_valido)
            
        return historico_valido
    except Exception:
        return {}

def guardar_historico(data):
    """Guarda a estrutura de histórico atualizada no ficheiro JSON."""
    with open(HISTORICO_FILE, "w") as f:
        json.dump(data, f, indent=4)

def registar_ticker_analisado(ticker):
    """Adiciona ou atualiza a data de análise de um ticker no histórico."""
    historico = carregar_e_limpar_historico()
    historico[ticker.upper()] = datetime.now().isoformat()
    guardar_historico(historico)

historico_tickers = carregar_e_limpar_historico()

st.title("📊 Agente de Análise Financeira")
st.caption("Fonte dados: Yahoo Finance | Motor plataforma: Google GenAI (Gemini 3.8 Flash)")

# ==============================================================================
# SIDEBAR: CONFIGURAÇÃO DO PERFIL DO INVESTIDOR & CARTEIRA
# ==============================================================================
st.sidebar.header("👤 Perfil do Investidor")

budget_option = st.sidebar.select_slider(
    "Montante Disponível para Investir (€):",
    options=["< 500 €", "500 € - 2.500 €", "2.500 € - 10.000 €", "> 10.000 €"],
    value="500 € - 2.500 €"
)

perfil_risco = st.sidebar.selectbox(
    "Tolerância ao Risco:",
    ["Conservador (Preservação de capital)", "Moderado (Equilíbrio)", "Agressivo (Foco em Crescimento / Alta volatilidade)"],
    index=1
)

horizonte_tempo = st.sidebar.selectbox(
    "Horizonte Temporal:",
    ["Curto Prazo (< 2 anos)", "Médio Prazo (2 a 5 anos)", "Longo Prazo (> 5 anos / Reforma)"],
    index=2
)

objetivo_capital = st.sidebar.selectbox(
    "Objetivo Principal:",
    ["Geração de Rendimento Passivo (Dividendos)", "Acumulação / Crescimento de Capital", "Preservação de Capital / Anti-Inflação"],
    index=1
)

experiencia_investor = st.sidebar.radio(
    "Experiência no Mercado:",
    ["Iniciante", "Intermédio", "Avançado"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.header("💼 Carteira de Investimentos Atual")

df_carteira_default = pd.DataFrame([
    {"Ticker": "VWCE.DE", "N_Acoes": 15.0, "Valor_Investido_EUR": 1800.0},
    {"Ticker": "AAPL", "N_Acoes": 5.0, "Valor_Investido_EUR": 950.0}
])

df_carteira_input = st.sidebar.data_editor(
    df_carteira_default,
    num_rows="dynamic",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", help="Ex: AAPL, VWCE.DE", required=True),
        "N_Acoes": st.column_config.NumberColumn("N.º Ações", min_value=0.0, step=1.0, format="%.2f"),
        "Valor_Investido_EUR": st.column_config.NumberColumn("Investido (€)", min_value=0.0, step=50.0, format="%.2f €")
    },
    use_container_width=True,
    key="editor_carteira"
)

df_carteira_limpa = df_carteira_input.dropna(subset=["Ticker"])
df_carteira_limpa = df_carteira_limpa[df_carteira_limpa["Ticker"].str.strip() != ""]

if not df_carteira_limpa.empty:
    total_carteira = df_carteira_limpa["Valor_Investido_EUR"].sum()
    if total_carteira > 0:
        df_carteira_limpa["Peso_%"] = (df_carteira_limpa["Valor_Investido_EUR"] / total_carteira * 100).round(2)
    else:
        df_carteira_limpa["Peso_%"] = 0.0
    
    carteira_texto_prompt = f"Total Carteira: {total_carteira:.2f} €\n" + df_carteira_limpa.to_markdown(index=False)
else:
    carteira_texto_prompt = "O investidor ainda não possui investimentos realizados (início de carteira do zero)."

st.sidebar.markdown("---")

# ==============================================================================
# FUNÇÕES AUXILIARES & TRATAMENTO DE DADOS
# ==============================================================================

def format_percentage(val):
    if not isinstance(val, (int, float)) or pd.isna(val):
        return "N/A"
    adjusted_val = val if abs(val) > 5.0 else val * 100
    return f"{adjusted_val:.2f}%"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financial_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    info = stock.info

    if not info or info.get('quoteType') is None:
        raise ValueError(f"Não foram encontrados dados válidos para o ticker '{ticker_symbol}'.")

    quote_type = info.get('quoteType', 'EQUITY')
    nome_ativo = info.get('longName', info.get('shortName', ticker_symbol))
    moeda = info.get('currency', 'N/A')
    preco_atual = info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))
    fecho_anterior = info.get('previousClose', 'N/A')

    if quote_type == 'ETF':
        raw_yield = info.get('yield', info.get('dividendYield', info.get('trailingAnnualDividendYield', 'N/A')))
        yield_pct = format_percentage(raw_yield)
        category = info.get('category', 'N/A')

        df_metricas = pd.DataFrame({
            "Métrica": ["Nome", "Tipo", "Moeda", "Preço Atual", "Fecho Anterior", "Categoria", "Yield Distribuição"],
            "Valor": [nome_ativo, "ETF", moeda, preco_atual, fecho_anterior, category, yield_pct]
        })

        dados_prompt_extra = f"""
        - Categoria do ETF: {category}
        - Yield de Distribuição (valor bruto API): {yield_pct}
        - Descrição do Fundo: {str(info.get('longBusinessSummary', 'N/A'))[:500]}
        """

    else:
        pe_ratio = info.get('trailingPE', info.get('forwardPE', 'N/A'))
        beta = info.get('beta', 'N/A')
        raw_div_yield = info.get('dividendYield', info.get('trailingAnnualDividendYield', 'N/A'))
        div_yield = format_percentage(raw_div_yield)
        ebitda = info.get('ebitda', 'N/A')
        margem_op = format_percentage(info.get('operatingMargins', 'N/A'))
        roe = format_percentage(info.get('returnOnEquity', 'N/A'))

        df_metricas = pd.DataFrame({
            "Métrica": ["Nome", "Tipo", "Moeda", "Preço Atual", "Fecho Anterior", "P/E Ratio", "Beta", "Rendimento Div. (%)"],
            "Valor": [nome_ativo, "Ação", moeda, preco_atual, fecho_anterior, pe_ratio, beta, div_yield]
        })

        dados_prompt_extra = f"""
        - EBITDA: {ebitda}
        - Margem Operacional: {margem_op}
        - ROE (Return on Equity): {roe}
        - Rendimento Div. Bruto API: {div_yield}
        - Setor / Indústria: {info.get('sector', 'N/A')} | {info.get('industry', 'N/A')}
        - Crescimento de Receita (YoY): {info.get('revenueGrowth', 'N/A')}
        """

    return df_metricas, dados_prompt_extra, nome_ativo

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_deep_dive_data(ticker_symbol):
    """Extrai demonstrações financeiras dos últimos 3-4 anos."""
    stock = yf.Ticker(ticker_symbol)
    
    financials = stock.financials
    balance = stock.balance_sheet
    cashflow = stock.cashflow
    
    if (financials is None or financials.empty) and (balance is None or balance.empty) and (cashflow is None or cashflow.empty):
        raise ValueError(f"Demonstrações financeiras indisponíveis para '{ticker_symbol}'.")
        
    def safe_slice(df, rows):
        if df is None or df.empty:
            return pd.DataFrame()
        df_recent = df.iloc[:, :4] if df.shape[1] >= 4 else df
        available = [r for r in rows if r in df_recent.index]
        return df_recent.loc[available] if available else pd.DataFrame()

    inc_rows = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'Research And Development']
    bal_rows = ['Cash And Cash Equivalents', 'Total Debt', 'Stockholders Equity', 'Working Capital']
    cf_rows = ['Operating Cash Flow', 'Capital Expenditure', 'Free Cash Flow']

    df_inc = safe_slice(financials, inc_rows)
    df_bal = safe_slice(balance, bal_rows)
    df_cf = safe_slice(cashflow, cf_rows)

    return df_inc, df_bal, df_cf

# ==============================================================================
# TRATAMENTO RESILIENTE (TENACITY)
# ==============================================================================

def is_transient_error(exception):
    err_msg = str(exception).upper()
    return "503" in err_msg or "UNAVAILABLE" in err_msg or "RESOURCE_EXHAUSTED" in err_msg

@retry(
    retry=retry_if_exception(is_transient_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=3, max=15),
    reraise=True
)
def call_gemini_with_retry(client, model_name, prompt_text):
    return client.models.generate_content(
        model=model_name,
        contents=prompt_text
    )

# ==============================================================================
# COMPONENTE REUTILIZÁVEL: CHATBOT DE SEGUIMENTO
# ==============================================================================

def inicializar_novo_contexto_chat(titulo_contexto, texto_contexto):
    """Define o novo contexto ativo e limpa o histórico de chat anterior."""
    st.session_state["chat_active"] = True
    st.session_state["chat_title"] = titulo_contexto
    st.session_state["chat_context"] = texto_contexto
    st.session_state["chat_history"] = []

def anexa_ao_contexto_chat(texto_adicional):
    """Anexa informação extra ao contexto existente (ex: resultado do Deep Dive)."""
    if "chat_context" in st.session_state:
        st.session_state["chat_context"] += f"\n\n### DEEP DIVE CONTABILÍSTICO ADICIONAL:\n{texto_adicional}"

def renderizar_interface_chatbot(key_prefix):
    """Renderiza a caixa de chat caso exista um contexto ativo na sessão."""
    if st.session_state.get("chat_active", False):
        st.markdown("---")
        st.subheader(f"💬 Assistente de Dúvidas: {st.session_state.get('chat_title', 'Análise Atual')}")
        st.caption("Faz perguntas sobre este relatório, métricas ou implicações de investimento.")

        # Exibir histórico existente
        for msg in st.session_state.get("chat_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input de nova pergunta
        if user_question := st.chat_input(f"Pergunta algo sobre {st.session_state.get('chat_title')}...", key=f"{key_prefix}_chat_input"):
            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            if api_key:
                client = genai.Client(api_key=api_key)
                with st.chat_message("assistant"):
                    with st.spinner("A analisar a tua dúvida..."):
                        prompt_chat = f"""
                        Atua como o analista financeiro sénior que gerou os relatórios presentes neste contexto.
                        Responde à dúvida do investidor de forma objetiva e fundamentada.

                        ### CONTEXTO ATIVO DA ANÁLISE:
                        {st.session_state['chat_context']}

                        ### PERFIL DO INVESTIDOR:
                        - Budget: {budget_option} | Risco: {perfil_risco} | Horizonte: {horizonte_tempo} | Objetivo: {objetivo_capital}
                        - Carteira: {carteira_texto_prompt}
                        
                        ### HISTÓRICO DA CONVERSA:
                        """
                        for h in st.session_state["chat_history"][:-1]:
                            prompt_chat += f"\n- {h['role'].upper()}: {h['content']}"

                        prompt_chat += f"\n- USER: {user_question}\n- ASSISTANT:"

                        res_chat = call_gemini_with_retry(client, 'gemini-3.8-flash', prompt_chat)
                        ans_text = res_chat.text
                        st.markdown(ans_text)

                st.session_state["chat_history"].append({"role": "assistant", "content": ans_text})

# ==============================================================================
# MÓDULO UNIFICADO: EXECUÇÃO DO DEEP DIVE
# ==============================================================================

def executar_e_exibir_deep_dive(ticker, client):
    try:
        with st.spinner(f"A extrair DRE, Balanço e Fluxo de Caixa para {ticker}..."):
            df_inc, df_bal, df_cf = fetch_deep_dive_data(ticker)

        st.subheader(f"Demonstrações Financeiras Recentes: {ticker}")
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.write("**Demonstração de Resultados**")
            st.dataframe(df_inc, use_container_width=True)
        with col_d2:
            st.write("**Balanço Patrimonial**")
            st.dataframe(df_bal, use_container_width=True)
        with col_d3:
            st.write("**Fluxo de Caixa**")
            st.dataframe(df_cf, use_container_width=True)

        if client:
            with st.spinner("Gemini a auditar a saúde contabilística e Free Cash Flow..."):
                prompt_deep_dive = f"""
                Atua como auditor e analista de equity research sénior. Efetua um Deep Dive Contabilístico do ativo **{ticker}**.

                ### DEMONSTRAÇÃO DE RESULTADOS:
                {df_inc.to_markdown()}

                ### BALANÇO PATRIMONIAL:
                {df_bal.to_markdown()}

                ### FLUXO DE CAIXA:
                {df_cf.to_markdown()}

                ESTRUTURA A RESPOSTA EM 4 SECÇÕES:
                1. **Qualidade do Crescimento Operacional:** Analisa a evolução da Receita vs. Lucro Líquido e margens ao longo dos anos.
                2. **Estrutura de Capital & Solvência:** Avalia o nível de Dívida vs. Caixa e o risco de insolvência/alavancagem.
                3. **Análise do Free Cash Flow (FCF):** Verifica a eficiência de conversão de lucro em dinheiro vivo e despesas de capital (CAPEX).
                4. **Veredicto Contabilístico:** Conclusão sobre a solidez financeira do balanço (Solidez Alta / Média / Alerta Vermelho).
                """
                res_dd = call_gemini_with_retry(client, 'gemini-3.8-flash', prompt_deep_dive)
                st.markdown("---")
                st.markdown(res_dd.text)

                # Se já existir contexto no chat (ex: Tab 1), anexa o Deep Dive. Caso contrário (ex: Tab 3 direta), inicializa um novo.
                if st.session_state.get("chat_active") and ticker in st.session_state.get("chat_title", ""):
                    anexa_ao_contexto_chat(res_dd.text)
                else:
                    contexto_dd = f"DRE:\n{df_inc.to_markdown()}\n\nBalanço:\n{df_bal.to_markdown()}\n\nFCF:\n{df_cf.to_markdown()}\n\nAnálise:\n{res_dd.text}"
                    inicializar_novo_contexto_chat(f"Deep Dive {ticker}", contexto_dd)

    except Exception as e:
        st.error(f"Erro ao executar o Deep Dive para '{ticker}': {e}")

# ==============================================================================
# NAVEGAÇÃO POR TABS DA APLICAÇÃO
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Análise Individual de Ativo", 
    "💡 Discovery Mode (Ideias & Gaps)", 
    "📊 Deep Dive & Balanços", 
    "📜 Histórico & Gestão"
])

# ------------------------------------------------------------------------------
# TAB 1: ANÁLISE RÁPIDA DE ATIVO
# ------------------------------------------------------------------------------
with tab1:
    ticker_input = st.text_input("Insere o Ticker do Ativo ou ETF:", value="", placeholder="Ex: NVDA, AAPL, IWDA.AS, VWCE.DE", key="t1_ticker").upper().strip()

    if st.button("Analisar Ativo", type="primary", key="t1_btn"):
        if not ticker_input:
            st.warning("Por favor, insere um ticker válido antes de analisar.")
        else:
            # Limpa análises anteriores ao mudar de ticker ou iniciar nova pesquisa
            st.session_state["t1_etapa1_texto"] = None
            st.session_state["t1_etapa2_texto"] = None
            st.session_state["ultimo_ticker_t1"] = ticker_input
            try:
                with st.spinner("A recolher dados de mercado via Yahoo Finance..."):
                    df_metricas, dados_prompt_extra, nome_ativo = fetch_financial_data(ticker_input)
                    st.session_state["t1_df_metricas"] = df_metricas
                    st.session_state["t1_dados_prompt_extra"] = dados_prompt_extra
                    st.session_state["t1_nome_ativo"] = nome_ativo
            except Exception as e:
                st.error(f"Erro ao processar o ativo '{ticker_input}': {e}")

    # Processamento e Exibição Sequencial por Etapas
    if st.session_state.get("ultimo_ticker_t1") == ticker_input and ticker_input:
        
        # Recupera dados recolhidos
        df_metricas = st.session_state.get("t1_df_metricas")
        dados_prompt_extra = st.session_state.get("t1_dados_prompt_extra")
        nome_ativo = st.session_state.get("t1_nome_ativo")

        if df_metricas is not None:
            st.subheader(f"Resultados de Mercado: {nome_ativo}")
            st.dataframe(df_metricas, use_container_width=True)

            client = genai.Client(api_key=api_key) if api_key else None

            # Executar ETAPA 1 (caso ainda não tenha sido gerada)
            if not st.session_state.get("t1_etapa1_texto") and client:
                try:
                    with st.spinner("Etapa 1/2: A gerar análise de métricas e pontos de atenção..."):
                        prompt_etapa1 = f"""
                        Atua como um analista financeiro sénior, claro e rigoroso.
                        Analisa o ativo **{ticker_input}** com base nos dados exatos abaixo:

                        ### TABELA DE MÉTRICAS:
                        {df_metricas.to_markdown(index=False)}

                        ### DADOS EXTRA:
                        {dados_prompt_extra}

                        ### REGRAS CRÍTICAS DE SANIDADE:
                        Valida semanticamente os valores. Se detetares anomalias de escala ou erro evidente da API (ex: Nvidia com yield de 44%), ignora o erro numérico bruto, assinala brevemente e usa o valor real conhecido de mercado.

                        Responde APENAS com as seguintes 3 secções:
                        1. **Perfil do Ativo:** Breve explicação do modelo de negócio ou tese do ETF.
                        2. **Análise das Métricas & Sanidade:** O que revelam os números em termos de valuation, rentabilidade e dividendos reais (corrigindo eventuais erros da API).
                        3. **Pontos de Atenção:** 2 a 3 riscos operacionais, macroeconómicos ou fatores críticos a monitorizar.
                        """
                        res_etapa1 = call_gemini_with_retry(client, 'gemini-3.8-flash', prompt_etapa1)
                        st.session_state["t1_etapa1_texto"] = res_etapa1.text
                        st.rerun()

                except Exception as e:
                    st.error(f"Erro ao gerar a Etapa 1: {e}")
            # ----------------------------------------------------
            # EXIBIÇÃO DA ETAPA 1 (visível assim que concluída)
            # ----------------------------------------------------
            if st.session_state.get("t1_etapa1_texto"):
                st.markdown("---")
                st.markdown(st.session_state["t1_etapa1_texto"])

                # Executar ETAPA 2 (se já foi gerada etapa 1)
                if not st.session_state.get("t1_etapa2_texto") and client:
                    try: 
                        with st.spinner("Etapa 2/2: A cruzar com o teu Perfil, Métricas Brutas e Carteira Atual..."):
                            prompt_etapa2 = f"""
                            Continuando a análise do ativo **{ticker_input}**:

                            ### METRICAS BRUTAS DO ATIVO:
                            {df_metricas.to_markdown(index=False)}
                            {dados_prompt_extra}

                            ### ANÁLISE TÉCNICA GERADA NA ETAPA 1:
                            {st.session_state['t1_etapa1_texto']}

                            ### CONTEXTO DO INVESTIDOR:
                            - Montante Disponível (Budget): {budget_option}
                            - Tolerância ao Risco: {perfil_risco}
                            - Horizonte Temporal: {horizonte_tempo}
                            - Objetivo do Capital: {objetivo_capital}
                            - Experiência: {experiencia_investor}

                            ### CARTEIRA DE INVESTIMENTOS ATUAL DO INVESTIDOR:
                            {carteira_texto_prompt}

                            Com base em TODOS estes dados (Métricas, Análise Prévia, Perfil/contexto e Carteira Atual), gera a secção final:

                            4. **Barómetro de Investimento & Adequação ao Perfil (Veredicto):**
                            - **Classificação de Sinal:** [FORTE COMPRA | COMPRA | NEUTRO / MANTER | VENDA].
                            - **Adequação ao Perfil:** [MUITO ADEQUADO | PARCIALMENTE ADEQUADO | INADEQUADO].
                            - **Análise de Compatibilidade & Budget ({budget_option}):** Avalia a adequação ao valor a aportar, horizonte temporal e custos de fricção.
                            - **Impacto na Carteira e Diversificação:** Avalia detalhadamente se a adição deste ativo traz diversificação saudável à carteira atual do utilizador ou se gera sobre-exposição/sobreposição no mesmo setor ou geografia. Se o utilizador não tiver carteira criada, avalia como uma primeira posição de raiz.
                            - **Tese de Investimento:** 2 a 3 frases objetivas a fundamentar o veredicto.

                            Formata em Markdown claro em Português de Portugal.
                            """
                            res_etapa2 = call_gemini_with_retry(client, 'gemini-3.8-flash', prompt_etapa2)
                            st.session_state["t1_etapa2_texto"] = res_etapa2.text

                            # Regista no histórico e inicializa chat completo
                            registar_ticker_analisado(ticker_input)
                            relatorio_completo = f"{st.session_state['t1_etapa1_texto']}\n\n{st.session_state['t1_etapa2_texto']}"
                            contexto_chat = f"Métricas:\n{df_metricas.to_markdown(index=False)}\n{dados_prompt_extra}\n\nRelatório:\n{relatorio_completo}"
                            inicializar_novo_contexto_chat(f"Análise {ticker_input}", contexto_chat)
                            st.rerun()

                    except Exception as e:
                        st.error(f"Ocorreu um erro ao gerar a Etapa 2 (Veredicto): {e}")
                        if st.button("🔄 Tentar novamente apenas a Etapa 2", key="btn_retry_etapa2"):
                            st.rerun()

                # ----------------------------------------------------
                # EXIBIÇÃO DA ETAPA 2 (mostra o veredicto abaixo da Etapa 1)
                # ----------------------------------------------------
                if st.session_state.get("t1_etapa2_texto"):
                    st.markdown("---")
                    st.markdown(st.session_state["t1_etapa2_texto"])

                # Deep Dive Opcional na Tab 1
                st.markdown("---")
                st.subheader("🔬 Aprofundar análise?")
                confirmar_dd_t1 = st.checkbox(
                f"Confirmar extração dos balanços e demonstrações financeiras para **{ticker_input}**",
                key="chk_confirm_dd_t1")
                
                if confirmar_dd_t1:
                    if st.button(f"🚀 Executar Deep Dive de {ticker_input}", type="primary", key="btn_exec_dd_t1"):
                        client_instance = genai.Client(api_key=api_key) if api_key else None
                        executar_e_exibir_deep_dive(ticker_input, client_instance)

                # Chatbot na Tab 1
                renderizar_interface_chatbot(key_prefix="tab1")

# ------------------------------------------------------------------------------
# TAB 2: DISCOVERY MODE
# ------------------------------------------------------------------------------
with tab2:
    st.header("💡 Discovery Mode: Sugestões e Análise Comparativa")
    st.write("Identifica lacunas na tua carteira e gera uma matriz comparativa de potenciais ativos para alocação.")

    if st.button("Identificar Gaps e Gerar Shortlist", type="primary", key="disc_f1_btn"):
        if not api_key:
            st.warning("⚠️ Adiciona a GEMINI_API_KEY no teu ficheiro .env para usar o Discovery Mode.")
        else:
            try:
                client = genai.Client(api_key=api_key)
                with st.spinner("A analisar a tua carteira e perfil de investidor..."):
                    prompt_disc_fase1 = f"""
                    Atua como um alocador de ativos e gestor de carteiras sénior.
                    
                    ### PERFIL DO INVESTIDOR:
                    - Budget: {budget_option}
                    - Risco: {perfil_risco}
                    - Horizonte: {horizonte_tempo}
                    - Objetivo: {objetivo_capital}
                    - Experiência: {experiencia_investor}

                    ### CARTEIRA ATUAL:
                    {carteira_texto_prompt}

                    TAREFA:
                    1. Identifica de forma concisa os principais **Gaps / Lacunas** da carteira atual (ex: concentração num setor, falta de diversificação geográfica, falta de renda fixa/dividendos, etc.).
                    2. Propõe uma **Shortlist Focada de exatamente 3 a 4 Tickers específicos** (podem ser Ações ou ETFs) que complementem a carteira de forma ideal.
                    3. Para cada Ticker sugerido, indica o Ticker exato (ex: ASML.AS, NOVO-B.CO, CSEM.DE) e 1 frase com a tese de seleção.

                    Responde em Português de Portugal com formatação Markdown clara.
                    """
                    res_disc1 = call_gemini_with_retry(client, 'gemini-3.8-flash', prompt_disc_fase1)
                    st.session_state["discovery_fase1_output"] = res_disc1.text

            except Exception as e:
                st.error(f"Erro na Fase 1 do Discovery: {e}")

    if "discovery_fase1_output" in st.session_state:
        st.markdown("---")
        st.markdown(st.session_state["discovery_fase1_output"])

        st.markdown("---")
        st.subheader("Fase 2: Extração em Lote & Relatório Comparativo de Escolha")
        
        tickers_suggested = st.text_input(
            "Confirma ou insere os 3 a 4 Tickers a comparar (separados por vírgulas):", 
            value="", 
            placeholder="Ex: ASML.AS, NOVO-B.CO, VWCE.DE",
            key="disc_tickers_input"
        )

        if st.button("Gerar Análise Comparativa dos Ativos Sugeridos", type="secondary", key="disc_f2_btn"):
            lista_tickers = [t.strip().upper() for t in tickers_suggested.split(",") if t.strip()]
            
            if len(lista_tickers) < 2:
                st.warning("Insere pelo menos 2 tickers válidos para efetuar uma análise comparativa.")
            else:
                try:
                    resumo_comparativo = []
                    with st.spinner("A recolher métricas numéricas em lote via Yahoo Finance..."):
                        for t in lista_tickers:
                            df_m, _, nome = fetch_financial_data(t)
                            dict_m = dict(zip(df_m["Métrica"], df_m["Valor"]))
                            dict_m["Ticker"] = t
                            dict_m["Nome"] = nome
                            resumo_comparativo.append(dict_m)

                    df_comp = pd.DataFrame(resumo_comparativo)
                    colunas_ruido = ["Tipo", "Moeda", "Fecho Anterior"]
                    colunas_validas = [col for col in df_comp.columns if col not in colunas_ruido]
                    df_comp_filtrado = df_comp[colunas_validas]

                    st.subheader("Matriz de Dados Extraídos em Lote")
                    st.dataframe(df_comp_filtrado, use_container_width=True)

                    if api_key:
                        client = genai.Client(api_key=api_key)
                        with st.spinner("A sintetizar relatório comparativo e ranking de alocação..."):
                            prompt_disc_fase2 = f"""
                            Atua como analista financeiro. Com base na matriz comparativa abaixo e no contexto do investidor, gera uma síntese de escolha.

                            ### MATRIZ DE DADOS DOS CANDIDATOS:
                            {df_comp_filtrado.to_markdown(index=False)}

                            ### CONTEXTO DO INVESTIDOR:
                            - Budget: {budget_option} | Risco: {perfil_risco} | Objetivo: {objetivo_capital}
                            - Carteira Atual: {carteira_texto_prompt}

                            GERA O SEGUINTE RELATÓRIO:
                            1. Tabela Comparativa Resumida & Rácio Risco/Retorno.**Destaca os pontos fortes e fracos de cada um face aos outros.
                            2. Impacto na Carteira e Diversificação.
                            3. Ranking de Alocação Recomendado.**Define uma ordem de prioridade/percentagem do budget ({budget_option}) a alocar.
                            """
                            res_disc2 = call_gemini_with_retry(client, 'gemini-3.8-flash', prompt_disc_fase2)
                            relatório_disc_texto = res_disc2.text
                            st.session_state["relatorio_disc_texto"] = relatório_disc_texto
                            st.markdown("---")
                            st.markdown(relatório_disc_texto)

                            for t in lista_tickers:
                                registar_ticker_analisado(t)
                            st.success("✅ Todos os tickers comparados foram registados no Histórico Local.")

                            # Reset inteligente do Chat para a comparação do Discovery
                            contexto_disc = f"Matriz Comparativa:\n{df_comp_filtrado.to_markdown(index=False)}\n\nRelatório Comparativo:\n{relatório_disc_texto}"
                            inicializar_novo_contexto_chat(f"Discovery ({', '.join(lista_tickers)})", contexto_disc)

                except Exception as e:
                    st.error(f"Erro na Fase 2 do Discovery: {e}")

        # Chatbot na Tab 2 (se o utilizador já tiver gerado o relatório comparativo)
        if "relatorio_disc_texto" in st.session_state:
            renderizar_interface_chatbot(key_prefix="tab2")

# ------------------------------------------------------------------------------
# TAB 3: DEEP DIVE & BALANÇOS
# ------------------------------------------------------------------------------
with tab3:
    st.header("📊 Deep Dive: Análise de Balanços e Demonstrações Financeiras")
    st.write("Aprofunda a análise dos demonstrativos contabilísticos dos ativos previamente verificados.")

    historico_atual = carregar_e_limpar_historico()
    ticker_dd = st.text_input("Insere o Ticker para Deep Dive:", value="", placeholder="Ex: NVDA, AAPL", key="dd_ticker_input").upper().strip()

    if ticker_dd:
        if ticker_dd not in historico_atual:
            st.warning(f"⚠️️ O ativo '{ticker_dd}' ainda não foi analisado nos últimos 30 dias. Por favor, corre primeiro uma Análise Rápida para registar o ativo.")
        else:
            st.info(f"✅ Ativo '{ticker_dd}' validado no Histórico (Última análise: {historico_atual[ticker_dd][:10]}).")
            
            if st.button("Executar Deep Dive Financeiro", type="primary", key="dd_exec_btn"):
                client_instance = genai.Client(api_key=api_key) if api_key else None
                executar_e_exibir_deep_dive(ticker_dd, client_instance)

    # Chatbot na Tab 3 (para tirar dúvidas específicas dos balanços)
    renderizar_interface_chatbot(key_prefix="tab3")

# ------------------------------------------------------------------------------
# TAB 4: HISTÓRICO & GESTÃO
# ------------------------------------------------------------------------------
with tab4:
    st.header("📜 Histórico Local de Ativos Analisados")
    st.write("Lista de tickers analisados recentemente. Registo válido por 30 dias.")

    historico_atual = carregar_e_limpar_historico()

    if not historico_atual:
        st.info("O histórico está atualmente vazio. Realiza análises de ativos para registar tickers.")
    else:
        df_hist = pd.DataFrame([
            {"Ticker": k, "Data Análise": v[:10], "Hora": v[11:16]} 
            for k, v in historico_atual.items()
        ])
        st.dataframe(df_hist, use_container_width=True)

        st.markdown("---")
        if st.button("🗑️ Limpar Histórico de Ativos Agora", type="secondary", key="clear_hist_btn"):
            guardar_historico({})
            st.session_state["chat_active"] = False
            st.success("Histórico limpo com sucesso!")
            st.rerun()