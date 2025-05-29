import streamlit as st
import os

# Tenta importar as bibliotecas LLM, lida com erros se não estiverem instaladas
try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from litellm import completion
    LITELLM_AVAILABLE = True # Usado para Meta Llama e potencialmente outros
except ImportError:
    LITELLM_AVAILABLE = False

# Configuração da página
st.set_page_config(page_title="Gerador de Histórias Infantis", page_icon="🧸", layout="centered")

# --- Modelos LLM ---
llm_providers = {
    "OpenAI": {
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "available": OPENAI_AVAILABLE,
        "requires_endpoint": False
    },
    "Anthropic": {
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"],
        "available": ANTHROPIC_AVAILABLE,
        "requires_endpoint": False
    },
    "Azure OpenAI": {
        "models": ["gpt-4", "gpt-35-turbo"], # Estes são nomes de implantação, usuário deve configurar
        "available": OPENAI_AVAILABLE, # Usa a mesma biblioteca base
        "requires_endpoint": True,
        "extra_fields": ["azure_endpoint", "azure_api_version", "azure_deployment_name"]
    },
    "Meta (via LiteLLM)": {
        "models": ["meta-llama/Llama-3-8b-chat-hf", "meta-llama/Llama-3-70b-chat-hf"], # Exemplos, requer configuração no ambiente LiteLLM
        "available": LITELLM_AVAILABLE,
        "requires_endpoint": False # LiteLLM abstrai isso, mas pode precisar de variáveis de ambiente
    }
}

# --- Funções Auxiliares ---
def gerar_historia_com_llm(api_key, provider, model, deployment_name, api_version, endpoint, prompt):
    try:
        if provider == "OpenAI":
            if not OPENAI_AVAILABLE:
                return "Erro: Biblioteca OpenAI não instalada."
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        elif provider == "Anthropic":
            if not ANTHROPIC_AVAILABLE:
                return "Erro: Biblioteca Anthropic não instalada."
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif provider == "Azure OpenAI":
            if not OPENAI_AVAILABLE:
                return "Erro: Biblioteca OpenAI (para Azure) não instalada."
            if not endpoint or not deployment_name:
                 return "Erro: Para Azure OpenAI, o Endpoint e o Nome da Implantação são obrigatórios."
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version if api_version else "2024-02-15-preview",
                azure_endpoint=endpoint
            )
            response = client.chat.completions.create(
                model=deployment_name, # No Azure, model é o nome da implantação
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        elif provider == "Meta (via LiteLLM)":
            if not LITELLM_AVAILABLE:
                return "Erro: Biblioteca LiteLLM não instalada."
            os.environ["OPENAI_API_KEY"] = api_key # LiteLLM usa essa var se não achar outra específica
            if "meta-llama" in model:
                 pass 

            response = completion(
                model=model, 
                messages=[{"role": "user", "content": prompt}],
                api_key=api_key 
            )
            return response.choices[0].message.content

        else:
            return f"Erro: Provedor {provider} não suportado."

    except Exception as e:
        return f"Erro ao gerar história com {provider}: {str(e)}"

def construir_prompt_adaptado(faixa_etaria, tema):
    # Documentação de referência para adaptação: /home/ubuntu/praticas_historias_infantis.md
    prompt_base = f"Você é um contador de histórias infantis especialista em adaptar narrativas para diferentes faixas etárias. " \
                  f"Crie uma história infantil sobre '{tema}'. "

    if faixa_etaria == "0-12 meses":
        prompt_base += "A história deve ser extremamente simples, com frases muito curtas, quase como uma canção de ninar. " \
                       "Use muitas repetições de sons e palavras simples. O foco é no ritmo e na sonoridade. " \
                       "Inclua elementos que remetam a texturas e sensações táteis de forma imaginativa. " \
                       "Exemplo de tom: 'O gatinho miau, miau, fofinho. A bolinha piu, piu, colorida.'"
    elif faixa_etaria == "1-3 anos":
        prompt_base += "A história deve ter um enredo linear e fácil de seguir, com poucos personagens (animais são ótimos). " \
                       "Use frases curtas e vocabulário concreto e familiar. Repetições de frases ou ações são muito bem-vindas. " \
                       "A história pode envolver emoções básicas e ações claras. " \
                       "Exemplo de tom: 'O cachorrinho correu, correu. Ele achou um osso! Que feliz!'"
    elif faixa_etaria == "4-6 anos":
        prompt_base += "A história pode ter um enredo um pouco mais elaborado, com um pequeno desafio ou aventura. " \
                       "Pode incluir diálogos simples entre os personagens. O vocabulário pode ser um pouco mais rico, mas ainda acessível. " \
                       "Uma moral simples ou uma lição sobre amizade, coragem, ou curiosidade pode ser incluída sutilmente. " \
                       "Exemplo de tom: 'A pequena estrela decidiu explorar o céu. Ela encontrou um cometa amigo que lhe ensinou a brilhar mais forte.'"
    return prompt_base

# --- Interface do Usuário ---
st.title("🧸 Gerador de Histórias Infantis Mágicas ✨")
st.markdown("---_Transforme ideias em contos encantadores para os pequenos!_---")

# Instruções de Uso
with st.expander("📘 Como Funciona?", expanded=False):
    st.markdown("""
    Bem-vindo ao Gerador de Histórias Infantis!

    1.  **Escolha a Faixa Etária:** Selecione a idade da criança para que a história seja adaptada.
    2.  **Insira sua Chave de API:** Cole sua chave de API do serviço LLM que você deseja usar.
    3.  **Selecione o Provedor e Modelo LLM:** Escolha o provedor (ex: OpenAI) e, em seguida, o modelo específico (ex: gpt-4).
        *   **Para Azure OpenAI:** Você precisará fornecer também o **Endpoint**, a **Versão da API** (ex: 2024-02-15-preview) e o **Nome da Implantação** do seu modelo.
        *   **Para Meta (via LiteLLM):** Certifique-se de que seu ambiente LiteLLM esteja configurado para acessar os modelos Meta (pode exigir chaves adicionais como HUGGINGFACE_API_KEY configuradas como variáveis de ambiente).
    4.  **Defina a Temática:** Escreva sobre o que será a história (ex: amizade, coragem, um animal específico).
    5.  **Clique em Gerar:** Deixe a magia acontecer!
    """)

st.sidebar.header("🎨 Configurações da História")
faixa_etaria_opcoes = ["0-12 meses", "1-3 anos", "4-6 anos"]
faixa_etaria_selecionada = st.sidebar.selectbox("🎯 Para qual faixa etária é a história?", faixa_etaria_opcoes)

api_key_input = st.sidebar.text_input("🔑 Sua Chave de API", type="password", placeholder="Cole sua chave aqui")

# Filtrar provedores disponíveis (bibliotecas instaladas)
provedores_disponiveis = {p: d for p, d in llm_providers.items() if d["available"]}
if not provedores_disponiveis:
    st.sidebar.error("Nenhuma biblioteca de LLM foi encontrada. Por favor, instale as dependências necessárias (OpenAI, Anthropic, LiteLLM).")
    st.stop()

llm_provider_selecionado = st.sidebar.selectbox("🤖 Provedor LLM", options=list(provedores_disponiveis.keys()))

azure_endpoint_input = None
azure_api_version_input = None
azure_deployment_name_input = None
llm_model_selecionado = None

if llm_provider_selecionado:
    provider_details = provedores_disponiveis[llm_provider_selecionado]
    modelos_do_provedor = provider_details["models"]
    
    if provider_details.get("requires_endpoint"):
        st.sidebar.markdown("**Configurações Adicionais para Azure OpenAI:**")
        azure_endpoint_input = st.sidebar.text_input("🌐 Endpoint do Azure", placeholder="https://seu-recurso.openai.azure.com/")
        azure_api_version_input = st.sidebar.text_input("🗓️ Versão da API Azure", placeholder="2024-02-15-preview")
        azure_deployment_name_input = st.sidebar.text_input("⚙️ Nome da Implantação Azure", placeholder="nome-da-sua-implantacao-gpt4")
        llm_model_selecionado = azure_deployment_name_input 
    else:
        llm_model_selecionado = st.sidebar.selectbox("⚙️ Modelo Específico", options=modelos_do_provedor)

tematica_input = st.sidebar.text_area("💡 Qual será o tema da história?", placeholder="Ex: A aventura de um coelhinho curioso na floresta")

if st.sidebar.button("🪄 Gerar História!"):
    if not api_key_input:
        st.sidebar.error("Por favor, insira sua Chave de API.")
    elif not tematica_input:
        st.sidebar.error("Por favor, digite uma temática para a história.")
    elif not llm_provider_selecionado:
        st.sidebar.error("Por favor, selecione um provedor LLM.")
    elif provider_details.get("requires_endpoint") and (not azure_endpoint_input or not azure_deployment_name_input):
        st.sidebar.error("Para Azure OpenAI, o Endpoint e o Nome da Implantação são obrigatórios.")
    elif not llm_model_selecionado and not provider_details.get("requires_endpoint"):
        st.sidebar.error("Por favor, selecione um modelo LLM específico.")
    else:
        with st.spinner(f"Gerando história com {llm_provider_selecionado}..."):
            prompt_final = construir_prompt_adaptado(faixa_etaria_selecionada, tematica_input)
            
            nome_modelo_ou_implantacao = llm_model_selecionado
            if llm_provider_selecionado == "Azure OpenAI":
                modelo_param = None 
                implantacao_param = azure_deployment_name_input
            else:
                modelo_param = llm_model_selecionado
                implantacao_param = None

            historia_gerada = gerar_historia_com_llm(
                api_key=api_key_input,
                provider=llm_provider_selecionado,
                model=modelo_param, 
                deployment_name=implantacao_param, 
                api_version=azure_api_version_input,
                endpoint=azure_endpoint_input,
                prompt=prompt_final
            )
            st.subheader("📖 Sua História Personalizada:")
            st.markdown(f"<div style='background-color: #FFFACD; padding: 15px; border-radius: 10px; border: 2px solid #FFD700;'>{historia_gerada}</div>", unsafe_allow_html=True)

st.markdown("<br><br><hr>", unsafe_allow_html=True)
st.markdown("<footer><p style='text-align:center; color:grey;'>Criado com ❤️ por Manus IA</p></footer>", unsafe_allow_html=True)

# Para executar este aplicativo: salve como app.py e rode `streamlit run story.py` no terminal.