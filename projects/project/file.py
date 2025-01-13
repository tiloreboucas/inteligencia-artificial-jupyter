from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Definir os prompts do sistema e do usuário
system_prompt = "Você é um assistente prestativo e está respondendo perguntas gerais"
user_prompt = "{input}"

token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# Criar o template do prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", token_s + system_prompt),
    ("user", user_prompt + token_e)
])

# Configurar o Ollama local
llm = ChatOllama(
    endpoint='http://localhost:8000',
    model='phi3',
    temperature=0.1,
)

# Configurar a cadeia de prompts e LLM
chain3 = prompt | llm

# Definir a entrada do usuário
input = "Explique para mim em até 1 parágrafo o conceito de redes neurais, de forma clara e objetiva"

# Invocar a cadeia com a entrada do usuário
res = chain3.invoke({"input": input})

# Imprimir o resultado
print(res)
