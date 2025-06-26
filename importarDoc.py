# Pacotes para importar
from llama_cpp import Llama
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os
import re

# Configurações iniciais
model_path = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
document_path = "./meusDocs/FerramentasdeIA.txt"

# Limpar dados inúteis do arquivo
def clean_text(text):
    # Remove caracteres especiais excessivos e espaços extras
    text = re.sub(r'[^\\w\\s,!?-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 1. Carregar e dividir o documento
def load_and_split_document():
    if not os.path.exists(document_path):
        print(f"Arquivo não encontrado: {document_path}")
        return None
        
    # Pré-processar o texto
    # Usar dois ou mais separadores:
    separators = [r"\n\n", r"\.\s", r"\? ", r"! "]  # exemplos de separadores
    # Combinar em uma única expressão regex
    regex_sep = "|".join(separators)
    # re.split para dividir o texto antes de usar o TextSplitter
    with open(document_path, "r", encoding="utf-8") as f:
      texto = f.read()
      
    # Dividir o texto manualmente
    partes = re.split(regex_sep, texto)
    # Juntar novamente com um separador padrão (por exemplo, '\n\n') para usar o splitter
    texto_preprocessado = "\n\n".join(partes)
    
    # Inicializa o splitador de texto
    # chunk_size: tamanho máximo de cada pedaço (em caracteres)
    # chunk_overlap: quantidade de caracteres que serão sobrepostos entre os pedaços
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=180, 
        chunk_overlap=50,
        length_function=len
    )
    texts = text_splitter.create_documents([texto_preprocessado])
    print(f"Dividido em {len(texts)} pedaços")
    return texts

# 2. Gerar embeddings e Criar o banco de vetores
def create_vector_db(texts):
    print("Gerando embeddings e construindo o banco de vetores...")

    # Este é um modelo pequeno e eficiente para CPU
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Este é um modelo para português do Brasil
    embeddings = SentenceTransformerEmbeddings(model_name="neuralmind/bert-base-portuguese-cased")

    # Criar o banco de vetores FAISS a partir dos textos e do modelo de embeddings
    db = FAISS.from_documents(texts, embeddings)
    print("Banco de vetores FAISS criado com sucesso!")
    return db

# 3. Inicializar o modelo TinyLlama
def initialize_tinyLlama():
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {model_path}")

    print(f"Carregando modelo em {model_path}...")
    llama = Llama(
        model_path=model_path, 
        n_ctx=2048, # Tamanho máximo do contexto
        n_gpu_layers=0, # Não usar GPU
        verbose=False # Não mostrar logs
    )
    print("TinyLlama carregado com sucesso!")
    return llama

# 4. Gerar a resposta com RAG
def generate_rag_response(query, vector_db, llm_model):
    # Buscar informações relevantes no banco de vetores e as usa 
    # para gerar a resposta com o TinyLlama
    print(f"Buscando informações relevantes para: {query}")
    # Buscar os top 5 pedaços mais similares no banco de vetores
    docs_with_scores = vector_db.similarity_search_with_score(query, k=5)
    # Criar o contexto para o TinyLlama com esses 5 pedaços de texto
    context = ""
    for doc, score in docs_with_scores:
        context += f"{doc.page_content[:180]}. \n"

    # Template de prompt para o TinyLlama
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "<|im_start|>system\Sua priooridade é usar apenas as informações no contexto fornecido para responder à pergunta em português do Brasil. "
            "Responda apenas com base nas informações do contexto e não faça suposições. "
            "Responda com a linha correspondente do contexto e nada mais."
            "Desejo que a resposta seja sempre no contexto colocado em português do Brasil."
            "Não faça suposições, apenas use as informações do contexto."
            "Não fuja da pergunta, responda apenas o que foi perguntado."
            "As frases do contexto estão separadas por um ponto final. Informe apenas a frase correspondente à pergunta."
            "Caso não exista a resposta no contexto informe: Informação não encontrada no contexto.'<|im_end|>\n"
            "<|im_start|>user\nContexto:\n{context}\n\nPergunta: {question}<|im_end|>\n"
        )
    )

    # Criar uma cadeia de LLM para gerar a resposta
    # Aqui, em vez de usar o LLMChain diretamente com o modelo GGUF, vamos chamar o modelo
    # diretamente com o prompt e o contexto para melhor controle sobre o processo de geração
    # de texto.
    formatted_prompt = prompt_template.format(context=context, question=query)

    print("Gerando resposta com o modelo TinyLlama...")
    response = llm_model(
        formatted_prompt,
        temperature=0.1,
        max_tokens=500, # Limite de tokens na resposta
        stop=["<|im_end|>", "."], # Tokens de parada
        echo=False, # Não mostrar o prompt e a resposta
    )

    # Verificar o prompt formatado
    # print("-----------------------------------")
    # print("Prompt do modelo: ", formatted_prompt)
    # print("-----------------------------------")
    generated_text = response["choices"][0]["text"].strip()
    return generated_text

# 5. Execução principal
if __name__ == "__main__":
    print("Iniciando o processo de importação e RAG...")

    # 1. Carregar e dividir o documento
    texts = load_and_split_document()
    if texts is None:
        print("Erro ao carregar e dividir o documento")
        exit(1)

    # 2. Gerar embeddings e Criar o banco de vetores
    db = create_vector_db(texts)

    # 3. Inicializar o modelo TinyLlama
    llama = initialize_tinyLlama()
    if llama is None:
        print("Erro ao carregar o modelo TinyLlama")
        exit(1)

    # 4. Gerar a resposta com RAG
    print("..IA de Conversação..")
    print("Para sair, digite 'sair'")
    while True:
        user_query = input("Você: ")
        if user_query.lower() == "sair":
            print("Saindo...")
            break
        response = generate_rag_response(user_query, db, llama)
        print(f"Resposta: {response}")
