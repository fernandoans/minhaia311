# IA Gratuita no seu Contexto
O objetivo deste projeto é ter uma IA complemente gratuita e que possa responder as perguntas dado um determinado contexto.

O pré-requisito obrigatório para todo esse projeto é a capacidade de tudo poder ser executado em um simples computador com 16 Gb de RAM e espaço em disco de no máximo 1 Tb.

Aqui foi utilizado o método RAG RAG (Retrieval-Augmented Generation), ou Geração Aumentada por Recuperação, é uma técnica da Inteligência Artificial, especialmente no contexto de LLMs (Grandes Modelos de Linguagem ). Combina a capacidade de geração de texto dos LLMs com a habilidade de recuperar informações de fontes externas.

Ferramenta utilizada:
Uso e recomendo o Cursor 1.1.3 que é um clone do Visual Studio.

Esse é o primeiro projeto de uma série. Neste foram utilizadas as seguintes tecnologias:
* Python 3.11 (de modo a manter a compatibilidade das ferramentas)
* Modelo tinyllama-1.1b-chat-v1.0.Q4_K_M (de modo a caber em uma pequena configuração de máquina)

Importante:

Usar a versão 3.11.x do Python e com esta criar um ambiente:

$ python3.11 -m venv meuambiente

Para ativar:

$ source meuambiente/bin/activate

E desativar:

$ deactivate

Bibliotecas para o Python (instalar todas com o pip)

* langchain-community 
* langchain-core 
* langchain-text-splitters 
* langchain
* langchain-huggingface
* faiss-cpu
* transformers
* sentence-transformers
* torch
* llama-cpp-python
* bitsandbytes

## Passos da transformação
Conforme o RAG, são os seguintes passos executados:
1. Carregar e dividir o contexto.
2. Gerar os embeddings e construir o banco de vetores (FAISS).
3. Inicializar o modelo TinyLlama.

## Ativar o chat
O próximo passo é a ativação do Chat de perguntas em si, no qual o usuário poderá realizar quaisquer perguntas sobre o contexto.

Todas as perguntas capturadas são processadas através do contexto informado, que nessa versão está disponível no arquivo na pasta "meusDocs" que contém a especificação de ferramentas de IA e suas utilizações.

Assim uma vez ativo o modelo pode responder perguntas como:
* Quero conversar naturalmente pelo telefone, o que posso usar?
* Qual é o assistente para o Telegram?
* Como posso gerar música gratuitamente?
* O que uso para edições e composições visuais?
* O que posso usar para criar arte visual?
* O que posso usar para criar logotipos?

Ao informar "sair" o ciclo será interrompido.