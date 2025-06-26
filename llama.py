from llama_cpp import Llama

model_path = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

llama = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=0, verbose=False) #, n_threads=4, n_batch=16)

prompt = "Who is Agent of IA?"

response = llama(
    prompt, 
    max_tokens=150,
    stop=["<|im_end|>", "</s>"], # stop generation when the token is found
    echo=False, # no return the prompt in the response
)

generated_text = response["choices"][0]["text"].strip()

print(f"IA: {generated_text}")