from transformers import AutoTokenizer, AutoModelForCausalLM

path = "/workspace/amit/Jahanvi/Meta-Llama-Guard-2-8B"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype="auto",
    device_map="auto"
)

print("loaded successfully")
