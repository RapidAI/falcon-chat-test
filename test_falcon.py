from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "/home/znsoft/models/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
	  model_kwargs={"load_in_8bit": True}
)
while(1):
	query=input("please input question:")

	text ="User: {}\nAssistant:\n".format(query) 

	sequences = pipeline(
    text,
    max_length=500,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
	)
	for seq in sequences:
		print(f"Result: {seq['generated_text']}")
