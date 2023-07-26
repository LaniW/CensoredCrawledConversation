from transformers import AutoModelForCausalLM, AutoModelForSeq2Seq, AutoTokenizer
import torch

model_name = "facebook/xlm-v-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_name)
print("====chatting 5 times with nucleus & top-k sampling & tweaking temperature & multiple sentences====")
for step in range(5):
    text = input(">> You:")
    input_ids = tokenizer.encode(tokenizer.eos_token + text, return_tensors="pt")
    bot_input_ids = torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
    chat_history_ids_list = model.generate(
        bot_input_ids,
        max_length=1000,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.75,
        num_return_sequences=5,
        pad_token_id=tokenizer.eos_token_id
    )
    for i in range(len(chat_history_ids_list)):
      output = tokenizer.decode(chat_history_ids_list[i][bot_input_ids.shape[-1]:], skip_special_tokens=True)
      print(f"T5 {i}: {output}")
    choice_index = int(input("Choose the response you want for the next input: "))
    chat_history_ids = torch.unsqueeze(chat_history_ids_list[choice_index], dim=0)