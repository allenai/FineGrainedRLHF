from policy import T5Policy
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

policy = T5Policy(
    model_ckpt="t5-large", 
    device=device,
    temperature=1.0,
    nlf_cond=True
)

tokenizer = policy.tokenizer

feedbacks = ["This is a very toxic response.", "This was very polite and unhurtful.", "You shouldn't be using those harmful words."]
questions = ["Every Christian should be killed.", "We should advocate for love and respect, regardless of everyone's individual beliefs.", "Nigga step out of my way!"]
print(feedbacks)
print(questions)
inputs = [tokenizer.feedback_prefix + feedback + " " + tokenizer.prompt_prefix + question for feedback, question in zip(feedbacks, questions)]
print(inputs)

inputs_dict = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs_dict)
prompts_input_ids = inputs_dict["input_ids"].to(device)
prompts_attention_mask = inputs_dict["attention_mask"].to(device)
outputs_dict = policy.sample(prompts_input_ids=prompts_input_ids, prompts_attention_mask=prompts_attention_mask, 
                             do_sample=True, top_p=0.9)
print(outputs_dict)
outputs_metrics = policy.forward_pass(prompts_input_ids=prompts_input_ids,
                                      prompts_attention_mask=prompts_attention_mask,
                                      generated_input_ids=outputs_dict["generated_input_ids"],
                                      generated_attention_mask=outputs_dict["generated_attention_mask"])
print(outputs_metrics)

