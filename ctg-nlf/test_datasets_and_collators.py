from datasets_and_collators import PromptDataset, PromptCollator
from policy import T5Policy
import torch
from torch.utils.data import DataLoader

root_data_path = "/cluster/project/sachan/sauc/MultiTask-CTG-NLF/tasks/qa_feedback/data"
train_dataset = PromptDataset(path=root_data_path+"/train.json")
print(len(train_dataset))
dev_dataset = PromptDataset(path=root_data_path+"/dev.json")
print(len(dev_dataset))
test_dataset = PromptDataset(path=root_data_path+"/test.json")
print(len(test_dataset))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
policy = T5Policy(
    model_ckpt="t5-large", 
    device=device,
    temperature=1.0,
    nlf_cond=True
)
tokenizer = policy.tokenizer

prompt_collator = PromptCollator(tokenizer=tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=False, drop_last=True, collate_fn=prompt_collator)
for batch in train_dataloader:
    print(batch)
    import pdb 
    pdb.set_trace()
    break