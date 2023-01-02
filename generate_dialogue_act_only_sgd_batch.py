import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
from utils.args_parser import ArgsParser
from metric import DiaactF1
from tqdm import tqdm

import json
import ipdb
import sys
import os

opt = ArgsParser().parse()
opt.multiwoz_version = "2.1"
opt.use_action = True
opt.use_knowledge = True
opt.context_knowledge = True
opt.lexical = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HISTORY_LEN = None
# USE_ORACLE_BELIEF = True
USE_ORACLE_BELIEF = True
# USE_ORACLE_ACTION = False
USE_ORACLE_ACTION = False
USE_DB_SEARCH = False  # doesn't matter. Only used for response
USE_DYNAMIC_DB = False
# EVAL_SPLIT = 'test'
EVAL_SPLIT = opt.split_set
decoding = opt.decoding
opt_delex = ArgsParser().parse()
opt_delex.multiwoz_version = "2.1"
bz = 32

assert EVAL_SPLIT in [
    "train",
    "validation",
    "test",
], "EVAL_SPLIT should be ither `train`, `validation`, `test`."


def get_action(sent):
    if "<|action|>" not in sent:
        return []
    elif "<|belief|>" in sent:
        tmp = (
            sent.split("<|belief|>")[-1]
            .split("<|response|>")[0]
            .split("<|action|>")[-1]
            .strip()
        )
    elif "<|action|>" in sent:
        tmp = sent.split("<|response|>")[0].split("<|action|>")[-1].strip()
    else:
        return []
    tmp = tmp.strip(" .,")
    tmp = tmp.replace("<|endofaction|>", "")
    tmp = tmp.replace("<|endoftext|>", "")
    action = tmp.split(",")
    new_action = []
    for act in action:
        if act == "":
            continue
        act = act.strip(" .,")
        if act not in new_action:
            act = act.split()
            if len(act) == 3:
                new_action.append(tuple(act))
            elif len(act) == 2:
                new_action.append((act[0], act[1], "none"))
            else:
                new_action.append(("none", act[0], "none"))

    return new_action


model_checkpoint = opt.checkpoint
parent_dir = "output/sgd-distilgpt2"
ckpts = [
    os.path.join(parent_dir, name)
    for name in os.listdir(parent_dir)
    if name.startswith("checkpoint")
]
data = (
    open(f"resources/sgd_0_1_simpletod/{EVAL_SPLIT}_sgd", "r")
    .read()
    .splitlines()
)
f1s = []
inputs = []
for model_checkpoint in ckpts:
    print(f"evaluating {model_checkpoint}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
    model.eval()
    model.to(DEVICE)
    break_tokens = tokenizer.encode(tokenizer.eos_token)
    MAX_LEN = model.config.n_ctx

    if len(inputs) == 0:
        # tokenize the input first
        for i in tqdm(range(0, len(data), bz), desc="Processing inputs"):
            contexts = []
            act_trues = []
            for text in data[i : i + bz]:
                act_trues.append(sorted(get_action(text)))
                contexts.append(text.split("<|action|>")[0].strip())
            input = tokenizer(
                contexts, return_tensors="pt", padding=True, truncation=True
            ).to(DEVICE)
            inputs.append(
                {
                    "input_ids": input["input_ids"],
                    "attention_mask": input["attention_mask"],
                    "contexts": contexts,
                    "act_trues": act_trues,
                }
            )

    outputs = []
    metric = DiaactF1()
    pbar = tqdm(inputs, desc="Testing")
    for i in pbar:
        outputs2 = model.generate(
            input_ids=i["input_ids"],
            attention_mask=i["attention_mask"],
            do_sample=False,
            max_length=MAX_LEN,
            pad_token_id=0,
        )
        predicted_texts = tokenizer.batch_decode(
            outputs2, skip_special_tokens=True
        )
        for j, t in enumerate(predicted_texts):
            act_pred = sorted(get_action(t))
            act_true = i["act_trues"][j]
            outputs.append(
                {
                    "context": i["contexts"][j],
                    "act_pred": act_pred,
                    "act_true": act_true,
                    "generated": t,
                }
            )
            metric(act_pred, act_true)
        f1 = metric.compute()
        pbar.set_postfix({"dia_f1": f"{f1:.3f}"})

    outputs = {"dia_f1": f"{metric.compute():.4f}", "outputs": outputs}
    save_path = os.path.join(model_checkpoint, f"{EVAL_SPLIT}_outputs.json")
    with open(save_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"outputs saved in {save_path}")
    f1s.append((metric.compute(), model_checkpoint))
print(max(f1s))
