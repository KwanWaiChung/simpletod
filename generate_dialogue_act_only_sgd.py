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
    return new_action


def predict(text: str) -> str:
    indexed_tokens = tokenizer.encode(text)
    if len(indexed_tokens) > MAX_LEN:
        indexed_tokens = indexed_tokens[-1 * MAX_LEN :]

    # Convert indexed tokens in a PyTorch tensor
    tokens_tensor = torch.tensor([indexed_tokens])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to("cuda")
    predicted_index = indexed_tokens[-1]

    with torch.no_grad():
        if decoding == "nucleus":
            sample_output = model.generate(
                tokens_tensor,
                # indexed_tokens,
                do_sample=True,
                max_length=MAX_LEN,
                top_p=0.5,
                top_k=0,
            )
            predicted_text = tokenizer.decode(sample_output[0])
            tmp = " ".join(
                [
                    predicted_text.split("<|endofresponse|>")[0],
                    "<|endofresponse|>",
                ]
            )
            predicted_text = tmp
        elif decoding == "greedy":
            while predicted_index not in break_tokens:
                outputs = model(tokens_tensor)
                predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item()
                indexed_tokens += [predicted_index]

                # sometime model generate repeated actions, we just use truncate actions if this happens
                predicted_text = tokenizer.decode(indexed_tokens)
                if "<|action|>" in predicted_text:
                    generated_actions = (
                        predicted_text.split("<|action|>")[-1]
                        .split("<|endofaction|>")[0]
                        .split(",")
                    )
                    new_actions = []
                    for a in generated_actions:
                        if a in ["", " "]:
                            continue
                        new_actions.append(a.strip())
                    len_actions = len(new_actions)
                    if len(list(set(new_actions))) > len(new_actions) or (
                        len_actions > 10 and not truncate_action
                    ):
                        actions = "<|action|> {} <|endofaction|>".format(
                            " , ".join(list(set(new_actions)))
                        )
                        indexed_tokens = tokenizer.encode(
                            "{} {}".format(
                                predicted_text.split("<|action|>")[0],
                                actions,
                            )
                        )
                        truncate_action = True

                tokens_tensor = torch.tensor([indexed_tokens]).to("cuda")
                if len(indexed_tokens) > MAX_LEN:
                    break
                if tokenizer.decode(indexed_tokens).endswith(
                    "<|endofaction|>"
                ):
                    break

            predicted_text = tokenizer.decode(indexed_tokens)
    return predicted_text


model_checkpoint = opt.checkpoint
parent_dir = "output/sgd-distilgpt2"
ckpts = [
    os.path.join(parent_dir, name)
    for name in os.listdir(parent_dir)
    if name.startswith("checkpoint")
]
f1s = []
for model_checkpoint in ckpts:
    print(f"evaluating {model_checkpoint}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
    model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
    model.eval()
    model.to("cuda")
    break_tokens = tokenizer.encode(tokenizer.eos_token)
    MAX_LEN = model.config.n_ctx

    data = (
        open(f"resources/sgd_0_1_simpletod/{EVAL_SPLIT}_sgd", "r")
        .read()
        .splitlines()
    )
    num_data = len(data)
    model_context = []
    generated = []
    act_pred = []
    act_true = []
    outputs = []
    metric = DiaactF1()
    pbar = tqdm(data, desc="Testing")
    for text in pbar:
        act = get_action(text)
        act_true.append(act)
        input_context = text.split("<|action|>")[0].strip()
        model_context.append(input_context)
        predicted_text = predict(input_context)
        generated.append(predicted_text)
        gen_action = get_action(predicted_text)
        act_pred.append(gen_action)
        outputs.append(
            {
                "context": input_context,
                "act_pred": sorted(gen_action),
                "act_true": sorted(act),
                "generated": predicted_text,
            }
        )
        metric(gen_action, act)
        f1 = metric.compute()
        pbar.set_postfix({"dia_f1": f"{f1:.3f}"})

    outputs = {"dia_f1": f"{metric.compute():.4f}", "outputs": outputs}
    save_path = os.path.join(model_checkpoint, f"{EVAL_SPLIT}_outputs.json")
    with open(save_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"outputs saved in {save_path}")
    f1s.append((metric.compute(), model_checkpoint))
print(max(f1s))
