import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from dotenv import load_dotenv
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoModel, AutoTokenizer, PretrainedConfig

load_dotenv()
classifier = os.getenv("PROMPT_CLASSIFIER_MODEL")

if not classifier:
    raise RuntimeError("Failed to load model name")  # raise needs an exception instance, not a bare string


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super().__init__()

        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map

        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)

        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()

            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(value, 3) for value in sublist] for sublist in top2_prob
            ]

            for sublist_idx, sublist in enumerate(top2_prob_rounded):
                if sublist[1] < 0.1:
                    top2_strings[sublist_idx][1] = "NA"

            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]
            return task_type_1, task_type_2, task_type_prob

        preds = torch.softmax(preds, dim=1)
        weights = np.array(self.weights_map[target])
        weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
        scores = weighted_sum / self.divisor_map[target]
        scores = [round(value, decimal) for value in scores]
        if target == "number_of_few_shots":
            scores = [x if x >= 0.05 else 0 for x in scores]
        return scores

    def process_logits(self, logits):
        result = {}

        task_type_results = self.compute_results(logits[0], target="task_type")
        result["task_type_1"], result["task_type_2"], result["task_type_prob"] = task_type_results

        for i, target in enumerate(
            [
                "creativity_scope",
                "reasoning",
                "contextual_knowledge",
                "number_of_few_shots",
                "domain_knowledge",
                "no_label_reason",
                "constraint_ct",
            ],
            start=1,
        ):
            result[target] = self.compute_results(logits[i], target=target)

        result["prompt_complexity_score"] = [
            round(
                0.35 * creativity
                + 0.25 * reasoning
                + 0.15 * constraint
                + 0.15 * domain_knowledge
                + 0.05 * contextual_knowledge
                + 0.05 * few_shots,
                5,
            )
            for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge, few_shots in zip(
                result["creativity_scope"],
                result["reasoning"],
                result["constraint_ct"],
                result["domain_knowledge"],
                result["contextual_knowledge"],
                result["number_of_few_shots"],
            )
        ]
        return result

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled = self.pool(last_hidden_state, attention_mask)
        logits = [self.heads[k](mean_pooled) for k in range(len(self.target_sizes))]
        return self.process_logits(logits)




@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PretrainedConfig.from_pretrained(classifier)
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(classifier)
    global model
    model = CustomModel(
        target_sizes=config.target_sizes,
        task_type_map=config.task_type_map,
        weights_map=config.weights_map,
        divisor_map=config.divisor_map,
    ).from_pretrained(classifier)
    model.eval()
    
    yield
    
    
app = FastAPI(lifespan=lifespan)


def main() -> None:
    uvicorn.run("classifier.app:app", host="0.0.0.0", port=8000, reload=False)


class PromptRequest(BaseModel):
    prompt: str
    
@app.get("/home")
def home():
    return {"message": "Hello World"}
    
@app.post("/classify")
def classify(req: PromptRequest):

    encoded_prompt = tokenizer(
        req.prompt,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )

    with torch.no_grad():
        result = model(encoded_prompt)

    return result