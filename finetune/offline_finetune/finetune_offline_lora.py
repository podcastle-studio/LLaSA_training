import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataclasses import dataclass, field
from typing import Optional
import sys
import transformers
import wandb
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np
import random
from datasets import load_dataset
from functools import partial
from transformers import TrainingArguments

from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F

from functools import partial
from torch.utils.data import DataLoader
import torch.nn.functional as F
from peft import PeftModel, PeftConfig

from transformers import TrainerCallback

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# class WeightedLossTrainer(Trainer):
#     def __init__(self, *args, yes_id=9891, no_id=2201, yes_weight=2.670, no_weight=0.616, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.yes_id = yes_id
#         self.no_id = no_id
#         self.yes_weight = yes_weight
#         self.no_weight = no_weight

#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         labels = inputs["labels"]

#         outputs = model(**inputs)
#         logits = outputs.logits

#         # Create weight tensor with float32 dtype to avoid dtype mismatch
#         # weight = torch.ones(logits.size(-1), device=logits.device, dtype=logits.dtype)
#         # weight[self.yes_id] = self.yes_weight
#         # weight[self.no_id] = self.no_weight

#         # loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=weight)
#         # loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
#         loss = outputs.loss

#         # Optional debug print
#         # print(f"Loss: {loss.item()}")

#         return (loss, outputs) if return_outputs else loss

class ManualEvalCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, eval_every_n_steps=500, batch_size=1):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.eval_every_n_steps = eval_every_n_steps
        self.batch_size = batch_size

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 and self.eval_dataset is not None:
            model = kwargs["model"]
            device = model.device
            eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=default_data_collator,
                num_workers=0,
                pin_memory=False,
            )
            eval_loss, eval_metrics = run_manual_evaluation(model, eval_dataloader, self.tokenizer, device)
            print(f"\n[Manual Eval] Step {state.global_step} - Eval loss: {eval_loss:.4f}, Metrics: {eval_metrics}")

            #if args.report_to == "wandb":
            import wandb
            wandb.log({"eval_loss": eval_loss, **{f"eval_{k}": v for k, v in eval_metrics.items()}}, step=state.global_step)



def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred

    answer_token_id = tokenizer.convert_tokens_to_ids("<|ANSWER|>")
    
    pred_ids = np.argmax(predictions, axis=-1)  # Use NumPy instead of torch
    labels = np.array(labels)

    def extract_label(ids):
        try:
            answer_pos = np.where(ids == answer_token_id)[0]
            if len(answer_pos) == 0 or answer_pos[0] + 1 >= len(ids):
                return -100
            return ids[answer_pos[0] + 1]
        except Exception:
            return -100

    pred_labels = [extract_label(seq) for seq in pred_ids]
    true_labels = [extract_label(seq) for seq in labels]

    pos_label_id = 9891
    binary_preds = [1 if p == pos_label_id else 0 for p in pred_labels]
    binary_refs = [1 if r == pos_label_id else 0 for r in true_labels]

    filtered = [(p, l) for p, l in zip(binary_preds, binary_refs) if l != -100]
    if not filtered:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    preds, refs = zip(*filtered)

    acc = accuracy_score(refs, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(refs, preds, average='binary')
    
    # Manual cleanup
    del predictions, labels, pred_ids, pred_labels, true_labels
    import gc; gc.collect()

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def run_manual_evaluation(model, dataloader, tokenizer, device):
    model.eval()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    total_batches = 0

    total_correct = 0
    total_count = 0

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    answer_token_id = tokenizer.convert_tokens_to_ids("<|ANSWER|>")

    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits  # (bs, seq_len, vocab_size)
            
            # Shift so that tokens < n predict n
            batch["labels"] = nn.functional.pad(batch["labels"], (0, 1), value=-100)
            batch["labels"] = batch["labels"][..., 1:].contiguous()
        
            loss = loss_fct(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
            total_loss += loss.item()
            total_batches += 1

            pred_ids = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for p_seq, l_seq in zip(pred_ids, labels):
                try:
                    answer_pos = np.where(l_seq != -100)[0]
                    if len(answer_pos) == 0 or answer_pos[0] + 1 >= len(l_seq):
                        continue

                    pred_label = p_seq[answer_pos[0]]
                    true_label = l_seq[answer_pos[0]]

                    if true_label == -100:
                        continue

                    if pred_label == true_label:
                        total_correct += 1

                    # Binary classification metrics
                    if true_label == 9891 and pred_label == 9891:
                        true_positives += 1
                    elif true_label == 2201 and pred_label == 9891:
                        false_positives += 1
                    elif true_label == 9891 and pred_label == 2201:
                        false_negatives += 1

                    total_count += 1
                except Exception:
                    continue

    avg_loss = total_loss / max(total_batches, 1)
    accuracy = total_correct / max(total_count, 1)

    # Avoid divide-by-zero with small epsilon
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_metrics(predictions, labels, tokenizer):
    answer_token_id = tokenizer.convert_tokens_to_ids("<|ANSWER|>")
    pred_ids = np.argmax(predictions, axis=-1)  # (batch_size, seq_len)
    labels = np.array(labels)

    def extract_label(ids, ref = None):
        try:
            if ref is None:
                answer_pos = np.where(ref == answer_token_id)[0]
            else:
                answer_pos = np.where(ids == answer_token_id)[0]
            if len(answer_pos) == 0 or answer_pos[0] + 1 >= len(ids):
                return -100
            return ids[answer_pos[0] + 1]
        except Exception:
            return -100

    pred_labels = [extract_label(seq, labels[i]) for i, seq in enumerate(pred_ids)]
    true_labels = [extract_label(seq) for seq in enumerate(labels)]

    pos_label_id = 9891
    binary_preds = [1 if p == pos_label_id else 0 for p in pred_labels]
    binary_refs = [1 if r == pos_label_id else 0 for r in true_labels]

    filtered = [(p, l) for p, l in zip(binary_preds, binary_refs) if l != -100]
    if not filtered:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    preds, refs = zip(*filtered)

    acc = accuracy_score(refs, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(refs, preds, average='binary')

    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


@dataclass
class ModelArguments:
    llm_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for the model."})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Root path to the memmap data."})

 
@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    
    report_to: Optional[str] = field(
        default=None, metadata={"help": "The integration to report the results and logs to."}
    )
    
    run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the run for logging."}
    )
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(
        default=None, metadata={"help": "The integration to report the results and logs to."}
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the run for logging."}
    )
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The learning rate scheduler to use."})

class TTSDataset(Dataset):
    def __init__(self, data_path, split, tokenizer):

        memmap_path = os.path.join(data_path, f'{split}_input_ids.memmap')
        shape_path = os.path.join(data_path, f'{split}_input_ids_shape.npy')

        self.input_ids = np.memmap(memmap_path, dtype='int32', mode='r', shape=tuple(np.load(shape_path)))
        self.length = self.input_ids.shape[0]
        self.pad_token_id = tokenizer.pad_token_id   
        self.tokenizer = tokenizer
        special_tokens = [
        # '<|TEXT_GENERATION_START|>', '<|TEXT_GENERATION_END|>',
        '<|TEXT_UNDERSTANDING_START|>', '<|TEXT_UNDERSTANDING_END|>',
        # '<|SPEECH_GENERATION_START|>', '<|SPEECH_GENERATION_END|>',
        '<|SPEECH_UNDERSTANDING_START|>', '<|SPEECH_UNDERSTANDING_END|>',
        '<|ANSWER|>'
    ]
        tokenizer.add_tokens(special_tokens)
        self.text_understanding_start_id, self.text_understanding_end_id, self.speech_understanding_start_id, self.speech_understanding_end_id, self.answer_start_id = tokenizer.convert_tokens_to_ids(special_tokens)

        self.max_length = 790 + 43
        self.ignore_index = -100  

    def __len__(self):
        return self.length

    def replace_tagged_token(self, token_list, target_token, new_sequence):
        idx = token_list.index(target_token)
        return token_list[:idx] + list(new_sequence) + token_list[idx+1:]

    def pad_sequence(self, sequence, max_length, value=0):
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = torch.full((max_length - len(sequence),), value, dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=0)

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        speech_understanding_end_positions = (input_ids == self.speech_understanding_end_id).nonzero(as_tuple=True)[0]
        speech_understand_end_idx = speech_understanding_end_positions[0].item()

        text_speech_sequence = input_ids[:speech_understand_end_idx + 1]
       
        answer_start_positions = (input_ids == self.answer_start_id).nonzero(as_tuple=True)[0]
        answer_start_idx = answer_start_positions[0].item()
        
        # speech_understand_end_idx = speech_understanding_end_positions[0].item()
        answer_sequence = input_ids[answer_start_idx:]

        chat = [
            {"role": "user", "content": "Detect hallucination in the speech:<|TEXT_UNDERSTANDING_START|>"},
            {"role": "assistant", "content": "<|ANSWER|>"}
        ]
        ids = self.tokenizer.apply_chat_template(chat, tokenize=True)

        ids = self.replace_tagged_token(ids, self.text_understanding_start_id, text_speech_sequence)
        ids = self.replace_tagged_token(ids, self.answer_start_id, answer_sequence)

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        try:
            answer_idx_in_input = (input_ids == self.answer_start_id).nonzero(as_tuple=True)[0].item()
            labels[answer_idx_in_input+1:] = input_ids[answer_idx_in_input+1:]
        except Exception as e:
            print(f"maybe Error in speech_gen_idx_in_input: {e}")
            labels = input_ids 

        attention_mask = (input_ids != self.pad_token_id).long()
        labels[input_ids == self.pad_token_id] = self.ignore_index
        #labels[answer_idx_in_input+2] = self.pad_token_id

        input_ids = self.pad_sequence(input_ids, self.max_length, value=self.pad_token_id)
        attention_mask = self.pad_sequence(attention_mask, self.max_length, value=0)
        labels = self.pad_sequence(labels, self.max_length, value=self.ignore_index)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def main():
    # 解析参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        default_config_file = 'finetune/offline_finetune/config_lora.json'
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(default_config_file))
     
    is_main_process = training_args.local_rank in [-1, 0]
    if training_args.report_to == "wandb" and is_main_process:
        wandb.init(
            project="llm_audio_classification",  
            config=training_args.to_sanitized_dict(),
            name=training_args.run_name
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.llm_model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_args.llm_model_name_or_path,
        torch_dtype='auto',
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        model_type="llama"
    )

    lora_config = LoraConfig(
        r=8,                      
        lora_alpha=32,           
        target_modules=["q_proj", "v_proj"],   
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    

    # model.resize_token_embeddings(len(tokenizer))
    # model = PeftModel.from_pretrained(model, "finetuneLogs/results_lora_0.12/checkpoint-15000", inference_mode=False)
    print("PEFT Configs:", model.peft_config)
    model = get_peft_model(model, lora_config)
    #model.peft_config['default'].inference_mode = False
    #print("PEFT Configs:", model.peft_config)

    # update model parameters to require grad
    # for name, param in model.named_parameters():
    #     # LoRA parameters usually have 'lora_' in their name
    #     if "lora_" in name:
    #         param.requires_grad = True

    
    print("LoRA微调模型参数信息：")
    model.print_trainable_parameters()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", trainable_params)

    # 测试获取一个样本，确保数据正确
    train_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='train',
        tokenizer=tokenizer
    )
    _ = train_dataset[0]
    eval_dataset = TTSDataset(
        data_path=data_args.data_path,
        split='val',
        tokenizer=tokenizer
    ) if os.path.exists(os.path.join(data_args.data_path, 'val_input_ids.memmap')) else None
    
    model.resize_token_embeddings(len(tokenizer))
    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[ManualEvalCallback(eval_dataset=eval_dataset, tokenizer=tokenizer, eval_every_n_steps=1000)],  # eval every 500 steps (adjust as you want)
         # Optional: specify a checkpoint to resume training

        #compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
    )

   
    trainer.train(resume_from_checkpoint="finetuneLogs/results_lora_0.12/checkpoint-15000", )
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
