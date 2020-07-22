import torch
import torch.nn.functional as F
import os
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import transformers
from transformers.file_utils import WEIGHTS_NAME
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from model import BertNER
import config
from datasets import NerProcessor, convert_examples_to_features
from engine import train_fn

def run_train():
    data_dir = config.DATA_DIR
    nerProcessor = NerProcessor()
    train_example = nerProcessor.get_train_examples(data_dir)
    label_list = nerProcessor.get_labels()
    tokenizer = transformers.BertTokenizer.from_pretrained(config.BERT_TOKENIZER_PATH)
    train_features = convert_examples_to_features(train_example, label_list, config.MAX_SEQ_LEN, tokenizer)

    # input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    # attention_mask = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    # token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)
    # label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)

    input_ids = torch.tensor([f["input_ids"] for f in train_features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in train_features], dtype=torch.long)
    token_type_ids = torch.tensor([f["token_type_ids"] for f in train_features], dtype=torch.long)
    label_ids = torch.tensor([f["label_ids"] for f in train_features])
    label_ids = F.one_hot(label_ids)
    label_ids = torch.tensor(label_ids.numpy(), dtype=torch.float)

    train_dataset = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids)
    sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config.TRAIN_BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertNER(config.BERT_MODEL_PATH, len(label_list)+1)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    num_training_step = len(train_dataset) // config.TRAIN_BATCH_SIZE * config.TRAIN_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_step
    )

    for epoch in range(config.TRAIN_EPOCHS):
        train_fn(model, device, train_dataloader, optimizer, scheduler)

        model_to_save = model.module if hasattr(model, "module") else model
        model_save_path = os.path.join(f"{config.BERT_OUTPUT}/{epoch+1}", WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), model_save_path)
        tokenizer.save_vocabulary(f"{config.BERT_OUTPUT}/{epoch+1}/vocab.txt")

    model_to_save = model.module if hasattr(model, "module") else model
    model_save_path = os.path.join(config.BERT_OUTPUT, WEIGHTS_NAME)
    torch.save(model_to_save.state_dict(), model_save_path)
    tokenizer.save_vocabulary(f"{config.BERT_OUTPUT}/vocab.txt")

if __name__ == '__main__':
    run_train()