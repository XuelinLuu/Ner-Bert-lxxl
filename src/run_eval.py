import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import transformers


from model import BertNER
import config
from datasets import NerProcessor, convert_examples_to_features
from engine import eval_fn

def run_eval():
    data_dir = config.DATA_DIR
    nerProcessor = NerProcessor()
    test_example = nerProcessor.get_valid_examples(data_dir)
    label_list = nerProcessor.get_labels()
    tokenizer = transformers.BertTokenizer.from_pretrained(f"{config.BERT_OUTPUT}/vocab.txt")
    test_features = convert_examples_to_features(test_example, label_list, config.MAX_SEQ_LEN, tokenizer)

    input_ids = torch.tensor([f["input_ids"] for f in test_features], dtype=torch.long)
    attention_mask = torch.tensor([f["attention_mask"] for f in test_features], dtype=torch.long)
    token_type_ids = torch.tensor([f["token_type_ids"] for f in test_features], dtype=torch.long)
    label_ids = torch.tensor([f["label_ids"] for f in test_features])
    label_ids = F.one_hot(label_ids)
    label_ids = torch.tensor(label_ids.numpy(), dtype=torch.float)
    real_len = torch.tensor([f['real_len'] for f in test_features])

    eval_dataset = TensorDataset(input_ids, attention_mask, token_type_ids, label_ids, real_len)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.EVAL_BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertNER(config.BERT_OUTPUT, len(label_list)+1)
    model.to(device)

    eval_fn(model, device, eval_dataloader, config.EVAL_OUTPUT)


if __name__ == '__main__':
    run_eval()