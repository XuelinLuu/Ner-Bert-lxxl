import torch.nn as nn
import transformers

class BertNER(nn.Module):
    def __init__(self, bert_path, num_labels):
        super(BertNER, self).__init__()
        self.bert_path = bert_path
        self.num_labels = num_labels
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, self.num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        output = bert_output[0]
        output = self.dropout(output)
        return self.softmax(self.classifier(output))