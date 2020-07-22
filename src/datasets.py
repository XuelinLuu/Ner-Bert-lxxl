import os


class NERExample:
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class NERFeature:
    def __init__(self, input_ids, attention_mask, token_type_ids, label_ids):
        self.input_ids = input_ids
        self.attention_mask = list(attention_mask),
        self.token_type_ids = list(token_type_ids),
        self.label_ids = label_ids


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError

    def get_valid_examples(self, data_dir):
        raise NotImplementedError

    def get_test_examples(self, data_dir):
        raise NotImplementedError

    @classmethod
    def _read_data(cls, data_file):
        lines = []
        with open(data_file, "r", encoding="utf-8") as f:
            words, labels = [], []
            for line in f.readlines():
                if line == " " or line == "\n" or line == "-DOCSTART-":
                    if words != []:  # 两个空行时防止添加过多样例
                        lines.append({"words": words, "labels": labels})
                        words, labels = [], []
                else:
                    try:
                        word_label = line.split(" ")
                        words.append(word_label[0])
                        if len(word_label) > 1:
                            labels.append(word_label[1].rstrip("\n"))
                        else:
                            labels.append("O")
                    except:
                        continue
            if words:
                lines.append({"words": words, "labels": labels})
        return lines


class NerProcessor(DataProcessor):
    def __init__(self):
        self.labels = ['B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                       'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                       'O', 'S-NAME', 'S-ORG', 'S-RACE']

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.char.bmes")),
            "train"
        )

    def get_valid_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "dev.char.bmes")),
            "valid"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.char.bmes")),
            "test"
        )

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, type_str):
        '''
        :param lines: [{"words": words, "labels": labels}.......]
        :param type_str:
        :return:
        '''
        examples = []
        for idx, line in enumerate(lines):
            words, labels = line["words"], line["labels"]
            guid = f"{type_str}-{idx}"
            labels_ner = []
            for label in labels:
                if 'M-' in label:
                    if label.replace('M-', 'I-') in self.labels:
                        labels_ner.append(label.replace('M-', 'I-'))
                    else:
                        labels_ner.append("O")
                elif 'E-' in label:
                    if label.replace('E-', 'I-') in self.labels:
                        labels_ner.append(label.replace('E-', 'I-'))
                    else:
                        labels_ner.append("O")
                else:
                    if label in self.labels:
                        labels_ner.append(label)
                    else:
                        labels_ner.append("")
            examples.append(
                NERExample(
                    guid=guid,
                    words=words,
                    labels=labels_ner
                )
            )
        return examples


def convert_examples_to_features(examples, labels_list, max_seq_len, tokenizer):
    features = []
    label_to_ids = {label: label_i for label_i, label in enumerate(labels_list)}
    for idx, data in enumerate(examples):
        words, labels = data.words, data.labels
        label_ids = [label_to_ids[l] for l in labels]
        if len(words) > max_seq_len - 2:
            words = words[:max_seq_len - 2]
            label_ids = label_ids[:max_seq_len - 2]

        words = ['[CLS]'] + words + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(words)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)
        label_ids = [label_to_ids['O']] + label_ids + [label_to_ids['O']]

        padding_len = int(max_seq_len - len(input_ids))

        input_ids = input_ids + [0] * padding_len
        attention_mask = attention_mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        # len(labels_list) 是label_ids中没有的数字
        label_ids.extend([len(labels_list)] * padding_len)

        # input_ids = torch.tensor(input_ids, dtype=torch.long)
        # attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        # token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        # label_ids = torch.tensor(label_ids)
        # label_ids = F.one_hot(label_ids, len(labels_list) + 2)

        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label_ids": label_ids,
            "real_len": max_seq_len - padding_len
        })
    return features
