import json
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question_text, context_text, label=None):
        """Constructs a InputExample.
        Args:
            example_id: Unique id for the example.
            question_text: list of str. The un_tokenized text of query.
            context_text: string. The un_tokenized text of context.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question_text = question_text
        self.context_text = context_text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def create_examples(input_file):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            context_text = paragraph["context"]
            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                is_impossible = qa["is_impossible"]
                if is_impossible:
                    label = True
                else:
                    label = False
                examples.append(InputExample(example_id=qas_id, question_text=question_text,
                                             context_text=context_text, label=label))
    return examples


def convert_examples_to_features(examples, tokenizer, max_query_length, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []
    for example_index, example in tqdm(enumerate(examples)):
        query_tokens = tokenizer.tokenize(example.question_text)
        context_tokens = tokenizer.tokenize(example.context_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_context = max_seq_length - len(query_tokens) - 3

        if len(context_tokens) > max_tokens_for_context:
            context_tokens = context_tokens[0:max_tokens_for_context]

        tokens = [tokenizer.cls_token] + query_tokens + [tokenizer.sep_token] + context_tokens + [tokenizer.sep_token]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(context_tokens) + 1)

        # Zero-pad up to the sequence length.
        padding = [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))

        input_mask += [0] * (max_seq_length - len(input_ids))
        segment_ids += [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        label_ids = int(example.label)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % unique_id)
            logger.info("tokens: {}".format(' '.join(tokens)))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            if is_training:
                if label_ids == 1:
                    logger.info("impossible example: %d", label_ids)
                elif label_ids == 0:
                    logger.info("possible example: %d", label_ids)
                else:
                    logger.warning("label should be 0 or 1")

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids)
        )
        unique_id += 1
    return features


def get_acc(preds, labels):
    acc = (preds == labels).mean()
    return {"answerable_acc": acc}
