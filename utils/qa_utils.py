from tqdm.auto import tqdm
import datasets
import collections
import numpy as np
import random
import re, string, sys
from collections import Counter

##Heyperparameters
n_best = 20
max_answer_length = 30

def preprocess_training_examples(examples, tokenizer, max_length, stride):
    questions = [q.strip() for q in examples["question"]]
    
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples, tokenizer, max_length, stride):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    
    return inputs

def preprocess_overlap_examples(examples, tokenizer, max_length, stride):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs["offset_mapping"].copy()
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    overlap_score = []
    x = 0
    for i, offset in enumerate(offset_mapping):
        #print("i is : ", i)
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        ids = inputs.input_ids[x]
        x = x + 1


        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1
        context_id = ids[context_start : context_end]
        context_length = context_end - context_start + 1
        #print(context_id)
        context_id = np.asarray(context_id)
        question_id = ids[1:context_start-1]
        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
            center_answer = -1
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)
            start = idx-1
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)
            end = idx + 1
            center_answer = int((start + end) / 2)

        score = []
        question_id = list(set(question_id))
        if center_answer != -1 and context_length > 1:
            for id in question_id:
                #print("id is : ", id)
                indx = np.where(context_id == id)[0]
                tmp = []
                if len(list(indx)) > 0:
                    for j in list(indx):
                        #print(j)
                        tmp.append((abs(j + context_start - center_answer)))
                    score.append(np.min(tmp))
                    #score.append(min(tmp))

            if len(score) == 0:
                score.append(400)
        else:
            score.append(400)

        overlap_score.append(np.mean(score))

    example_ids = []
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]


    inputs["example_id"] = example_ids
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs['overlap'] = overlap_score
    return inputs

from tqdm.auto import tqdm


def compute_metrics(start_logits, end_logits, features, examples, metric):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)



## Evaluation metric impelementation for QA task

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def evaluate(dataset, predictions):
    exact_match = total = 0
    f1 = []
    exact_match =[]
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = [x["text"] for x in qa["answers"]]
                prediction = predictions[qa["id"]]
                #exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1.append(metric_max_over_ground_truths(f1_score, prediction, ground_truths))
                exact_match.append(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))

    # exact_match = 100.0 * exact_match / total
    #print(total)
    #print("len f1 : ",len(f1))
    #f1 = 100.0 * f1 / total

    return {"exact_match": exact_match, "f1": f1}

_CITATION = """\
@inproceedings{Rajpurkar2016SQuAD10,
  title={SQuAD: 100, 000+ Questions for Machine Comprehension of Text},
  author={Pranav Rajpurkar and Jian Zhang and Konstantin Lopyrev and Percy Liang},
  booktitle={EMNLP},
  year={2016}
}
"""

_DESCRIPTION = """
This metric wrap the official scoring script for version 1 of the Stanford Question Answering Dataset (SQuAD).
Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by
crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span,
from the corresponding reading passage, or the question might be unanswerable.
"""

_KWARGS_DESCRIPTION = """
Computes SQuAD scores (F1 and EM).
Args:
    predictions: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair as given in the references (see below)
        - 'prediction_text': the text of the answer
    references: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair (see above),
        - 'answers': a Dict in the SQuAD dataset format
            {
                'text': list of possible texts for the answer, as a list of strings
                'answer_start': list of start positions for the answer, as a list of ints
            }
            Note that answer_start values are not taken into account to compute the metric.
Returns:
    'exact_match': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
Examples:
    >>> predictions = [{'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
    >>> references = [{'answers': {'answer_start': [97], 'text': ['1976']}, 'id': '56e10a3be3433e1400422b22'}]
    >>> squad_metric = datasets.load_metric("squad")
    >>> results = squad_metric.compute(predictions=predictions, references=references)
    >>> print(results)
    {'exact_match': 100.0, 'f1': 100.0}
"""

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Custom_Squad(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {"id": datasets.Value("string"), "prediction_text": datasets.Value("string")},
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
            codebase_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
            reference_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
        )

    def _compute(self, predictions, references):
        pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
        #overlap_dict = {prediction["id"]: prediction["overlap"] for prediction in predictions}
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [{"text": answer_text} for answer_text in ref["answers"]["text"]],
                                "id": ref["id"],

                            }
                            for ref in references
                        ]
                    }
                ]
            }
        ]
        # print(len(dataset))
        # print(len(pred_dict))
        score = evaluate(dataset=dataset, predictions=pred_dict)
        return score



metric = Squad()
def compute_metrics_persample(start_logits, end_logits, features, examples):
    global overlap_t
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    overlap_t = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []


        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]
            tmp_over = features[feature_index]["overlap"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                        "overlap": tmp_over
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
            overlap_t.append(best_answer["overlap"])
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})
            overlap_t.append(-100)

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    # print(len(theoretical_answers))
    # print(len(predicted_answers))
    # print(len(overlap_t))
    return metric.compute(predictions=predicted_answers, references=theoretical_answers), overlap_t
