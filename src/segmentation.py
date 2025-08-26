import numpy as np
import torch.nn.functional as F
from constants import SPECIAL_TOKEN_LABEL_ID
from transformers import PreTrainedTokenizerFast


def logistic(x):
    return 1 / (1 + np.exp(-6 * (x - 0.5)))


def ensemble_logits(a, b, axis=0):
    assert a.shape == b.shape
    reshape = [1] * len(a.shape)
    reshape[axis] = -1
    b_factors = logistic(np.linspace(0, 1, len(a))).reshape(*reshape)
    a_factors = 1 - b_factors
    return np.add(a * a_factors, b * b_factors)


def softmax(x):
    return F.softmax(x, dim=-1)  # along labels


def ensemble_results(model_output, segmented, overlap_length=50):
    model_output = softmax(model_output[:, 1:-1, :]).cpu().numpy()
    probs = model_output[0, :, :]
    assert (probs > 0).all() and (probs < 1).all()
    offsets = segmented[0]["offset_mapping"][1:-1]
    for index, segment in enumerate(model_output[1:, :, :]):
        left = probs[:-overlap_length, :]
        right = segment[overlap_length:, :]
        middle = ensemble_logits(
            probs[-overlap_length:, :], segment[:overlap_length, :], axis=0
        )
        probs = np.concatenate([left, middle, right], axis=0)
        offsets = offsets[:-overlap_length]
        offsets.extend(segmented[index + 1]["offset_mapping"][1:-1])
        probs = probs[: len(offsets), :]
    probs = probs[: len(offsets), :]
    assert len(probs) == len(offsets)
    return probs, np.array(offsets)


def merge_examples(*examples):
    assert all(
        "labels" not in e and "tokenized" not in e for e in examples
    ), "Merging should be done before tokenization"
    merged = {"text": "", "entities": []}
    text_offset = 0
    for example in examples:
        merged["text"] += example["text"] + "\n"
        for entity in example["entities"]:
            assert len(entity["offsets"]) == 1, "Only support single token entity"
            start, end = entity["offsets"][0]
            merged["entities"].append(
                {
                    "offsets": [[start + text_offset, end + text_offset]],
                    "type": entity["type"],
                }
            )
        text_offset += len(example["text"]) + 1
    return merged


def segment_tokenized(
    example,
    tokenizer: PreTrainedTokenizerFast,
    max_length,
    overlap,
    add_special_tokens=True,
):
    tokenized = example["tokenized"]
    max_length -= 2 if add_special_tokens else 0
    assert max_length > overlap * 2
    _tokens = tokenized["input_ids"]
    _offsets = tokenized["offset_mapping"]
    if "labels" in tokenized:
        _labels = tokenized["labels"]
    segments = []
    for start_idx in range(0, len(_tokens), max_length - overlap):
        end_idx = start_idx + max_length
        seg_tokens = _tokens[start_idx:end_idx]
        seg_offsets = _offsets[start_idx:end_idx]
        if "labels" in tokenized:
            seg_labels = _labels[start_idx:end_idx]
        if add_special_tokens:
            seg_tokens = (
                [tokenizer.cls_token_id] + seg_tokens + [tokenizer.sep_token_id]
            )
            seg_offsets = [(0, 0)] + seg_offsets + [(0, 0)]
            if "labels" in tokenized:
                seg_labels = (
                    [SPECIAL_TOKEN_LABEL_ID] + seg_labels + [SPECIAL_TOKEN_LABEL_ID]
                )
        segments.append(
            {
                "id": example.get("id"),
                "text_offset": start_idx,
                "input_ids": seg_tokens,
                "offset_mapping": seg_offsets,
                **({"labels": seg_labels} if "labels" in tokenized else {}),
            }
        )
    return segments


def batched_segment_tokenized(
    examples,
    tokenizer: PreTrainedTokenizerFast,
    max_length,
    overlap,
    add_special_tokens=True,
):
    segments = {
        "id": [],
        "text_offset": [],
        "input_ids": [],
        "offset_mapping": [],
        "labels": [],
    }
    for i in range(len(examples["id"])):
        example = {k: examples[k][i] for k in examples.keys()}
        # print(example, flush=True)
        for segment in segment_tokenized(
            example, tokenizer, max_length, overlap, add_special_tokens
        ):
            segments["id"].append(segment["id"])
            segments["text_offset"].append(segment["text_offset"])
            segments["input_ids"].append(segment["input_ids"])
            segments["offset_mapping"].append(segment["offset_mapping"])
            segments["labels"].append(segment["labels"])
    return segments
