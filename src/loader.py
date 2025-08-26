from collections import defaultdict
from functools import cached_property
from os import cpu_count
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import snapshot_download
from loguru import logger
from thefuzz import fuzz
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer)
from transformers import \
    DataCollatorForTokenClassification as \
    BaseDataCollatorForTokenClassification
from transformers import PreTrainedTokenizerFast

from categories import load_bigbio_category_mapping
from constants import MODELS_BASEDIR, SPECIAL_TOKEN_LABEL_ID


def map_entity_types(example, type_mapping):
    for entity in example["entities"]:
        entity["type"] = type_mapping.get(entity["type"])
    return example


def map_ds_entity_types(ds, bigbio_name):
    category_mapping = load_bigbio_category_mapping()
    mapping = {
        k: v.replace(" ", "_").upper()
        for k, v in category_mapping[bigbio_name].items()
        if v != "Other or Mixed"
    }
    return ds.map(
        map_entity_types,
        fn_kwargs={"type_mapping": mapping},
        batched=False,
        num_proc=cpu_count(),
        load_from_cache_file=False,
    )


class EntityLabels:
    LABEL_O_ID = 0

    def __init__(self, names=None) -> None:
        self._names_map: dict[str, int] = {}
        names = list(sorted(set(names)))
        for name in names or []:
            self.add(name)

    def __hash__(self):
        return hash(tuple(self.names))

    def __repr__(self):
        return f"EntityLabels({self.names})"

    @classmethod
    def from_dataset(cls, ds):
        if isinstance(ds, DatasetDict):
            ds = concatenate_datasets(ds.values())
        return cls([x["type"] for x in sum(ds["entities"], [])])

    def __getitem__(self, key: str) -> int:
        if key == "O":
            return self.LABEL_O_ID
        return self._names_map[key]

    def get(self, key: str) -> int:
        if key not in self._names_map:
            self.add(key)
        return self[key]

    # @staticmethod
    # def normalize(key: str) -> str:
    #     # return key.replace(" ", "_").upper()
    #     return key

    def add(self, key: str) -> int:
        self._names_map[key] = len(self._names_map)

    @property
    def names(self):
        return list(self._names_map.keys())

    @staticmethod
    def get_iob_label_id(label_id: int, is_begin: bool) -> int:
        iob_label_id = label_id * 2 + 1
        if not is_begin:
            iob_label_id += 1
        return iob_label_id

    def __len__(self):
        return len(self.label2id)

    @cached_property
    def label2id(self):
        mapping = {"O": self.LABEL_O_ID}
        for name, label_id in self._names_map.items():
            for is_begin, prefix in enumerate(("I", "B")):
                mapping[f"{prefix}-{name}"] = self.get_iob_label_id(
                    label_id, is_begin=bool(is_begin)
                )
        return mapping

    @cached_property
    def id2label(self):
        return {v: k for k, v in self.label2id.items()}

    def to_id(self, label: str):
        return self.label2id[label]

    def from_id(self, label_id: int):
        return self.id2label[label_id]

    @classmethod
    def loads(cls, root_dir):
        return cls(
            root_dir.joinpath("labels.list").read_text(encoding="utf8").split("\n")
        )

    def dumps(self, root_dir) -> None:
        Path(root_dir).joinpath("labels.list").write_text(
            "\n".join(self.names), encoding="utf8"
        )


SHARED_LABELS = EntityLabels(
    [
        "Anatomical Structure",
        "Biological State or Process",
        "Biomedical Procedure and Device",
        "Cell Type and Cell Line",
        "Chemical",
        "Clinical Condition",
        "Genetic Element",
        "Health Indicator",
        "Laboratory Procedure",
        "Lifestyle",
        "Organism and Virus",
        "Sequence Feature",
    ]
)


def filter_entity_labels(ds, labels):
    def _filter(example):
        example["entities"] = [
            entity for entity in example["entities"] if entity["type"] in labels
        ]
        return example

    return ds.map(
        _filter,
        batched=False,
        num_proc=cpu_count(),
        load_from_cache_file=False,
    )


def fallback_low_map(ds, func, fn_kwargs):
    assert isinstance(ds, DatasetDict)
    splits = {}
    for split_name, split_ds in ds.items():
        data = []
        for example in split_ds:
            data.append(func(example, **fn_kwargs))
        splits[split_name] = Dataset.from_list(data)
    return DatasetDict(splits)


def merge_passages(example, debug=False):
    text = ""
    for passage in example["passages"]:
        assert len(passage["text"]) == 1, passage
        text += passage["text"][0] + "\n"
    text = text[:-1]
    # NOTE: This is toxic!!!
    if text.endswith("(ABSTRACT TRUNCATED AT 250 WORDS)"):
        if debug:
            logger.debug("Truncation detected!")
        text = text[:-33]
    return text


def entity_text(entity):
    return " ... ".join(entity["text"])


def discard_discontinuous_entities(text, entities, debug=False):
    for entity in entities[:]:
        if len(entity["offsets"]) > 1 or len(entity["text"]) > 1:
            offsets = entity["offsets"]
            _start, _end = max(0, offsets[0][0] - 10), min(
                len(text), offsets[-1][1] + 10
            )
            if debug:
                logger.debug(
                    f'Skipping entity found! "{entity_text(entity)}" in "{text[_start:_end]}"'
                )
            entities.remove(entity)


def discard_overflow_entities(text, entities, debug=False):
    for entity in entities[:]:
        start, end = entity["offsets"][0]
        assert start >= 0
        if start >= len(text) or end > len(text):
            if debug:
                logger.debug("Overflow entity detected!", entity)
            entities.remove(entity)


def discard_nested_entities(text, entities, keep_longest, debug=False):
    offset_type_ids = [None] * len(text)
    overlay_entities = []
    sorted_entities = sorted(
        entities,
        key=lambda x: x["offsets"][0][1] - x["offsets"][0][0],
        reverse=keep_longest,
    )
    for ie, entity in enumerate(sorted_entities):
        start, end = entity["offsets"][0]
        target_area = set(offset_type_ids[start:end]) - {None}
        if len(target_area) != 0:
            overlay_entities.append((entity, target_area))
            entities.remove(entity)
            _entity_texts = [entity_text(sorted_entities[x]) for x in target_area]
            if debug:
                logger.debug(
                    f"Overlay entity detected! {entity_text(entity)} -> {_entity_texts}"
                )
        for index in range(start, end):
            offset_type_ids[index] = ie
    return overlay_entities


def clean_text(dirty):
    return dirty


def assert_offsets(text, entities):
    for entity in entities:
        for (start, end), (index, term) in zip(
            entity["offsets"], enumerate(entity["text"])
        ):
            substring = text[start:end]
            if substring != term:
                if substring.strip() == term:
                    new_start = start + substring.index(term)
                    new_end = new_start + len(term)
                    entity["offsets"][index] = [new_start, new_end]
                    assert text[new_start:new_end] == term
                else:
                    logger.warning(f'"{substring}" (substring) != "{term}" (annotated)')
                    assert (
                        fuzz.partial_ratio(substring, term) == 100
                        or fuzz.ratio(substring, term) >= 90
                    ), f'"{substring}" != "{term}"'
                    entity["text"][index] = substring


def discard_entity_without_type(entities):
    for entity in entities[:]:
        if entity["type"] is None:
            entities.remove(entity)


def validate_entities(
    example,
    overlay_keep_longest=True,
    debug=False,
    keep_discontinuous=False,
    keep_nested=False,
):
    text = clean_text(merge_passages(example, debug=debug))
    entities = example["entities"]
    discard_entity_without_type(entities)
    discard_overflow_entities(text, entities, debug=debug)
    if not keep_discontinuous:
        discard_discontinuous_entities(text, entities, debug=debug)
    if not keep_nested:
        discard_nested_entities(
            text, entities, keep_longest=overlay_keep_longest, debug=debug
        )
    try:
        assert_offsets(text, entities)
    except AssertionError as e:
        logger.error(f"Assertion error in {example['id']}! {e}")
        return {
            "id": example["id"],
            "text": None,
            "entities": [],
            "error": True,
        }
    else:
        return {
            "id": example["id"],
            "text": text,
            "entities": [
                {
                    "type": entity["type"],
                    "offsets": entity["offsets"],
                    "term": " ... ".join(entity["text"]),
                }
                for entity in entities
            ],
            "error": False,
        }


def dataset_dict_stats(dataset_dict):
    PERCENTILE_LABELS = {
        0: "min",
        5: "5%",
        50: "median",
        95: "95%",
        100: "max",
    }

    def _array_stats(list_):
        if not list_:
            return {"count": 0}
        array = np.array(list_)
        values = np.percentile(array, list(PERCENTILE_LABELS.keys()))
        return {
            **{
                label: int(value)
                for label, value in zip(PERCENTILE_LABELS.values(), values)
            },
            "total": int(array.sum()),
            "count": len(array),
        }

    results = {}
    for split_name, dataset in dataset_dict.items():
        # TODO: cooccurance, term type count
        metrics = {
            "text": {"character_length": []},
            "entities": {
                "containing_text_count": defaultdict(int),
                "character_length": defaultdict(list),
            },
        }
        for sample in tqdm(dataset, desc=split_name):
            if sample["text"] is None:
                continue
            metrics["text"]["character_length"].append(len(sample["text"]))
            entity_types = set()
            for entity in sample["entities"]:
                entity_type = entity["type"]
                entity_types.add(entity_type)
                metrics["entities"]["character_length"][entity_type].append(
                    entity["offsets"][0][1] - entity["offsets"][0][0]
                )
            for entity_type in entity_types:
                metrics["entities"]["containing_text_count"][entity_type] += 1
        results[split_name] = {
            "text": {
                "character_length": _array_stats(metrics["text"]["character_length"])
            },
            "entities": {
                "containing_text_count": dict(
                    metrics["entities"]["containing_text_count"]
                ),
                "character_length": {
                    k: _array_stats(v)
                    for k, v in metrics["entities"]["character_length"].items()
                },
            },
        }
    # TODO: Visualize
    return results


def validate_entity_types(ds, min_entity_count=10):
    stats = dataset_dict_stats(ds)
    if "train" not in stats or "test" not in stats:
        logger.warning("No train/test split found! Skipping type validation...")
    else:
        train_types = set(stats["train"]["entities"]["containing_text_count"].keys())
        test_types = set(stats["test"]["entities"]["containing_text_count"].keys())
        assert train_types.issuperset(test_types), (
            test_types - train_types,
            train_types,
        )
    for split in stats.values():
        for entity_type, entity_stats in list(
            split["entities"]["character_length"].items()
        ):
            if entity_stats["count"] < min_entity_count:
                logger.warning(f"Discarding {entity_type} due to low count!")
                del split["entities"]["character_length"][entity_type]
                del split["entities"]["containing_text_count"][entity_type]
    return stats


class DataCollatorForTokenClassification(BaseDataCollatorForTokenClassification):
    def __call__(self, dataset: Dataset | list, device=None):
        if isinstance(dataset, list):
            dataset = Dataset.from_list(dataset)
        unnecessary_columns = set(dataset.column_names) - {"input_ids", "labels"}
        if unnecessary_columns:
            dataset = dataset.remove_columns(unnecessary_columns)
        if device:
            dataset.set_format("torch", device=device)
        return super().__call__(dataset)


def validate_model(model_name: str):
    logger.info(f"Validating model {model_name}")
    model_name = model_name.strip()
    if not model_name or model_name.startswith("#"):
        logger.warning(f"Skipping model {model_name}")
        return
    model_dir = MODELS_BASEDIR / model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    except Exception:
        snapshot_download(
            repo_id=model_name,
            endpoint="https://hf-mirror.com",
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=[
                "*.msgpack",
                "*.safetensors",
                "*.h5",
                "*.ot",
                "*.onnx",
                "*.ckpt.*",
                "onnx/*",
            ],
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
    finally:
        inputs = tokenizer("Hello", return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)
        outputs = model(**inputs, labels=labels)
        logger.debug(outputs.logits)
        logger.success(f"Model {model_name} is valid")


def load_ner_model(model_name, labels, load_model_config_only=False, device=None):
    model_dir = MODELS_BASEDIR / model_name
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = (
        AutoModelForTokenClassification.from_config(
            AutoConfig.from_pretrained(
                model_dir, id2label=labels.id2label, label2id=labels.label2id
            )
        )
        if load_model_config_only
        else AutoModelForTokenClassification.from_pretrained(
            model_dir,
            id2label=labels.id2label,
            label2id=labels.label2id,
            low_cpu_mem_usage=True,
        )
    )
    if device:
        model.to(device)

    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=model.config.max_position_embeddings,
        label_pad_token_id=SPECIAL_TOKEN_LABEL_ID,
    )
    return model, tokenizer, collator


def check_unk_tokens(tokenized, text, unk_token_id):
    offset_mapping = tokenized["offset_mapping"]
    for i in range(len(tokenized["input_ids"])):
        token_id = tokenized["input_ids"][i]
        if token_id == unk_token_id:
            context_left_span = offset_mapping[max(0, i - 5) : i]
            context_left = text[context_left_span[0][0] : context_left_span[-1][1]]
            context_right_span = offset_mapping[i : max(i + 6, len(offset_mapping))]
            context_right = text[context_right_span[0][0] : context_right_span[-1][1]]
            unk_start, unk_end = offset_mapping[i]
            unk = text[unk_start:unk_end]
            context = f"{context_left} <red>{unk}</red> {context_right}"
            logger.opt(colors=True).warning(f'[unk] found in: "{context}"')


def check_entity_token_boundaries(tokenized, entities, text):
    def _span_overlap(span1, span2):
        return span1[0] < span2[1] and span1[1] > span2[0]

    offset_mapping = tokenized["offset_mapping"]
    token_offsets = set(sum(map(list, offset_mapping), []))
    left_most_token_map = {}
    last_end = 0
    for token_index, (start, end) in enumerate(offset_mapping):
        for i in range(last_end, start):
            left_most_token_map[i] = token_index - 1
        for i in range(start, end):
            left_most_token_map[i] = token_index
        last_end = end
    for i in range(last_end, len(text)):
        left_most_token_map[i] = len(offset_mapping) - 1
    right_most_token_map = {}
    last_end = 0
    for token_index, (start, end) in enumerate(offset_mapping):
        for i in range(last_end, start):
            right_most_token_map[i] = token_index
        for i in range(start, end):
            right_most_token_map[i] = token_index
        last_end = end
    for i in range(last_end, len(text)):
        right_most_token_map[i] = len(offset_mapping) - 1
    for entity in entities:
        entity_span = entity["offsets"][0]
        if set(entity_span).issubset(token_offsets):
            continue
        try:
            left_most_token_index = left_most_token_map[entity_span[0]]
            right_most_token_index = right_most_token_map[entity_span[1]]
        except KeyError as e:
            print(entity_span, flush=True)
            print(left_most_token_map.keys(), flush=True)
            print(right_most_token_map.keys(), flush=True)
            raise e
        context_span = offset_mapping[
            max(left_most_token_index - 5, 0) : min(
                right_most_token_index + 5, len(offset_mapping)
            )
        ]
        context_start, context_end = context_span[0][0], context_span[-1][1]
        chars = list(text[context_start:context_end])
        _overlapped = False
        for token_offset in context_span:
            token_start = token_offset[0] - context_start
            if _span_overlap(token_offset, entity_span):
                chars[token_start] = f"🪚{chars[token_start]}"
                _overlapped = True
            elif _overlapped:
                chars[token_start] = f"🪚{chars[token_start]}"
                break
        entity_start, entity_end = entity_span
        entity_start -= context_start
        entity_end -= context_start
        chars[entity_start] = f"<red>{chars[entity_start]}"
        # print(entity_end, len(chars), flush=True)
        chars[entity_end - 1] = f"{chars[entity_end - 1]}</red>"
        context = "".join(chars)
        logger.opt(colors=True).warning(f'Entity not in token splits: "{context}"')


def add_label_ids(tokenized, entities, labels, labels_key="labels"):
    offset_mapping = tokenized["offset_mapping"][:]
    token_label_ids = []
    for entity in sorted(entities, key=lambda x: x["offsets"][0][0]):
        start, end = entity["offsets"][0]
        while offset_mapping[0][0] < start:
            offset_mapping.pop(0)
            token_label_ids.append(labels.LABEL_O_ID)
        begin_token_id = labels.get_iob_label_id(labels[entity["type"]], is_begin=True)
        token_label_ids.append(begin_token_id)
        offset_mapping.pop(0)
        while len(offset_mapping) and offset_mapping[0][-1] <= end:
            offset_mapping.pop(0)
            inner_token_id = labels.get_iob_label_id(
                labels[entity["type"]], is_begin=False
            )
            token_label_ids.append(inner_token_id)
    for _ in range(len(offset_mapping)):
        token_label_ids.append(labels.LABEL_O_ID)
    assert len(token_label_ids) == len(tokenized["input_ids"])
    tokenized[labels_key] = token_label_ids
    return tokenized


def tokenize_example(example, labels, tokenizer: PreTrainedTokenizerFast):
    text = example["text"]
    tokenized = tokenizer(
        text,
        return_offsets_mapping=True,
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    check_unk_tokens(tokenized, example["text"], unk_token_id=tokenizer.unk_token_id)
    if "entities" in example:
        check_entity_token_boundaries(tokenized, example["entities"], example["text"])
        tokenized = add_label_ids(tokenized, example["entities"], labels)
    return {"tokenized": tokenized}
