# import gc
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from loguru import logger
from nltk.corpus import stopwords
from peft import PeftModel
from seqeval.metrics.sequence_labeling import get_entities
from tqdm.auto import tqdm
from transformers import logging

from constants import LORA_BASEDIR
from loader import SHARED_LABELS, load_ner_model, tokenize_example
from segmentation import ensemble_results, segment_tokenized

logging.set_verbosity_error()
STOP_WORDS = set(
    stopwords.words("english") + [x.capitalize() for x in stopwords.words("english")]
)
SOURCE_MAPPING = {
    "anat_em": "an",
    "bc5cdr": "bc",
    "bioinfer": "bi",
    "bionlp_shared_task_2009": "b9",
    "bionlp_st_2011_id": "1i",
    "bionlp_st_2011_rel": "1r",
    "bionlp_st_2013_cg": "3c",
    "bionlp_st_2013_ge": "3g",
    "bionlp_st_2013_pc": "3p",
    "biored": "bd",
    "biorelex": "bx",
    "chebi_nactem_fullpaper": "cf",
    "chem_dis_gene": "cd",
    "chemdner": "cn",
    "chemprot": "cp",
    "ddi_corpus": "dd",
    "drugprot": "dp",
    "genetaggold": "gt",
    "genia_relation_corpus": "gr",
    "genia_term_corpus": "gc",
    "gnormplus": "gp",
    "linnaeus": "ln",
    "mirna": "mi",
    "mlee": "ml",
    "ncbi_disease": "nd",
    "neurotrial_ner": "nt",
    "nlm_gene": "ng",
    "pdr": "pd",
    "ppr": "pp",
    "scai_disease": "sd",
    "spl_adr_200db_train": "sa",
    "tmvar_v1": "t1",
}
assert len(SOURCE_MAPPING) == len(set(SOURCE_MAPPING.values())), "Duplicate mappings"


def series_to_nested_dict(series, levels=None):
    if levels is None:
        levels = series.index.names
    if len(levels) == 1:
        # Base case: single level, map index to value
        return series.to_dict()
    else:
        # Recursive case: group by the first level and recurse
        first_level = levels[0]
        nested_dict = {}
        for val in series.index.get_level_values(first_level).unique():
            # Select subset where first level is val
            subset = series.xs(val, level=first_level)
            # Recurse with remaining levels
            nested_dict[val] = series_to_nested_dict(subset, levels=levels[1:])
        return nested_dict


TOKENIZER, COLLATOR, ALL_ADAPTERS, MODEL = None, None, None, None


def load_model(device="cuda:0"):
    global TOKENIZER, COLLATOR, ALL_ADAPTERS, MODEL
    if (
        MODEL is not None
        or ALL_ADAPTERS is not None
        or TOKENIZER is not None
        or COLLATOR is not None
    ):
        return
    logger.info(f"Loading the BERT model on {device}...")
    bert, TOKENIZER, COLLATOR = load_ner_model(
        "microsoft/PubMedBERT", labels=SHARED_LABELS, device=device
    )
    logger.success("Models loaded!")
    logger.info("Loading adapters...")
    adapter_dirs = list(LORA_BASEDIR.iterdir())
    with tqdm(total=len(adapter_dirs), desc="Loaded adapters", leave=False) as bar:
        MODEL = PeftModel.from_pretrained(
            bert, LORA_BASEDIR / "bc5cdr", adapter_name="bc5cdr"
        )
        bar.update()
        ALL_ADAPTERS = {"bc5cdr"}
        for adapter_dir in adapter_dirs:
            if adapter_dir.name == "bc5cdr":
                continue
            ds_name = adapter_dir.name
            MODEL.load_adapter(adapter_dir, adapter_name=ds_name)
            ALL_ADAPTERS.add(ds_name)
            bar.update()
    logger.success("Adapters loaded!")


def geo_mean(iterable):
    return np.exp(np.log(iterable).mean())


class NamedEntity:
    def __init__(self, entity_type, offsets, example, token_probs=None):
        self.entity_type = entity_type
        self.offsets = list(tuple(x) for x in offsets)
        assert (
            len(self.offsets) == 1
        ), f"Multiple offsets are not supported: {self.offsets}"
        assert "text" in example, f"Example does not have a text: {example}"
        self.example = example
        self.token_probs = token_probs
        self.term = " ... ".join(
            [self.example["text"][offset[0] : offset[1]] for offset in self.offsets]
        )
        self.auto_adjust()

    def auto_adjust(self):
        while len(self.term) and self.term[0] in " \\/-,.)]}":
            self.term = self.term[1:]
            self.offsets[0] = (self.offsets[0][0] + 1, self.offsets[0][1])
        while len(self.term) and self.term[-1] in " \\/-,.([{":
            self.term = self.term[:-1]
            self.offsets[-1] = (self.offsets[-1][0], self.offsets[-1][1] - 1)

    def __repr__(self):
        start, end = self.offsets[0]
        return f"{self.entity_type:35} {self.term:50} {start}:{end} ({self.prob:.2f})"

    @property
    def prob(self):
        return geo_mean(self.token_probs) if self.token_probs else None

    @property
    def start_offset(self):
        return self.offsets[0][0]

    @property
    def end_offset(self):
        return self.offsets[0][1]

    @classmethod
    def batched_from_example(cls, example):
        return [
            cls(entity["type"], entity["offsets"], example)
            for entity in example["entities"]
        ]

    def extend(self, entity):
        self.offsets[0] = (self.start_offset, entity.end_offset)

    def __eq__(self, other):
        return self.entity_type == other.entity_type and all(
            x == y for x, y in zip(self.offsets, other.offsets)
        )

    def __hash__(self):
        return hash((self.entity_type, tuple(self.offsets)))

    def to_dict(self):
        return {
            "type": self.entity_type,
            "span": (self.start_offset, self.end_offset),
            "term": self.term,
            **({"prob": self.prob} if self.token_probs else {}),
        }


def batched_prediction(model, segmented, collator, batch_size=3):
    outputs = []
    for i in range(0, len(segmented), batch_size):
        batch = segmented[i : i + batch_size]
        output = model(**collator(batch).to(model.device)).logits.detach()
        outputs.append(output)
    return torch.cat(outputs, dim=0)


def predict(
    example,
    adapter_names,
    overlap_length=50,
    return_df=True,
    shorten_sources=False,
):
    all_entities = {}
    example.update(
        tokenize_example(example, labels=SHARED_LABELS, tokenizer=TOKENIZER)
    )
    segmented = segment_tokenized(
        example, tokenizer=TOKENIZER, max_length=512, overlap=overlap_length
    )
    num_segments = len(segmented)
    for adapter_name in adapter_names or ALL_ADAPTERS:
        MODEL.set_adapter(adapter_name)
        # output = MODEL(**COLLATOR(segmented).to(MODEL.device)).logits.detach().cpu()
        output = batched_prediction(MODEL, segmented, COLLATOR, batch_size=3)
        probs, offsets = ensemble_results(
            output, segmented, overlap_length=overlap_length
        )
        # del output
        # gc.collect()
        assert probs.shape[0] == offsets.shape[0], "Probs and offsets length mismatch"
        assert probs.shape[1] == len(SHARED_LABELS), "Probs and labels length mismatch"
        assert offsets.shape[1] == 2, "Offsets should have 2 columns"
        token_labels = probs.argmax(axis=-1).tolist()
        assert len(token_labels) == len(
            offsets
        ), f"Token labels and offsets length mismatch: {len(token_labels)} != {len(offsets)}"
        entity_spans = get_entities([SHARED_LABELS.id2label[x] for x in token_labels])
        entities = []
        last_entity = None
        for entity in entity_spans:
            entity_type, start, end = entity
            token_offsets = offsets[start : end + 1, :]
            all_token_probs = probs[start : end + 1, :]
            if not token_offsets.size:
                continue
            assert all_token_probs.size, (token_offsets, all_token_probs)
            label_id = SHARED_LABELS[entity_type]
            label_bid = SHARED_LABELS.get_iob_label_id(label_id, is_begin=True)
            label_iid = SHARED_LABELS.get_iob_label_id(label_id, is_begin=False)
            token_probs = [all_token_probs[0][[label_bid, label_iid]].max().item()]
            for _probs in all_token_probs[1:]:
                token_probs.append(_probs[[label_bid, label_iid]].max().item())
            assert len(token_probs) == len(token_offsets), "Token probs length mismatch"

            entity_start, entity_end = token_offsets[0][0], token_offsets[-1][1]
            entity = NamedEntity(
                entity_type,
                [(entity_start, entity_end)],
                example,
                token_probs=token_probs,
            )
            if (
                last_entity
                and entity.start_offset == last_entity.end_offset
                and entity.entity_type == last_entity.entity_type
            ):
                last_entity.extend(entity)
            elif len(entity.term) > 1 and entity.term not in STOP_WORDS:
                entities.append(entity)
                last_entity = entity
        all_entities[adapter_name] = entities
    # del segmented
    # gc.collect()
    if return_df:
        all_entities = (
            pd.DataFrame(
                sum(
                    [
                        [{"model": ds_name, **entity.to_dict()} for entity in entities]
                        for ds_name, entities in all_entities.items()
                    ],
                    [],
                )
            )
            .assign(
                span=lambda x: x.span.apply(lambda x: f"{x[0]}:{x[1]}"),
                **(
                    {"model": lambda x: x.model.map(SOURCE_MAPPING)}
                    if shorten_sources
                    else {}
                ),
            )
            .set_index(["type", "term", "span", "model"])
            .prob.sort_index()
        )
    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    return all_entities, num_segments


def extract_entities(
    text,
    datasets=None,
    return_probs=True,
    return_sources=False,
    return_spans=False,
    shorten_sources=False,
    min_prob=0,
):
    adapter_names = ALL_ADAPTERS
    if datasets:
        adapter_names = adapter_names & set(datasets)

    start = perf_counter()
    all_entities, num_segments = predict(
        example={"text": text},
        adapter_names=adapter_names,
        shorten_sources=shorten_sources,
        return_df=True,
    )
    end = perf_counter()
    duration = end - start
    # print("Predicted:", all_entities)

    if not return_spans and not return_sources:
        all_entities = all_entities.groupby(level=[0, 1]).mean()
    elif return_spans and not return_sources:
        all_entities = all_entities.groupby(level=[0, 1, 2]).mean()
    elif not return_spans and return_sources:
        all_entities = all_entities.groupby(level=[0, 1, 3]).mean()
    else:
        pass

    if return_probs:
        if min_prob:
            all_entities = all_entities[all_entities >= min_prob]
        all_entities = all_entities.round(2)
    else:
        all_entities = all_entities.reset_index(level=-1, drop=False).iloc[:, 0]
        all_entities = all_entities.groupby(all_entities.index.names).apply(list)

    return {
        "entities": series_to_nested_dict(all_entities.sort_index()),
        "stats": {
            "num_chars": len(text),
            "elapsed": round(duration, 3),
            "elapsed_per_k_chars": round(duration / len(text) * 1000, 3),
            "num_categories": len(all_entities.index.levels[0]),
            "num_terms": len(all_entities.index.levels[1]),
            "num_segments": num_segments,
        }
    }


def nested_entities_to_df(entities):
    if len(entities) == 0:
        return pd.DataFrame()
    return (
        pd.DataFrame(
            [
                {
                    "category": cat,
                    "term": term,
                    **dict(zip(("start", "end"), map(int, span.split(":")))),
                    "prob": prob,
                }
                for cat, terms in entities.items()
                for term, spans in terms.items()
                for span, prob in spans.items()
            ]
        )
        .set_index(["category", "term", "start", "end"])
        .sort_index()
        ["prob"]
    )
