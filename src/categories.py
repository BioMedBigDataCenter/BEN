# from utils import embedding
# import numpy as np
from enum import Enum

import pandas as pd
from pydantic import BaseModel, field_validator

from constants import PROJECT_ROOT

NON_CAPITALIZE = (
    "a",
    "an",
    "the",
    "at",
    "by",
    "in",
    "of",
    "to",
    "and",
    "but",
    "if",
    "nor",
    "or",
    "so",
    "yet",
)


class EntityLabel(BaseModel):
    term: str
    description: str

    @field_validator("term", mode="before")
    def term_must_be_camelcase(cls, v):
        return cls.capitalize(v)

    @staticmethod
    def capitalize(text):
        return " ".join(
            [x.capitalize() if x not in NON_CAPITALIZE else x for x in text.split(" ")]
        )

    @property
    def const_name(self):
        return self.term.upper().replace(" ", "_")

    def to_prompt(self):
        return f'- **{self.term}**: {self.description}'

    def __hash__(self) -> int:
        return hash(self.term)


ORGANISM_OR_VIRUS = EntityLabel(
    term="Organism or Virus",
    description="Virus or a living entity capable of independent growth, reproduction, and response to stimuli, encompassing plants, fungi, viruses, bacteria, archaea, and eukaryotes including humans. It also includes organisms that have been genetically modified, selectively bred, or otherwise manipulated to exhibit characteristics of a disease.",
)

CHEMICAL = EntityLabel(
    term="Chemical",
    description="Any synthetic or naturally occurring substance that can influence physiological processes in living organisms. This includes organic and inorganic compounds, metabolites, drugs, hormones, neurotransmitters, and other bioactive molecules. Excludes nucleotides, amino acids, and their polymeric forms (proteins, peptides, DNA, RNA).",
)

GENETIC_ELEMENT = EntityLabel(
    term="Genetic Element",
    description="Individual biological molecules including genomes, genes, nucleic acids, nucleotides, DNAs, RNAs, proteins (enzymes, antibodies, etc), amino acid, peptides, protein complexes, and their variants/mutant, isoforms, families, subunits, assemblies.",
)

SEQUENCE_FEATURE = EntityLabel(
    term="Sequence Feature",
    description="A identifiable structural or functional element within a nucleic acid or protein sequence, including but not limited to: protein domains, binding motifs, catalytic sites, coding regions, regulatory elements, splice sites, conserved regions, and post-translational modification sites.",
)

CELL_TYPE_OR_CELL_LINE = EntityLabel(
    term="Cell Type or Cell Line",
    description="Distinct categories of cells characterized by their morphology, function, or origin within an organism, and populations of cells derived from a single cell and maintained in culture for research purposes, including those employed as experimental cell models to study disease mechanisms.",
)

# FIXME: structural proteins?
ANATOMICAL_STRUCTURE = EntityLabel(
    term="Anatomical Structure",
    description="A tangible anatomical structure, encompassing body systems, organs, tissues, body fluids, spaces and junctions, cell components, pathological formations and acquired/congenital abnormalities (e.g., tumor, scar tissue, accessory auricle) and functional units (e.g., the nephron in the kidney). This includes both fully developed and embryonic structures, formed either in vivo or cultivated in vitro.",
)

CLINICAL_CONDITION = EntityLabel(
    term="Clinical Condition",
    description="Medical and health-related states encompassing: diseases and disorders, signs and symptoms, pathological processes, cell or molecular dysfunction, patient demographic characteristics, and clinically relevant physical and physiological attributes.",
)

HEALTH_INDICATOR = EntityLabel(
    term="Health Indicator",
    description="Quantifiable health metrics including biometrics, physical performance, sensory capabilities, cognitive measures, behavioral markers, and population statistics (mortality, disease rates, etc).",
)

# FIXME: merge to clinical conditions?
BIOLOGIC_STATE_OR_PROCESS = EntityLabel(
    term="Biologic State or Process",
    description="Normal states, functions, pathways, and processes occurring at various biological levels, including molecular, genetic, cellular, and organ/tissue.",
)

BIOMEDICAL_PROCEDURE_OR_DEVICE = EntityLabel(
    term="Biomedical Procedure or Device",
    description="Any clinical, biomedical, or laboratory process, including diagnostic/therapeutic procedures, use of medical devices, and experimental, procedural, and analytical methods for treatment, healthcare delivery, or scientific investigation.",
)

LIFESTYLE = EntityLabel(
    term="Lifestyle",
    description="Lifestyle factors that might affect physical and mental wellbeing including diet, dishes, formulas, food and nutrients and edible product intake, physical activity, sleep, stress management, social interactions, substance use pattern and others.",
)

FAMILY_ROLE = EntityLabel(
    term="Family Role",
    description="A social unit of individuals connected by blood, marriage, or adoption. Includes nuclear and extended families, recognizing diverse familial configurations defined by societal and cultural norms.",
)

TEMPORAL_DESCRIPTOR = EntityLabel(
    term="Temporal Descriptor",
    description="Entities that capture time-related nuances in medical and biological domains, encompassing intervals, phases, chronological markers, and timing details crucial for interpreting event sequences, medical schedules, clinical progressions, etc.",
)

ORGANIZATION = EntityLabel(
    term="Organization",
    description="Entities representing institutions, companies, societies, and other organized entities involved in healthcare, research, education, support, advocacy, and other biomedical activities.",
)


OTHER_OR_MIXED = EntityLabel(
    term="Other or Mixed",
    description="Entities that do not fit into any of the above categories or are a mix of multiple categories.",
)

ALL_ENTITY_TYPES = [
    ANATOMICAL_STRUCTURE,
    BIOLOGIC_STATE_OR_PROCESS,
    BIOMEDICAL_PROCEDURE_OR_DEVICE,
    CELL_TYPE_OR_CELL_LINE,
    CHEMICAL,
    CLINICAL_CONDITION,
    HEALTH_INDICATOR,
    GENETIC_ELEMENT,
    LIFESTYLE,
    ORGANISM_OR_VIRUS,
    SEQUENCE_FEATURE,
    FAMILY_ROLE,
    TEMPORAL_DESCRIPTOR,
    ORGANIZATION,
]
EntityType = Enum(
    "EntityType",
    {item.const_name: item for item in [*ALL_ENTITY_TYPES, OTHER_OR_MIXED]},
)
ENTITY_MAP = {item.term: EntityType[item.const_name] for item in ALL_ENTITY_TYPES}
# ENTITY_MAP.update({"Cell Type or Cell Line": EntityType.BIOLOGICAL_MODELS})
# ENTITY_TYPE_EMBEDDINGS = [embedding([entity.to_prompt()])[0] for entity in ALL_ENTITY_TYPES]


# def entity_type_best_match(text, top_k=3):
#     scores = np.dot(np.array(embedding([text])[0]), np.array(list(ENTITY_TYPE_EMBEDDINGS)).T)
#     top_indices = np.argsort(scores)[::-1][:top_k]
#     return [(ALL_ENTITY_TYPES[i], float(scores[i])) for i in top_indices]


def load_bigbio_category_mapping():
    mapping = {}
    with pd.ExcelFile(PROJECT_ROOT.joinpath("data/bigbio_kb/mappings.xlsx")) as xls:
        for sheet in xls.sheet_names:
            df = (
                pd.read_excel(xls, sheet, index_col=0)
                .dropna(how="all", axis=1)
                .dropna(how="all", axis=0)
            )
            final_decision = df.iloc[-1]
            assert final_decision.name == "F"
            mapping[sheet] = final_decision.to_dict()
    return mapping
