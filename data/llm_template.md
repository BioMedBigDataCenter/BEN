### Purpose

Named Entity Recognition (NER) for Biomedical Text

### Instruction

Extract entities belonging to the specified categories (`Anatomical Structure`, `Cell Type or Cell Line`, `Chemical`, `Clinical Condition`, `Genetic Element`, `Organism or Virus`, `Biologic State or Process`) from the provided text.

### Output Format

For each identified entity, provide the following structured response:
- `<entity_term>` is a <entity_category>
And append a mandatory finishing line as:
- (nothing else)

### Entity Categories with Definitions

- `Anatomical Structure`: A tangible anatomical structure, encompassing body systems, organs, tissues, body fluids, spaces and junctions, cell components, pathological formations and acquired/congenital abnormalities (e.g., tumor, scar tissue, accessory auricle) and functional units (e.g., the nephron in the kidney). This includes both fully developed and embryonic structures, formed either in vivo or cultivated in vitro.
- `Cell Type or Cell Line`: Distinct categories of cells characterized by their morphology, function, or origin within an organism, and populations of cells derived from a single cell and maintained in culture for research purposes, including those employed as experimental cell models to study disease mechanisms.
- `Chemical`: Any synthetic or naturally occurring substance that can influence physiological processes in living organisms. This includes organic and inorganic compounds, metabolites, drugs, hormones, neurotransmitters, and other bioactive molecules. Excludes nucleotides, amino acids, and their polymeric forms (proteins, peptides, DNA, RNA).
- `Clinical Condition`: Medical and health-related states encompassing: diseases and disorders, signs and symptoms, pathological processes, cell or molecular dysfunction, patient demographic characteristics, and clinically relevant physical and physiological attributes.
- `Genetic Element`: Individual biological molecules including genomes, genes, nucleic acids, nucleotides, DNAs, RNAs, proteins (enzymes, antibodies, etc), amino acid, peptides, protein complexes, and their variants/mutant, isoforms, families, subunits, assemblies.
- `Organism or Virus`: Virus or a living entity capable of independent growth, reproduction, and response to stimuli, encompassing plants, fungi, viruses, bacteria, archaea, and eukaryotes including humans. It also includes organisms that have been genetically modified, selectively bred, or otherwise manipulated to exhibit characteristics of a disease.
- `Biologic State or Process`: Normal states, functions, pathways, and processes occurring at various biological levels, including molecular, genetic, cellular, and organ/tissue.

### Text to be Extracted

{text}
