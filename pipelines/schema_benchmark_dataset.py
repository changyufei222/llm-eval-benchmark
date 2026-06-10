from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_COUNTS = {
    "protein_profile": 8,
    "protein_identifier": 4,
    "interaction_overview": 8,
    "affinity": 8,
    "developability": 6,
    "annotation": 4,
    "provenance": 4,
    "structure": 4,
    "digestive_assay": 2,
    "immunogenicity": 2,
    "loop_annotation": 2,
    "loop_flexibility": 2,
    "protein_flexibility": 2,
    "target_variant": 2,
    "source_metadata": 2,
}


def _normalize(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.casefold() in {"nan", "none", "null", "n/a", "na"}:
            return None
        return text or None
    return value


def _clean_expected(values: list[Any]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        normalized = _normalize(value)
        if normalized is None:
            continue
        rendered = str(normalized).strip()
        if not rendered:
            continue
        cleaned.append(rendered)
    return cleaned


def _pick(*values: Any) -> str | None:
    for value in values:
        normalized = _normalize(value)
        if normalized is not None:
            return str(normalized)
    return None


def _as_bool(value: Any) -> bool | None:
    normalized = _pick(value)
    if normalized is None:
        return None
    lowered = normalized.casefold()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def _load_tables(schema_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(schema_dir.glob("*.csv")):
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        tables[csv_path.stem] = frame.apply(lambda column: column.map(_normalize))
    return tables


def _records(tables: dict[str, pd.DataFrame], name: str) -> list[dict[str, Any]]:
    frame = tables.get(name)
    if frame is None:
        return []
    return frame.to_dict(orient="records")


def _index_by(frame: pd.DataFrame, key: str) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dict(orient="records"):
        value = row.get(key)
        if value is None:
            continue
        lookup.setdefault(str(value), []).append(row)
    return lookup


def _choice(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = records[:]
    rng.shuffle(pool)
    return pool[: min(count, len(pool))]


def _protein_profile_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    domains_by_protein = _index_by(tables["domains"], "protein_id")
    rows = []
    for protein in _records(tables, "proteins"):
        protein_id = protein.get("protein_id")
        domains = domains_by_protein.get(str(protein_id), [])
        if not protein_id or not domains:
            continue
        domain = domains[0]
        canonical_name = _pick(protein.get("canonical_name"))
        organism = _pick(protein.get("organism"))
        scaffold_type = _pick(domain.get("scaffold_type"))
        domain_name = _pick(domain.get("domain_name"), domain.get("domain_id"))
        if not canonical_name or not organism or not scaffold_type or not domain_name:
            continue
        rows.append(
            {
                "question": f"Give a concise profile for protein {protein_id}. Include canonical name, organism, and scaffold/domain information.",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Protein {protein_id} is {canonical_name} from {organism}. "
                    f"Its main domain is {domain_name} and the scaffold type is {scaffold_type}."
                ),
                "expected_answer_contains": [
                    canonical_name,
                    organism,
                    scaffold_type,
                ],
                "expected_source_contains": ["protein_cards.jsonl"],
                "expected_table_contains": ["proteins", "domains"],
                "expected_primary_ids": [protein_id, domain.get("domain_id")],
                "tags": ["schema_tables", "protein_profile"],
                "query_type": "protein_profile",
                "difficulty": "easy",
            }
        )
    return _choice(rows, count, seed)


def _protein_identifier_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    identifiers = _records(tables, "protein_identifiers")
    rows = []
    for item in identifiers:
        protein_id = _pick(item.get("protein_id"))
        id_type = _pick(item.get("id_type"))
        id_value = _pick(item.get("id_value"))
        if not protein_id or not id_type or not id_value:
            continue
        rows.append(
            {
                "question": f"What {id_type} identifier is recorded for protein {protein_id}?",
                "record_type": "jsonl",
                "ground_truth": f"Protein {protein_id} has {id_type} identifier {id_value}.",
                "expected_answer_contains": [id_type, id_value],
                "expected_source_contains": ["protein_cards.jsonl"],
                "expected_table_contains": ["proteins", "protein_identifiers"],
                "expected_primary_ids": [protein_id, item.get("identifier_id")],
                "tags": ["schema_tables", "protein_identifier"],
                "query_type": "protein_identifier",
                "difficulty": "easy",
            }
        )
    return _choice(rows, count, seed)


def _interaction_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    domains_by_id = {row["domain_id"]: row for row in _records(tables, "domains") if row.get("domain_id")}
    proteins_by_id = {row["protein_id"]: row for row in _records(tables, "proteins") if row.get("protein_id")}
    rows = []
    for interaction in _records(tables, "interactions"):
        interaction_id = interaction.get("interaction_id")
        domain = domains_by_id.get(interaction.get("domain_id"))
        protein = proteins_by_id.get(domain.get("protein_id")) if domain else None
        if not interaction_id or not domain or not protein:
            continue
        inhibitory_value = _as_bool(interaction.get("is_inhibitory"))
        inhibitory = "Yes" if inhibitory_value is True else "No" if inhibitory_value is False else "Unknown"
        canonical_name = _pick(protein.get("canonical_name"))
        domain_name = _pick(domain.get("domain_name"), domain.get("domain_id"))
        if not canonical_name or not domain_name:
            continue
        rows.append(
            {
                "question": f"For interaction {interaction_id}, which protein/domain is involved and is it inhibitory?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Interaction {interaction_id} involves protein {canonical_name} "
                    f"through domain {domain_name}. Inhibitory status: {inhibitory}."
                ),
                "expected_answer_contains": [canonical_name, domain_name, inhibitory],
                "expected_source_contains": ["interaction_cards.jsonl"],
                "expected_table_contains": ["interactions", "domains", "proteins"],
                "expected_primary_ids": [interaction_id, domain.get("domain_id"), protein.get("protein_id")],
                "tags": ["schema_tables", "interaction_overview"],
                "query_type": "interaction_overview",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def _affinity_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for affinity in _records(tables, "affinity_data"):
        interaction_id = _pick(affinity.get("interaction_id"))
        value = _pick(affinity.get("value"))
        determination_method = _pick(affinity.get("determination_method"))
        unit = _pick(affinity.get("unit"))
        if not interaction_id or not value:
            continue
        method_text = determination_method or "unknown method"
        answer_contains = [value]
        if determination_method:
            answer_contains.append(determination_method)
        rows.append(
            {
                "question": f"What affinity is recorded for interaction {interaction_id} and how was it determined?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Interaction {interaction_id} has affinity {value} "
                    f"{unit or ''} measured by {method_text}."
                ).strip(),
                "expected_answer_contains": answer_contains,
                "expected_source_contains": ["interaction_cards.jsonl"],
                "expected_table_contains": ["affinity_data", "interactions"],
                "expected_primary_ids": [interaction_id, affinity.get("affinity_id")],
                "tags": ["schema_tables", "affinity"],
                "query_type": "affinity",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def _developability_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    domains_by_id = {row["domain_id"]: row for row in _records(tables, "domains") if row.get("domain_id")}
    proteins_by_id = {row["protein_id"]: row for row in _records(tables, "proteins") if row.get("protein_id")}
    rows = []
    for cmc in _records(tables, "cmc_data"):
        domain = domains_by_id.get(cmc.get("domain_id"))
        protein = proteins_by_id.get(domain.get("protein_id")) if domain else None
        if not domain or not protein:
            continue
        protein_id = protein.get("protein_id")
        if not protein_id:
            continue
        signals: list[str] = []
        expected_signals: list[str] = [protein_id]
        expression_system = _pick(cmc.get("expression_system"))
        oral_properties = _pick(cmc.get("oral_properties"))
        stability_rating = _pick(cmc.get("stability_rating"))
        if expression_system:
            signals.append(f"expression system {expression_system}")
            expected_signals.append(expression_system)
        if oral_properties:
            signals.append(f"oral property {oral_properties}")
            expected_signals.append(oral_properties)
        if stability_rating:
            signals.append(f"stability rating {stability_rating}")
            expected_signals.append(stability_rating)
        if not signals:
            continue
        rows.append(
            {
                "question": f"What developability or CMC signal is recorded for protein {protein_id}?",
                "record_type": "jsonl",
                "ground_truth": f"Protein {protein_id} has " + ", ".join(signals[:-1]) + (", and " if len(signals) > 1 else "") + signals[-1] + ".",
                "expected_answer_contains": expected_signals[:3],
                "expected_source_contains": ["protein_cards.jsonl"],
                "expected_table_contains": ["cmc_data", "proteins", "domains"],
                "expected_primary_ids": [protein_id, domain.get("domain_id"), cmc.get("cmc_id")],
                "tags": ["schema_tables", "developability"],
                "query_type": "developability",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def _annotation_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for ann in _records(tables, "functional_annotations"):
        protein_id = _pick(ann.get("protein_id"))
        annotation_type = _pick(ann.get("annotation_type"))
        annotation_value = _pick(ann.get("annotation_value"))
        if not protein_id or not annotation_type or not annotation_value:
            continue
        snippet = annotation_value[:120]
        rows.append(
            {
                "question": f"What {annotation_type} annotation is recorded for protein {protein_id}?",
                "record_type": "jsonl",
                "ground_truth": f"Protein {protein_id} has {annotation_type} annotation: {snippet}",
                "expected_answer_contains": [annotation_type, snippet.split(";")[0]],
                "expected_source_contains": ["protein_cards.jsonl"],
                "expected_table_contains": ["functional_annotations", "proteins"],
                "expected_primary_ids": [protein_id, ann.get("annotation_id")],
                "tags": ["schema_tables", "annotation"],
                "query_type": "annotation",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def _provenance_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    sources_by_id = {row["source_id"]: row for row in _records(tables, "sources") if row.get("source_id")}
    rows = []
    for interaction in _records(tables, "interactions"):
        source = sources_by_id.get(interaction.get("source_id"))
        if not interaction.get("interaction_id") or not source:
            continue
        source_ref = _pick(source.get("title"), source.get("identifier"))
        if not source.get("source_id") or not source_ref:
            continue
        rows.append(
            {
                "question": f"What source is linked to interaction {interaction.get('interaction_id')}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Interaction {interaction.get('interaction_id')} is linked to source {source.get('source_id')}: "
                    f"{source_ref}."
                ),
                "expected_answer_contains": [source.get("source_id"), source_ref],
                "expected_source_contains": ["interaction_cards.jsonl"],
                "expected_table_contains": ["interactions", "sources"],
                "expected_primary_ids": [interaction.get("interaction_id"), source.get("source_id")],
                "tags": ["schema_tables", "provenance"],
                "query_type": "provenance",
                "difficulty": "hard",
            }
        )
    return _choice(rows, count, seed)


def _structure_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    domains_by_id = {row["domain_id"]: row for row in _records(tables, "domains") if row.get("domain_id")}
    proteins_by_id = {row["protein_id"]: row for row in _records(tables, "proteins") if row.get("protein_id")}
    rows = []
    for struct in _records(tables, "structural_info"):
        domain = domains_by_id.get(struct.get("domain_id"))
        protein = proteins_by_id.get(domain.get("protein_id")) if domain else None
        if not domain or not protein or not struct.get("flexibility_desc"):
            continue
        protein_id = protein.get("protein_id")
        structure_id = _pick(struct.get("structure_id"))
        flexibility_desc = _pick(struct.get("flexibility_desc"))
        if not protein_id or not structure_id or not flexibility_desc:
            continue
        rows.append(
            {
                "question": f"What structural flexibility annotation is recorded for protein {protein_id}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Protein {protein_id} has flexibility annotation {flexibility_desc} "
                    f"on structure {structure_id}."
                ),
                "expected_answer_contains": [protein_id, flexibility_desc],
                "expected_source_contains": ["protein_cards.jsonl"],
                "expected_table_contains": ["structural_info", "proteins", "domains"],
                "expected_primary_ids": [protein_id, domain.get("domain_id"), structure_id],
                "tags": ["schema_tables", "structure"],
                "query_type": "structure",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def _digestive_assay_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    domains_by_id = {row["domain_id"]: row for row in _records(tables, "domains") if row.get("domain_id")}
    proteins_by_id = {row["protein_id"]: row for row in _records(tables, "proteins") if row.get("protein_id")}
    rows = []
    for assay in _records(tables, "digestive_assays"):
        domain = domains_by_id.get(assay.get("domain_id"))
        protein = proteins_by_id.get(domain.get("protein_id")) if domain else None
        if not domain or not protein:
            continue
        enzyme_name = _pick(assay.get("enzyme_name"))
        result_value = _pick(assay.get("result_value"))
        if not enzyme_name or not result_value:
            continue
        protein_display = _pick(protein.get("canonical_name"), protein.get("protein_id"))
        rows.append(
            {
                "question": f"What digestive assay result is recorded for {protein_display} against {enzyme_name}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"{protein_display} has digestive assay result {result_value} against {enzyme_name} "
                    f"with data type {_pick(assay.get('data_type')) or 'unknown'}."
                ),
                "expected_answer_contains": [protein_display, enzyme_name, result_value],
                "expected_source_contains": ["digestive_assays.csv"],
                "expected_table_contains": ["digestive_assays", "domains", "proteins"],
                "expected_primary_ids": [assay.get("assay_id"), domain.get("domain_id"), protein.get("protein_id")],
                "tags": ["schema_tables", "digestive_assay"],
                "query_type": "digestive_assay",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def _immunogenicity_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for item in _records(tables, "immunogenicity_summary"):
        sequence_id = _pick(item.get("structure_unique_sequence_id"))
        judgement = _pick(item.get("overall_final_judgement"))
        confidence = _pick(item.get("overall_confidence_level"))
        if not sequence_id or not judgement:
            continue
        rows.append(
            {
                "question": f"What immunogenicity judgement is recorded for sequence {sequence_id}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Sequence {sequence_id} has overall immunogenicity judgement {judgement} "
                    f"with confidence level {confidence or 'unknown'}."
                ),
                "expected_answer_contains": [sequence_id, judgement, confidence or "unknown"],
                "expected_source_contains": ["immunogenicity_summary.csv"],
                "expected_table_contains": ["immunogenicity_summary"],
                "expected_primary_ids": [item.get("protein_row_id"), sequence_id],
                "tags": ["schema_tables", "immunogenicity"],
                "query_type": "immunogenicity",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def _loop_annotation_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for item in _records(tables, "loop_annotations"):
        loop_id = _pick(item.get("loop_id"))
        sequence_id = _pick(item.get("structure_unique_sequence_id"))
        label = _pick(item.get("loop_label"))
        if not loop_id or not sequence_id or not label:
            continue
        manual_decision = _pick(item.get("loop_manual_decision")) or "unknown"
        confidence_tier = _pick(item.get("loop_confidence_tier")) or "unknown"
        rows.append(
            {
                "question": f"What loop annotation is recorded for loop {loop_id} in sequence {sequence_id}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Loop {loop_id} in sequence {sequence_id} is annotated as {label}, "
                    f"with manual decision {manual_decision} and confidence tier {confidence_tier}."
                ),
                "expected_answer_contains": [loop_id, label, manual_decision],
                "expected_source_contains": ["loop_annotations.csv"],
                "expected_table_contains": ["loop_annotations"],
                "expected_primary_ids": [loop_id, item.get("protein_row_id"), sequence_id],
                "tags": ["schema_tables", "loop_annotation"],
                "query_type": "loop_annotation",
                "difficulty": "hard",
            }
        )
    return _choice(rows, count, seed)


def _loop_flexibility_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for item in _records(tables, "loop_flexibility_results"):
        loop_id = _pick(item.get("loop_id"))
        consensus = _pick(item.get("flexibility_consensus_label"))
        predictor = _pick(item.get("itsflex_class"))
        if not loop_id or not consensus:
            continue
        conflict_flag = _pick(item.get("flexibility_conflict_flag")) or "unknown"
        rows.append(
            {
                "question": f"What flexibility assessment is reported for loop {loop_id}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Loop {loop_id} has flexibility consensus {consensus}, "
                    f"itsflex class {predictor or 'unknown'}, and conflict flag {conflict_flag}."
                ),
                "expected_answer_contains": [loop_id, consensus, predictor or "unknown"],
                "expected_source_contains": ["loop_flexibility_results.csv"],
                "expected_table_contains": ["loop_flexibility_results"],
                "expected_primary_ids": [loop_id, item.get("protein_row_id"), item.get("structure_unique_sequence_id")],
                "tags": ["schema_tables", "loop_flexibility"],
                "query_type": "loop_flexibility",
                "difficulty": "hard",
            }
        )
    return _choice(rows, count, seed)


def _protein_flexibility_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for item in _records(tables, "protein_flexibility_summary"):
        sequence_id = _pick(item.get("structure_unique_sequence_id"))
        if not sequence_id:
            continue
        total_loops = _pick(item.get("flexibility_loop_count_total"))
        conflict_loops = _pick(item.get("flexibility_conflict_loop_count"))
        dynamic_score = _pick(item.get("flexibility_dynamic_score_mean"))
        if not total_loops or not dynamic_score:
            continue
        rows.append(
            {
                "question": f"What protein-level flexibility summary is reported for sequence {sequence_id}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Sequence {sequence_id} has {total_loops} flexibility-tracked loops, "
                    f"dynamic score mean {dynamic_score}, and conflict loop count {conflict_loops or '0'}."
                ),
                "expected_answer_contains": [sequence_id, total_loops, dynamic_score],
                "expected_source_contains": ["protein_flexibility_summary.csv"],
                "expected_table_contains": ["protein_flexibility_summary"],
                "expected_primary_ids": [item.get("protein_row_id"), sequence_id],
                "tags": ["schema_tables", "protein_flexibility"],
                "query_type": "protein_flexibility",
                "difficulty": "hard",
            }
        )
    return _choice(rows, count, seed)


def _target_variant_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    concepts_by_id = {row["target_concept_id"]: row for row in _records(tables, "targets_conceptual") if row.get("target_concept_id")}
    rows = []
    for item in _records(tables, "target_species_variants"):
        target_variant_id = _pick(item.get("target_variant_id"))
        target_concept_id = _pick(item.get("target_concept_id"))
        concept = concepts_by_id.get(target_concept_id)
        species_name = _pick(item.get("species_name"))
        gene_name = _pick(item.get("gene_name_species"), (concept or {}).get("gene_name"), (concept or {}).get("protein_name"))
        if not target_variant_id or not species_name or not gene_name:
            continue
        rows.append(
            {
                "question": f"What target variant context is recorded for {target_variant_id}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Target variant {target_variant_id} is associated with species {species_name} "
                    f"and gene/protein label {gene_name}."
                ),
                "expected_answer_contains": [target_variant_id, species_name, gene_name],
                "expected_source_contains": ["target_species_variants.csv"],
                "expected_table_contains": ["target_species_variants", "targets_conceptual"],
                "expected_primary_ids": [target_variant_id, target_concept_id],
                "tags": ["schema_tables", "target_variant"],
                "query_type": "target_variant",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def _source_metadata_rows(tables: dict[str, pd.DataFrame], count: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    for source in _records(tables, "sources"):
        source_id = _pick(source.get("source_id"))
        source_ref = _pick(source.get("title"), source.get("identifier"))
        publication_date = _pick(source.get("publication_date"))
        source_type = _pick(source.get("source_type"))
        if not source_id or not source_ref:
            continue
        rows.append(
            {
                "question": f"What source metadata is recorded for source {source_id}?",
                "record_type": "jsonl",
                "ground_truth": (
                    f"Source {source_id} is a {source_type or 'unknown'} source titled or identified as {source_ref}, "
                    f"with publication date {publication_date or 'unknown'}."
                ),
                "expected_answer_contains": [source_id, source_ref, publication_date or "unknown"],
                "expected_source_contains": ["sources.csv"],
                "expected_table_contains": ["sources"],
                "expected_primary_ids": [source_id],
                "tags": ["schema_tables", "source_metadata"],
                "query_type": "source_metadata",
                "difficulty": "medium",
            }
        )
    return _choice(rows, count, seed)


def build_schema_benchmark_rows(schema_dir: Path, counts: dict[str, int] | None = None, seed: int = 42) -> list[dict[str, Any]]:
    tables = _load_tables(schema_dir)
    counts = counts or DEFAULT_COUNTS
    rows: list[dict[str, Any]] = []
    rows.extend(_protein_profile_rows(tables, counts.get("protein_profile", 0), seed + 1))
    rows.extend(_protein_identifier_rows(tables, counts.get("protein_identifier", 0), seed + 2))
    rows.extend(_interaction_rows(tables, counts.get("interaction_overview", 0), seed + 3))
    rows.extend(_affinity_rows(tables, counts.get("affinity", 0), seed + 4))
    rows.extend(_developability_rows(tables, counts.get("developability", 0), seed + 5))
    rows.extend(_annotation_rows(tables, counts.get("annotation", 0), seed + 6))
    rows.extend(_provenance_rows(tables, counts.get("provenance", 0), seed + 7))
    rows.extend(_structure_rows(tables, counts.get("structure", 0), seed + 8))
    rows.extend(_digestive_assay_rows(tables, counts.get("digestive_assay", 0), seed + 9))
    rows.extend(_immunogenicity_rows(tables, counts.get("immunogenicity", 0), seed + 10))
    rows.extend(_loop_annotation_rows(tables, counts.get("loop_annotation", 0), seed + 11))
    rows.extend(_loop_flexibility_rows(tables, counts.get("loop_flexibility", 0), seed + 12))
    rows.extend(_protein_flexibility_rows(tables, counts.get("protein_flexibility", 0), seed + 13))
    rows.extend(_target_variant_rows(tables, counts.get("target_variant", 0), seed + 14))
    rows.extend(_source_metadata_rows(tables, counts.get("source_metadata", 0), seed + 15))

    for idx, row in enumerate(rows, start=1):
        row["expected_answer_contains"] = _clean_expected(list(row.get("expected_answer_contains", [])))
        row["expected_source_contains"] = _clean_expected(list(row.get("expected_source_contains", [])))
        row["expected_table_contains"] = _clean_expected(list(row.get("expected_table_contains", [])))
        row["expected_primary_ids"] = _clean_expected(list(row.get("expected_primary_ids", [])))
        row["question_id"] = f"schema-{idx:03d}"
    return rows


def write_schema_benchmark_dataset(schema_dir: Path, output_path: Path, counts: dict[str, int] | None = None, seed: int = 42) -> dict[str, Any]:
    rows = build_schema_benchmark_rows(schema_dir, counts=counts, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    summary = {
        "schema_dir": str(schema_dir),
        "output_path": str(output_path),
        "rows": len(rows),
        "counts": counts or DEFAULT_COUNTS,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
