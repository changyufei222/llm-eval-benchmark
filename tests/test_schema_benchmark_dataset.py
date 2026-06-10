from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines.schema_benchmark_dataset import build_schema_benchmark_rows


class SchemaBenchmarkDatasetTests(unittest.TestCase):
    def test_build_rows_generates_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_dir = Path(temp_dir) / "schema_tables"
            schema_dir.mkdir()

            pd.DataFrame(
                [{"protein_id": "PROT-1", "canonical_name": "ITI-D2", "organism": "Homo sapiens", "description": "Kunitz binder"}]
            ).to_csv(schema_dir / "proteins.csv", index=False)
            pd.DataFrame(
                [{"identifier_id": "ID-1", "protein_id": "PROT-1", "id_type": "UniProt", "id_value": "P00001", "status": "Reviewed"}]
            ).to_csv(schema_dir / "protein_identifiers.csv", index=False)
            pd.DataFrame(
                [{"domain_id": "TBP-1", "protein_id": "PROT-1", "domain_name": "Example Domain", "scaffold_type": "kunitz", "is_engineered": "True"}]
            ).to_csv(schema_dir / "domains.csv", index=False)
            pd.DataFrame(
                [{"interaction_id": "INT-1", "domain_id": "TBP-1", "target_variant_id": "TVAR-1", "source_id": "SRC-1", "interaction_class": "binding", "is_inhibitory": "True"}]
            ).to_csv(schema_dir / "interactions.csv", index=False)
            pd.DataFrame(
                [{"affinity_id": "AFF-1", "interaction_id": "INT-1", "determination_method": "Experimental", "value": "6.2", "unit": "nM"}]
            ).to_csv(schema_dir / "affinity_data.csv", index=False)
            pd.DataFrame(
                [{"cmc_id": "CMC-1", "domain_id": "TBP-1", "expression_system": "E. coli", "oral_properties": "Medium", "stability_rating": "Stable"}]
            ).to_csv(schema_dir / "cmc_data.csv", index=False)
            pd.DataFrame(
                [{"annotation_id": "ANN-1", "protein_id": "PROT-1", "annotation_type": "Keywords", "annotation_value": "Toxin"}]
            ).to_csv(schema_dir / "functional_annotations.csv", index=False)
            pd.DataFrame(
                [{"structure_id": "STR-1", "domain_id": "TBP-1", "flexibility_desc": "A:5-10"}]
            ).to_csv(schema_dir / "structural_info.csv", index=False)
            pd.DataFrame(
                [{"target_variant_id": "TVAR-1", "target_concept_id": "TGT-1", "species_name": "Homo sapiens", "gene_name_species": "ELANE (human)"}]
            ).to_csv(schema_dir / "target_species_variants.csv", index=False)
            pd.DataFrame(
                [{"target_concept_id": "TGT-1", "gene_name": "ELANE", "protein_name": "Neutrophil elastase", "description": "Target"}]
            ).to_csv(schema_dir / "targets_conceptual.csv", index=False)
            pd.DataFrame(
                [{"source_id": "SRC-1", "source_type": "Literature", "title": "Example paper", "identifier": "PMID:1"}]
            ).to_csv(schema_dir / "sources.csv", index=False)
            pd.DataFrame(
                [{"assay_id": "ASSAY-1", "domain_id": "TBP-1", "enzyme_name": "trypsin", "result_value": "75.0", "data_type": "Experimental"}]
            ).to_csv(schema_dir / "digestive_assays.csv", index=False)
            pd.DataFrame(
                [{"protein_row_id": "7", "structure_unique_sequence_id": "SEQ-1", "overall_final_judgement": "High", "overall_confidence_score_0_100": "90.0", "overall_confidence_level": "High"}]
            ).to_csv(schema_dir / "immunogenicity_summary.csv", index=False)
            pd.DataFrame(
                [{"loop_id": "SEQ-1:A:L2", "protein_row_id": "7", "structure_unique_sequence_id": "SEQ-1", "loop_label": "replaceable_scaffold_loop_candidate", "loop_manual_decision": "approved", "loop_confidence_tier": "medium"}]
            ).to_csv(schema_dir / "loop_annotations.csv", index=False)
            pd.DataFrame(
                [{"loop_id": "SEQ-1:A:L2", "protein_row_id": "7", "structure_unique_sequence_id": "SEQ-1", "flexibility_consensus_label": "intermediate", "itsflex_class": "high_confidence_rigid", "flexibility_conflict_flag": "False"}]
            ).to_csv(schema_dir / "loop_flexibility_results.csv", index=False)
            pd.DataFrame(
                [{"protein_row_id": "7", "structure_unique_sequence_id": "SEQ-1", "flexibility_loop_count_total": "2", "flexibility_dynamic_score_mean": "0.17", "flexibility_conflict_loop_count": "1"}]
            ).to_csv(schema_dir / "protein_flexibility_summary.csv", index=False)
            pd.DataFrame(
                [{"target_variant_id": "TVAR-1", "target_concept_id": "TGT-1", "species_name": "Homo sapiens", "gene_name_species": "ELANE"}]
            ).to_csv(schema_dir / "target_species_variants.csv", index=False)
            pd.DataFrame(
                [{"target_concept_id": "TGT-1", "gene_name": "ELANE", "protein_name": "Neutrophil elastase", "description": "Target"}]
            ).to_csv(schema_dir / "targets_conceptual.csv", index=False)

            rows = build_schema_benchmark_rows(
                schema_dir,
                counts={
                    "protein_profile": 1,
                    "protein_identifier": 1,
                    "interaction_overview": 1,
                    "affinity": 1,
                    "developability": 1,
                    "annotation": 1,
                    "provenance": 1,
                    "structure": 1,
                    "digestive_assay": 1,
                    "immunogenicity": 1,
                    "loop_annotation": 1,
                    "loop_flexibility": 1,
                    "protein_flexibility": 1,
                    "target_variant": 1,
                    "source_metadata": 1,
                },
                seed=1,
            )

        self.assertEqual(len(rows), 15)
        self.assertTrue(all("expected_table_contains" in row for row in rows))
        self.assertTrue(all("expected_primary_ids" in row for row in rows))
        self.assertTrue(all("question_id" in row for row in rows))

    def test_developability_row_omits_placeholder_na_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            schema_dir = Path(temp_dir) / "schema_tables"
            schema_dir.mkdir()

            pd.DataFrame(
                [{"protein_id": "PROT-1", "canonical_name": "ITI-D2", "organism": "Homo sapiens"}]
            ).to_csv(schema_dir / "proteins.csv", index=False)
            pd.DataFrame(
                [{"identifier_id": "ID-1", "protein_id": "PROT-1", "id_type": "UniProt", "id_value": "P00001"}]
            ).to_csv(schema_dir / "protein_identifiers.csv", index=False)
            pd.DataFrame(
                [{"domain_id": "TBP-1", "protein_id": "PROT-1", "domain_name": "Example Domain", "scaffold_type": "kunitz"}]
            ).to_csv(schema_dir / "domains.csv", index=False)
            pd.DataFrame(
                [{"interaction_id": "INT-1", "domain_id": "TBP-1", "target_variant_id": "TVAR-1", "source_id": "SRC-1", "interaction_class": "binding", "is_inhibitory": "True"}]
            ).to_csv(schema_dir / "interactions.csv", index=False)
            pd.DataFrame(
                [{"affinity_id": "AFF-1", "interaction_id": "INT-1", "determination_method": "Experimental", "value": "6.2", "unit": "nM"}]
            ).to_csv(schema_dir / "affinity_data.csv", index=False)
            pd.DataFrame(
                [{"cmc_id": "CMC-1", "domain_id": "TBP-1", "expression_system": "", "oral_properties": "Medium", "stability_rating": "Stable"}]
            ).to_csv(schema_dir / "cmc_data.csv", index=False)
            pd.DataFrame(
                [{"annotation_id": "ANN-1", "protein_id": "PROT-1", "annotation_type": "Keywords", "annotation_value": "Toxin"}]
            ).to_csv(schema_dir / "functional_annotations.csv", index=False)
            pd.DataFrame(
                [{"structure_id": "STR-1", "domain_id": "TBP-1", "flexibility_desc": "A:5-10"}]
            ).to_csv(schema_dir / "structural_info.csv", index=False)
            pd.DataFrame(
                [{"target_variant_id": "TVAR-1", "target_concept_id": "TGT-1", "species_name": "Homo sapiens", "gene_name_species": "ELANE (human)"}]
            ).to_csv(schema_dir / "target_species_variants.csv", index=False)
            pd.DataFrame(
                [{"target_concept_id": "TGT-1", "gene_name": "ELANE", "protein_name": "Neutrophil elastase", "description": "Target"}]
            ).to_csv(schema_dir / "targets_conceptual.csv", index=False)
            pd.DataFrame(
                [{"source_id": "SRC-1", "source_type": "Literature", "title": "Example paper", "identifier": "PMID:1"}]
            ).to_csv(schema_dir / "sources.csv", index=False)

            rows = build_schema_benchmark_rows(
                schema_dir,
                counts={
                    "protein_profile": 0,
                    "protein_identifier": 0,
                    "interaction_overview": 0,
                    "affinity": 0,
                    "developability": 1,
                    "annotation": 0,
                    "provenance": 0,
                    "structure": 0,
                },
                seed=1,
            )

        self.assertEqual(len(rows), 1)
        self.assertNotIn("n/a", rows[0]["ground_truth"].lower())
        self.assertIn("Medium", rows[0]["ground_truth"])


if __name__ == "__main__":
    unittest.main()
