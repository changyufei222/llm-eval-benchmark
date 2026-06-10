from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from pipelines.prepare_benchmark_dataset import build_fixed_benchmark_rows, write_fixed_benchmark_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


class PrepareBenchmarkDatasetTests(unittest.TestCase):
    def _write_schema_tables(self, schema_dir: Path) -> None:
        pd.DataFrame(
            [
                {"protein_id": "PROT-1", "canonical_name": "ITI-D2", "organism": "Homo sapiens"},
                {"protein_id": "PROT-2", "canonical_name": "Delta-toxin", "organism": "Dendroaspis angusticeps"},
            ]
        ).to_csv(schema_dir / "proteins.csv", index=False)
        pd.DataFrame(
            [
                {"identifier_id": "ID-1", "protein_id": "PROT-1", "id_type": "UniProt", "id_value": "P00001"},
            ]
        ).to_csv(schema_dir / "protein_identifiers.csv", index=False)
        pd.DataFrame(
            [
                {"domain_id": "DOM-1", "protein_id": "PROT-1", "domain_name": "Example Domain", "scaffold_type": "kunitz"},
                {"domain_id": "DOM-2", "protein_id": "PROT-2", "domain_name": "Snake Toxin Domain", "scaffold_type": "kunitz"},
            ]
        ).to_csv(schema_dir / "domains.csv", index=False)
        pd.DataFrame(
            [
                {"interaction_id": "INT-1", "domain_id": "DOM-1", "source_id": "SRC-1", "is_inhibitory": "True"},
            ]
        ).to_csv(schema_dir / "interactions.csv", index=False)
        pd.DataFrame(
            [
                {"affinity_id": "AFF-1", "interaction_id": "INT-1", "determination_method": "Experimental", "value": "6.2", "unit": "nM"},
            ]
        ).to_csv(schema_dir / "affinity_data.csv", index=False)
        pd.DataFrame(
            [
                {"cmc_id": "CMC-1", "domain_id": "DOM-1", "expression_system": "E. coli", "oral_properties": "Medium", "stability_rating": "Stable"},
            ]
        ).to_csv(schema_dir / "cmc_data.csv", index=False)
        pd.DataFrame(
            [
                {"annotation_id": "ANN-1", "protein_id": "PROT-1", "annotation_type": "Keywords", "annotation_value": "Toxin"},
            ]
        ).to_csv(schema_dir / "functional_annotations.csv", index=False)
        pd.DataFrame(
            [
                {"structure_id": "STR-1", "domain_id": "DOM-1", "flexibility_desc": "A:5-10"},
            ]
        ).to_csv(schema_dir / "structural_info.csv", index=False)
        pd.DataFrame(
            [
                {"source_id": "SRC-1", "source_type": "Literature", "title": "Example paper", "identifier": "PMID:1"},
            ]
        ).to_csv(schema_dir / "sources.csv", index=False)

    def test_build_fixed_rows_combines_curated_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            ragppi_path = tmp / "ragppi_eval.jsonl"
            doc_design_path = tmp / "doc_design_eval.jsonl"
            schema_dir = tmp / "schema_tables"
            schema_dir.mkdir()
            self._write_schema_tables(schema_dir)

            _write_jsonl(
                ragppi_path,
                [
                    {
                        "question": "What is reported about the interaction between KLK7 and CDSN?",
                        "record_type": "csv",
                        "ground_truth": "KLK7 cleaves CDSN.",
                        "expected_answer_contains": ["KLK7", "CDSN"],
                        "expected_source_contains": ["Set_1"],
                        "tags": ["ragppi", "gold-standard"],
                    },
                    {
                        "question": "What is reported about the interaction between MED15 and sbp-1?",
                        "record_type": "csv",
                        "ground_truth": "MED15 works with sbp-1.",
                        "expected_answer_contains": ["MED15", "sbp-1"],
                        "expected_source_contains": ["Set_2"],
                        "tags": ["ragppi", "gold-standard"],
                    },
                ],
            )
            _write_jsonl(
                doc_design_path,
                [
                    {
                        "question": "What is the unified execution chain used by the benchmark-ready RAG system?",
                        "record_type": "text",
                        "ground_truth": "The benchmark-ready RAG system uses planner, retrieve, answer, and quality as its unified execution chain.",
                        "expected_answer_contains": ["planner", "retrieve", "quality"],
                        "expected_source_contains": ["rag_architecture_overview.md"],
                        "tags": ["doc_design", "architecture"],
                    },
                    {
                        "question": "What is reported about the interaction between KLK7 and CDSN?",
                        "record_type": "csv",
                        "ground_truth": "Duplicate csv row should not be used for doc questions.",
                        "expected_answer_contains": ["KLK7", "CDSN"],
                        "expected_source_contains": ["Set_1"],
                        "tags": ["duplicate"],
                    },
                ],
            )

            rows = build_fixed_benchmark_rows(
                ragppi_path=ragppi_path,
                doc_design_path=doc_design_path,
                schema_dir=schema_dir,
                ragppi_count=2,
                doc_count=1,
                schema_counts={
                    "protein_profile": 1,
                    "protein_identifier": 0,
                    "interaction_overview": 0,
                    "affinity": 0,
                    "developability": 0,
                    "annotation": 0,
                    "provenance": 0,
                    "structure": 0,
                    "digestive_assay": 0,
                    "immunogenicity": 0,
                    "loop_annotation": 0,
                    "loop_flexibility": 0,
                    "protein_flexibility": 0,
                    "target_variant": 0,
                    "source_metadata": 0,
                },
            )

        self.assertEqual(len(rows), 4)
        self.assertEqual([row["question_id"] for row in rows], ["fixed-001", "fixed-002", "fixed-003", "fixed-004"])
        self.assertEqual({row["benchmark_source_group"] for row in rows}, {"ragppi_gold", "doc_design", "schema_tables"})
        self.assertEqual(sum(1 for row in rows if row["record_type"] == "text"), 1)
        self.assertTrue(all("benchmark_fixed_set" in row["tags"] for row in rows))

    def test_write_fixed_dataset_creates_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            ragppi_path = tmp / "ragppi_eval.jsonl"
            doc_design_path = tmp / "doc_design_eval.jsonl"
            schema_dir = tmp / "schema_tables"
            schema_dir.mkdir()
            self._write_schema_tables(schema_dir)

            _write_jsonl(
                ragppi_path,
                [
                    {
                        "question": "What is reported about the interaction between KLK7 and CDSN?",
                        "record_type": "csv",
                        "ground_truth": "KLK7 cleaves CDSN.",
                        "expected_answer_contains": ["KLK7", "CDSN"],
                        "expected_source_contains": ["Set_1"],
                        "tags": ["ragppi", "gold-standard"],
                    },
                ],
            )
            _write_jsonl(
                doc_design_path,
                [
                    {
                        "question": "What is the primary role of RAGAS in the benchmark pipeline?",
                        "record_type": "text",
                        "ground_truth": "RAGAS serves as the primary benchmark evaluation framework for fixed-set RAG versus Direct comparison.",
                        "expected_answer_contains": ["RAGAS", "primary", "evaluation"],
                        "expected_source_contains": ["benchmark_phase1_protocol.md"],
                        "tags": ["doc_design", "benchmark_protocol"],
                    },
                ],
            )

            output_path = tmp / "fbtp_eval_fixed.jsonl"
            summary = write_fixed_benchmark_dataset(
                output_path=output_path,
                ragppi_path=ragppi_path,
                doc_design_path=doc_design_path,
                schema_dir=schema_dir,
                ragppi_count=1,
                doc_count=1,
                schema_counts={
                    "protein_profile": 1,
                    "protein_identifier": 0,
                    "interaction_overview": 0,
                    "affinity": 0,
                    "developability": 0,
                    "annotation": 0,
                    "provenance": 0,
                    "structure": 0,
                    "digestive_assay": 0,
                    "immunogenicity": 0,
                    "loop_annotation": 0,
                    "loop_flexibility": 0,
                    "protein_flexibility": 0,
                    "target_variant": 0,
                    "source_metadata": 0,
                },
            )

            written_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            summary_path = output_path.with_suffix(".summary.json")
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(len(written_rows), 3)
        self.assertEqual(loaded_summary["rows"], 3)
        self.assertEqual(loaded_summary["source_group_counts"]["ragppi_gold"], 1)
        self.assertEqual(loaded_summary["source_group_counts"]["doc_design"], 1)
        self.assertEqual(loaded_summary["source_group_counts"]["schema_tables"], 1)

    def test_write_fixed_dataset_supports_40_20_60_balanced_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp = Path(temp_dir)
            ragppi_path = tmp / "ragppi_eval.jsonl"
            doc_design_path = tmp / "doc_design_eval.jsonl"
            schema_dir = tmp / "schema_tables"
            schema_dir.mkdir()
            self._write_schema_tables(schema_dir)

            _write_jsonl(
                ragppi_path,
                [
                    {
                        "question": f"What is reported about interaction pair {idx}?",
                        "record_type": "csv",
                        "ground_truth": f"Interaction pair {idx} is supported by curated evidence.",
                        "expected_answer_contains": [f"pair {idx}"],
                        "expected_source_contains": [f"Set_{idx:03d}"],
                        "tags": ["ragppi", "gold-standard"],
                    }
                    for idx in range(1, 41)
                ],
            )
            _write_jsonl(
                doc_design_path,
                [
                    {
                        "question": f"What design principle {idx} is documented for the RAG system?",
                        "record_type": "text",
                        "ground_truth": f"Design principle {idx} is grounded in the authoritative benchmark design corpus.",
                        "expected_answer_contains": [f"principle {idx}"],
                        "expected_source_contains": ["rag_architecture_overview.md"],
                        "tags": ["doc_design", "authoritative"],
                    }
                    for idx in range(1, 21)
                ],
            )

            summary = write_fixed_benchmark_dataset(
                output_path=tmp / "balanced.jsonl",
                ragppi_path=ragppi_path,
                doc_design_path=doc_design_path,
                schema_dir=schema_dir,
                ragppi_count=40,
                doc_count=20,
                schema_counts={
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
                },
            )

        self.assertEqual(summary["ragppi_count"], 40)
        self.assertEqual(summary["doc_count"], 20)
        self.assertEqual(summary["schema_counts"]["protein_profile"], 8)
        self.assertEqual(summary["schema_counts"]["source_metadata"], 2)
        self.assertEqual(summary["source_group_counts"]["ragppi_gold"], 40)
        self.assertEqual(summary["source_group_counts"]["doc_design"], 20)
        self.assertGreater(summary["source_group_counts"]["schema_tables"], 0)


if __name__ == "__main__":
    unittest.main()
