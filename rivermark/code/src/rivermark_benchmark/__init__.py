"""Rivermark native Isaac capture, validation, and dataset-admission utilities."""

from typing import Any

__all__ = [
    "DatasetCollector",
    "verify_candidate_episode",
    "verify_dataset_integrity",
    "ValidationIssue",
    "validate_episode_manifest",
    "FrameRecord",
    "IsaacCapture",
    "ReleaseManifestError",
    "ReleaseBuildError",
    "load_release_manifest",
    "build_release_manifest",
    "download_shards",
    "plan_download",
    "SearchMetrics",
    "score_search_episode",
    "bootstrap_summary",
    "EvaluatorSubmissionError",
    "SubmissionIssue",
    "EpisodeScore",
    "SubmissionReport",
    "validate_submission",
    "evaluate_submission",
    "evaluate_submission_file",
    "FixtureError",
    "CpuFixture",
    "CpuFixtureVerification",
    "create_cpu_fixture",
    "verify_cpu_fixture",
    "ResearcherEntryError",
    "run_researcher_smoke",
    "FailureRecord",
    "append_failure_record",
    "load_failure_ledger",
    "summarize_failure_ledger",
    "CrashLeftRecovery",
    "recover_crash_left_attempts",
    "SupplyChainError",
    "SupplyChainIssue",
    "validate_supply_chain_manifest",
    "verify_supply_chain_manifest",
    "supply_chain_sha256",
    "AssetProvenanceError",
    "AssetProvenanceReport",
    "inspect_usd",
    "inspect_many",
    "CollectionProtocolError",
    "CollectionProtocolIssue",
    "validate_collection_protocol",
    "load_collection_protocol",
    "protocol_sha256",
    "derive_episode_seed",
    "validate_collection_binding",
    "resolve_collection_binding",
    "required_paired_episodes",
    "coverage_report",
    "CONDITION_REALIZATION_SCHEMA",
    "condition_request_from_protocol",
    "validate_condition_request",
    "evaluate_condition_realization",
    "PreflightReport",
    "RuntimePreflightRequirements",
    "run_preflight",
    "AbiError",
    "AbiCompatibilityReport",
    "OBSERVATION_ABI_SCHEMA",
    "validate_observation_abi",
    "validate_formal_observation_abi",
    "assess_observation_abi_compatibility",
    "load_observation_abi",
    "observation_abi_sha256",
    "ProjectionError",
    "ZarrProjectionResult",
    "project_episode_to_zarr",
    "read_zarr_array",
    "read_zarr_array_independent",
    "read_zarr_array_external",
    "ParquetProjectionError",
    "ParquetProjectionResult",
    "project_development_capture_to_parquet",
    "read_development_parquet_table",
    "RldsProjectionError",
    "RldsProjectionResult",
    "RLDS_INTERCHANGE_SCHEMA",
    "project_state_action_to_rlds",
    "iter_rlds_records",
    "verify_rlds_interchange",
    "READINESS_SCHEMA",
    "ReleaseReadinessIssue",
    "ReleaseReadinessReport",
    "audit_release_readiness",
]
__version__ = "0.1.0-dev"


def __getattr__(name: str) -> Any:
    if name in {"DatasetCollector", "verify_candidate_episode", "verify_dataset_integrity"}:
        from .formal_dataset import DatasetCollector, verify_candidate_episode, verify_dataset_integrity

        return {
            "DatasetCollector": DatasetCollector,
            "verify_candidate_episode": verify_candidate_episode,
            "verify_dataset_integrity": verify_dataset_integrity,
        }[name]
    if name in {"ValidationIssue", "validate_episode_manifest"}:
        from .validate import ValidationIssue, validate_episode_manifest

        return {
            "ValidationIssue": ValidationIssue,
            "validate_episode_manifest": validate_episode_manifest,
        }[name]
    if name in {"FrameRecord", "IsaacCapture"}:
        from .isaac_dataset import FrameRecord, IsaacCapture

        return {"FrameRecord": FrameRecord, "IsaacCapture": IsaacCapture}[name]
    if name in {
        "ReleaseManifestError",
        "ReleaseBuildError",
        "load_release_manifest",
        "build_release_manifest",
        "download_shards",
        "plan_download",
    }:
        from .release_manifest import (
            ReleaseBuildError,
            ReleaseManifestError,
            build_release_manifest,
            download_shards,
            load_release_manifest,
            plan_download,
        )

        return {
            "ReleaseManifestError": ReleaseManifestError,
            "ReleaseBuildError": ReleaseBuildError,
            "load_release_manifest": load_release_manifest,
            "build_release_manifest": build_release_manifest,
            "download_shards": download_shards,
            "plan_download": plan_download,
        }[name]
    if name in {"FixtureError", "CpuFixture", "CpuFixtureVerification", "create_cpu_fixture", "verify_cpu_fixture"}:
        from .fixture import CpuFixture, CpuFixtureVerification, FixtureError, create_cpu_fixture, verify_cpu_fixture

        return {
            "FixtureError": FixtureError,
            "CpuFixture": CpuFixture,
            "CpuFixtureVerification": CpuFixtureVerification,
            "create_cpu_fixture": create_cpu_fixture,
            "verify_cpu_fixture": verify_cpu_fixture,
        }[name]
    if name in {"ResearcherEntryError", "run_researcher_smoke"}:
        from .researcher_entry import ResearcherEntryError, run_researcher_smoke

        return {"ResearcherEntryError": ResearcherEntryError, "run_researcher_smoke": run_researcher_smoke}[name]
    if name in {"CleanRoomSmokeError", "run_clean_room_smoke"}:
        raise AttributeError(name)
    if name in {"SearchMetrics", "score_search_episode", "bootstrap_summary"}:
        from .metrics import SearchMetrics, bootstrap_summary, score_search_episode

        return {
            "SearchMetrics": SearchMetrics,
            "score_search_episode": score_search_episode,
            "bootstrap_summary": bootstrap_summary,
        }[name]
    if name in {
        "EvaluatorSubmissionError",
        "SubmissionIssue",
        "EpisodeScore",
        "SubmissionReport",
        "validate_submission",
        "evaluate_submission",
        "evaluate_submission_file",
    }:
        from .evaluator import (
            EpisodeScore,
            EvaluatorSubmissionError,
            SubmissionIssue,
            SubmissionReport,
            evaluate_submission,
            evaluate_submission_file,
            validate_submission,
        )

        return {
            "EvaluatorSubmissionError": EvaluatorSubmissionError,
            "SubmissionIssue": SubmissionIssue,
            "EpisodeScore": EpisodeScore,
            "SubmissionReport": SubmissionReport,
            "validate_submission": validate_submission,
            "evaluate_submission": evaluate_submission,
            "evaluate_submission_file": evaluate_submission_file,
        }[name]
    if name in {"FailureRecord", "append_failure_record", "load_failure_ledger", "summarize_failure_ledger", "CrashLeftRecovery", "recover_crash_left_attempts"}:
        from .failure_ledger import CrashLeftRecovery, FailureRecord, append_failure_record, load_failure_ledger, recover_crash_left_attempts, summarize_failure_ledger

        return {
            "FailureRecord": FailureRecord,
            "append_failure_record": append_failure_record,
            "load_failure_ledger": load_failure_ledger,
            "summarize_failure_ledger": summarize_failure_ledger,
            "CrashLeftRecovery": CrashLeftRecovery,
            "recover_crash_left_attempts": recover_crash_left_attempts,
        }[name]
    if name in {"SupplyChainError", "SupplyChainIssue", "validate_supply_chain_manifest", "verify_supply_chain_manifest", "supply_chain_sha256"}:
        from .supply_chain import SupplyChainError, SupplyChainIssue, supply_chain_sha256, validate_supply_chain_manifest, verify_supply_chain_manifest

        return {
            "SupplyChainError": SupplyChainError,
            "SupplyChainIssue": SupplyChainIssue,
            "validate_supply_chain_manifest": validate_supply_chain_manifest,
            "verify_supply_chain_manifest": verify_supply_chain_manifest,
            "supply_chain_sha256": supply_chain_sha256,
        }[name]
    if name in {"AssetProvenanceError", "AssetProvenanceReport", "inspect_usd", "inspect_many"}:
        from .asset_provenance import (
            AssetProvenanceError,
            AssetProvenanceReport,
            inspect_many,
            inspect_usd,
        )

        return {
            "AssetProvenanceError": AssetProvenanceError,
            "AssetProvenanceReport": AssetProvenanceReport,
            "inspect_usd": inspect_usd,
            "inspect_many": inspect_many,
        }[name]
    if name in {
        "CollectionProtocolError",
        "CollectionProtocolIssue",
        "validate_collection_protocol",
        "load_collection_protocol",
        "protocol_sha256",
        "derive_episode_seed",
        "validate_collection_binding",
        "resolve_collection_binding",
        "required_paired_episodes",
        "coverage_report",
    }:
        from .collection_protocol import (
            CollectionProtocolError,
            CollectionProtocolIssue,
            coverage_report,
            derive_episode_seed,
            load_collection_protocol,
            protocol_sha256,
            resolve_collection_binding,
            required_paired_episodes,
            validate_collection_protocol,
            validate_collection_binding,
        )

        return {
            "CollectionProtocolError": CollectionProtocolError,
            "CollectionProtocolIssue": CollectionProtocolIssue,
            "validate_collection_protocol": validate_collection_protocol,
            "load_collection_protocol": load_collection_protocol,
            "protocol_sha256": protocol_sha256,
            "derive_episode_seed": derive_episode_seed,
            "validate_collection_binding": validate_collection_binding,
            "resolve_collection_binding": resolve_collection_binding,
            "required_paired_episodes": required_paired_episodes,
            "coverage_report": coverage_report,
        }[name]
    if name in {
        "CONDITION_REALIZATION_SCHEMA",
        "condition_request_from_protocol",
        "validate_condition_request",
        "evaluate_condition_realization",
    }:
        from .condition_realization import (
            CONDITION_REALIZATION_SCHEMA,
            condition_request_from_protocol,
            evaluate_condition_realization,
            validate_condition_request,
        )

        return {
            "CONDITION_REALIZATION_SCHEMA": CONDITION_REALIZATION_SCHEMA,
            "condition_request_from_protocol": condition_request_from_protocol,
            "validate_condition_request": validate_condition_request,
            "evaluate_condition_realization": evaluate_condition_realization,
        }[name]
    if name in {"PreflightReport", "RuntimePreflightRequirements", "run_preflight"}:
        from .preflight import PreflightReport, RuntimePreflightRequirements, run_preflight

        return {
            "PreflightReport": PreflightReport,
            "RuntimePreflightRequirements": RuntimePreflightRequirements,
            "run_preflight": run_preflight,
        }[name]
    if name in {"AbiError", "AbiCompatibilityReport", "OBSERVATION_ABI_SCHEMA", "validate_observation_abi", "validate_formal_observation_abi", "assess_observation_abi_compatibility", "load_observation_abi", "observation_abi_sha256"}:
        from .abi import AbiCompatibilityReport, AbiError, OBSERVATION_ABI_SCHEMA, assess_observation_abi_compatibility, load_observation_abi, observation_abi_sha256, validate_formal_observation_abi, validate_observation_abi

        return {
            "AbiError": AbiError,
            "AbiCompatibilityReport": AbiCompatibilityReport,
            "OBSERVATION_ABI_SCHEMA": OBSERVATION_ABI_SCHEMA,
            "validate_observation_abi": validate_observation_abi,
            "validate_formal_observation_abi": validate_formal_observation_abi,
            "assess_observation_abi_compatibility": assess_observation_abi_compatibility,
            "load_observation_abi": load_observation_abi,
            "observation_abi_sha256": observation_abi_sha256,
        }[name]
    if name in {
        "ProjectionError",
        "ZarrProjectionResult",
        "project_episode_to_zarr",
        "read_zarr_array",
        "read_zarr_array_independent",
        "read_zarr_array_external",
    }:
        from .research_projection import (
            ProjectionError,
            ZarrProjectionResult,
            project_episode_to_zarr,
            read_zarr_array,
            read_zarr_array_independent,
            read_zarr_array_external,
        )

        return {
            "ProjectionError": ProjectionError,
            "ZarrProjectionResult": ZarrProjectionResult,
            "project_episode_to_zarr": project_episode_to_zarr,
            "read_zarr_array": read_zarr_array,
            "read_zarr_array_independent": read_zarr_array_independent,
            "read_zarr_array_external": read_zarr_array_external,
        }[name]
    if name in {
        "ParquetProjectionError",
        "ParquetProjectionResult",
        "project_development_capture_to_parquet",
        "read_development_parquet_table",
    }:
        from .parquet_projection import (
            ParquetProjectionError,
            ParquetProjectionResult,
            project_development_capture_to_parquet,
            read_development_parquet_table,
        )

        return {
            "ParquetProjectionError": ParquetProjectionError,
            "ParquetProjectionResult": ParquetProjectionResult,
            "project_development_capture_to_parquet": project_development_capture_to_parquet,
            "read_development_parquet_table": read_development_parquet_table,
        }[name]
    if name in {
        "RldsProjectionError",
        "RldsProjectionResult",
        "RLDS_INTERCHANGE_SCHEMA",
        "project_state_action_to_rlds",
        "iter_rlds_records",
        "verify_rlds_interchange",
    }:
        from .rlds_projection import (
            RldsProjectionError,
            RldsProjectionResult,
            RLDS_INTERCHANGE_SCHEMA,
            iter_rlds_records,
            project_state_action_to_rlds,
            verify_rlds_interchange,
        )

        return {
            "RldsProjectionError": RldsProjectionError,
            "RldsProjectionResult": RldsProjectionResult,
            "RLDS_INTERCHANGE_SCHEMA": RLDS_INTERCHANGE_SCHEMA,
            "project_state_action_to_rlds": project_state_action_to_rlds,
            "iter_rlds_records": iter_rlds_records,
            "verify_rlds_interchange": verify_rlds_interchange,
        }[name]
    if name in {
        "SERVICE_RESULT_SCHEMA",
        "EvaluatorServiceError",
        "EvaluatorAuthenticationError",
        "EvaluatorRateLimitError",
        "LocalEvaluatorService",
        "verify_signed_result",
    }:
        raise AttributeError(name)
    if name in {"READINESS_SCHEMA", "ReleaseReadinessIssue", "ReleaseReadinessReport", "audit_release_readiness"}:
        from .release_readiness import READINESS_SCHEMA, ReleaseReadinessIssue, ReleaseReadinessReport, audit_release_readiness

        return {
            "READINESS_SCHEMA": READINESS_SCHEMA,
            "ReleaseReadinessIssue": ReleaseReadinessIssue,
            "ReleaseReadinessReport": ReleaseReadinessReport,
            "audit_release_readiness": audit_release_readiness,
        }[name]
    raise AttributeError(name)
