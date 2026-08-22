from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from aerocity_method.archives.qd import ArchiveSpec, DescriptorAxis, Elite, QDArchive
from aerocity_method.contracts.io import canonical_sha256
from aerocity_method.contracts.privacy import PublicBoundaryError, walk_public_payload
from aerocity_method.contracts.snapshot import build_source_snapshot
from aerocity_method.fragments.provenance import evaluate_provenance
from aerocity_method.learning.rb_sf_sac import RBSFSAC, RBSFSACConfig
from aerocity_method.runtime.process_boundary import (
    build_public_request,
    sanitized_method_environment,
    validate_public_request,
)


@pytest.mark.parametrize(
    "field",
    (
        "evaluator_private",
        "oracle_outcome",
        "target_coordinates",
        "target_truth",
        "fault_spec",
        "failed_agent_ids",
        "blind_identity",
        "split_id",
        "seed",
        "episode_seed",
        "target_family",
        "target_distance",
        "complete_mesh",
        "private_esdf",
        "evaluator_esdf",
    ),
)
def test_private_or_experimental_identity_fields_are_rejected_at_public_boundary(field):
    with pytest.raises(PublicBoundaryError):
        walk_public_payload({"nested": {field: "do-not-expose"}})


def test_nested_canary_is_rejected_before_method_start():
    with pytest.raises(PublicBoundaryError):
        walk_public_payload(
            {"safe_name": ["safe", {"still_safe": "blind-canary-007"}]},
            canaries=("blind-canary-007",),
        )


def test_environment_removes_private_keys_and_canary_values():
    cleaned = sanitized_method_environment(
        {
            "PATH": "safe",
            "EVALUATOR_SECRET": "x",
            "PUBLIC_VALUE": "prefix-blind-canary-007-suffix",
        },
        canaries=("blind-canary-007",),
    )
    assert cleaned == {"PATH": "safe"}


def test_staged_snapshot_rejects_private_canary(tmp_path):
    (tmp_path / "safe.py").write_text("TOKEN = 'blind-canary-007'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="private canary"):
        build_source_snapshot(
            tmp_path,
            ("safe.py",),
            canaries=("blind-canary-007",),
        )


def test_request_rejects_integer_mask_even_with_recomputed_outer_hash(context, manifests):
    request = build_public_request(context, manifests, (True, False, True))
    request["legal_mask"] = [1, 0, 1]
    request["request_hash"] = canonical_sha256(
        {key: value for key, value in request.items() if key != "request_hash"}
    )
    with pytest.raises(ValueError, match="legal mask"):
        validate_public_request(request)


def test_request_rejects_context_rebinding_with_recomputed_outer_hash(context, manifests):
    request = build_public_request(context, manifests, (True, False, True))
    request["context"]["context_id"] = "forged-context"
    request["request_hash"] = canonical_sha256(
        {key: value for key, value in request.items() if key != "request_hash"}
    )
    with pytest.raises(ValueError, match="rebound"):
        validate_public_request(request)


def test_request_rejects_candidate_replacement_with_recomputed_outer_hash(context, manifests):
    request = build_public_request(context, manifests, (True, False, True))
    request["candidates"][0]["quality_hint"] = 999.0
    request["request_hash"] = canonical_sha256(
        {key: value for key, value in request.items() if key != "request_hash"}
    )
    with pytest.raises(ValueError, match="candidate content hash"):
        validate_public_request(request)


def test_source_observation_cannot_be_rebound_across_agent(manifests, outcomes, token):
    outcome = replace(outcomes[1], source_observation_agent_id="uav-other")
    decision = evaluate_provenance(manifests[0].fragments[1], outcome, token, manifests[0])
    assert decision.reason_code == "SOURCE_OBSERVATION_IDENTITY_MISMATCH"


def test_rl_checkpoint_hash_rejects_parameter_replacement():
    config = RBSFSACConfig(context_dim=2, candidate_dim=2, hidden_dim=16)
    source = RBSFSAC(config, seed=13)
    checkpoint = source.state_dict()
    first_parameter = next(iter(checkpoint["actor"].values()))
    first_parameter.add_(1.0)
    target = RBSFSAC(config, seed=14)
    with pytest.raises(ValueError, match="content hash mismatch"):
        target.load_state_dict(checkpoint)


def test_archive_checkpoint_hash_rejects_elite_replacement(manifests):
    archive = QDArchive(
        ArchiveSpec((DescriptorAxis("x", 0.0, 1.0, 2), DescriptorAxis("z", 0.0, 1.0, 2)))
    )
    archive.add_or_update(
        Elite(
            candidate_id=manifests[0].candidate_id,
            manifest_hash=manifests[0].manifest_hash,
            behavior_hash=canonical_sha256({"behavior": 1}),
            realised_descriptor=(0.2, 0.2),
            quality=1.0,
            cost=0.1,
            feasible=True,
            source="test",
        )
    )
    checkpoint = copy.deepcopy(archive.state_dict())
    checkpoint["cells"][0]["elite"]["quality"] = 999.0
    with pytest.raises(ValueError, match="content hash mismatch"):
        QDArchive.from_state_dict(checkpoint)
