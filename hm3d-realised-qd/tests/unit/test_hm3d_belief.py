from __future__ import annotations

from aerocity_method.runtime.hm3d_belief import (
    FREE,
    OCCUPIED,
    PublicRangeRayOutcome,
    SparseVoxelBelief,
)


def test_supercover_keeps_the_measured_hit_voxel_occupied() -> None:
    belief = SparseVoxelBelief("scene0", "uav0", 0.25)
    endpoint_m = (1.125, 1.125, 1.125)
    outcome = PublicRangeRayOutcome(
        "diagonal-hit",
        "uav0",
        1.0,
        (0.125, 0.125, 0.125),
        endpoint_m,
        True,
    )

    assert belief.integrate_ray(outcome) is True
    assert belief.state(belief.world_to_voxel(endpoint_m)) == OCCUPIED
    assert belief.integrate_ray(outcome) is False


def test_confirmed_free_voxel_is_never_downgraded_by_a_later_ray() -> None:
    """Explored free volume must be monotone.

    A sparse ray can graze an obstacle edge and terminate inside a voxel
    that an earlier pass-through ray already proved free.  The continuous
    PhysX route guard physically admitted that voxel, so downgrading it to
    occupied would shrink the explored set and break the monotone metric
    contract.  Confirmed free is sticky.
    """

    belief = SparseVoxelBelief("scene0", "uav0", 0.25)
    key = belief.world_to_voxel((1.125, 1.125, 1.125))
    belief.set_state(key, FREE)
    assert belief.state(key) == FREE

    hit = PublicRangeRayOutcome(
        "later-grazing-hit",
        "uav0",
        2.0,
        (0.125, 0.125, 0.125),
        (1.125, 1.125, 1.125),
        True,
    )
    belief.integrate_ray(hit)
    assert belief.state(key) == FREE
