"""
Core - Ядро библиотеки

Универсальная математика, не зависящая от домена.
НЕ ТРОГАТЬ без веской причины.

Содержит:
- geometry: Sphere, d_geo, centroid, slerp
- shape: normalize, Shape
- metrics: d_eff, kl_divergence
"""

from .geometry import (
    Sphere,
    d_geo,
    d_geo_deg,
    d_geo_batch,
    d_geo_pairwise,
    centroid,
    slerp,
    frechet_mean,
    distance_matrix,
)

from .shape import (
    normalize,
    normalize_with_energy,
    denormalize,
    Shape,
    to_shape,
    dict_to_shape,
    concat_and_normalize,
)

from .metrics import (
    d_eff,
    d_eff_spectrum,
    d_eff_at_thresholds,
    kl_divergence,
    kl_from_geodesic,
    geodesic_from_kl,
    variance_on_sphere,
    std_on_sphere,
    separation_ratio,
)

__all__ = [
    # Geometry
    'Sphere',
    'd_geo',
    'd_geo_deg',
    'd_geo_batch',
    'd_geo_pairwise',
    'centroid',
    'slerp',
    'frechet_mean',
    'distance_matrix',
    
    # Shape
    'normalize',
    'normalize_with_energy',
    'denormalize',
    'Shape',
    'to_shape',
    'dict_to_shape',
    'concat_and_normalize',
    
    # Metrics
    'd_eff',
    'd_eff_spectrum',
    'd_eff_at_thresholds',
    'kl_divergence',
    'kl_from_geodesic',
    'geodesic_from_kl',
    'variance_on_sphere',
    'std_on_sphere',
    'separation_ratio',
]
