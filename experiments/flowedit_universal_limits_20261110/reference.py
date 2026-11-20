"""FlowEdit sign-paired queries. CPU reference; no model or scheduler mutation."""

import numpy as np


def flowedit_increment(source_velocity, target_velocity, x, y, t, noise):
    """Positive-step increment v_source - v_target; conditions live in callables."""
    source_query = (1.0 - t) * x + t * noise
    target_query = source_query + y - x
    return source_velocity(source_query, t) - target_velocity(target_query, t)


def paired_increment(source_velocity, target_velocity, x, y, t, noise):
    """Keep x/y/t fixed for both calls. Caller supplies fresh Gaussian noise."""
    plus = flowedit_increment(source_velocity, target_velocity, x, y, t, noise)
    minus = flowedit_increment(source_velocity, target_velocity, x, y, t, -noise)
    return 0.5 * (plus + minus)


def parity_parts(values, reflected_values):
    values = np.asarray(values, dtype=np.float64)
    reflected_values = np.asarray(reflected_values, dtype=np.float64)
    return (values + reflected_values) / 2, (values - reflected_values) / 2
