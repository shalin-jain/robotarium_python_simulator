import unittest
import jax.numpy as jnp
import numpy as np
from rps_jax.utilities.barrier_certificates2 import create_robust_barriers
# from rps.utilities.barrier_certificates2 import create_robust_barriers as create_robust_barriers_py
# jax.config.update("jax_enable_x64", True)

class TestRobustBarriers(unittest.TestCase):
    """unit tests for test_barrier_certificates2.py"""

    def test_safety_radius_violation_prevention(self):
        barrier_fn = create_robust_barriers(base_length=0.1, wheel_radius=0.01, safety_radius = 0.1, gamma=100)

        # Two robots moving directly toward each other
        x = jnp.array([
            [0.0, 0.2],  # x positions
            [0.0, 0.0],  # y positions
            [0.0, jnp.pi]  # orientations (facing each other)
        ])
        dxu = jnp.array([
            [0.1, -0.1],  # Velocities that would lead to collision
            [0.0, 0.0]
        ])

        dxu_safe = barrier_fn(dxu, x, [])

        # Compute new positions after applying safe velocities
        x_new = x[:2, :] + dxu_safe * 0.01  # Simulating a small time step
        distance_after = jnp.linalg.norm(x_new[:, 0] - x_new[:, 1])

        # Assert the new distance is at least the safety radius
        self.assertEqual(dxu.shape, (2,2))
        self.assertGreaterEqual(distance_after, 0.12)
    
    def test_no_control_modification(self):
        barrier_fn = create_robust_barriers(base_length=0.1, wheel_radius=0.01, safety_radius=0.1, gamma=100)

        # Two robots moving directly toward each other
        x = jnp.array([
            [0.0, 2],  # x positions
            [0.0, 0.0],  # y positions
            [0.0, jnp.pi]  # orientations (facing each other)
        ])
        dxu = jnp.array([
            [0.1, -0.1],  # Velocities that would not lead to collision
            [0.0, 0.0]
        ])

        dxu_safe = barrier_fn(dxu, x, [])

        # Assert the control inputs are unchanged
        self.assertEqual(dxu.shape, (2,2))
        self.assertLessEqual(jnp.sum((dxu - dxu_safe)**2), 1e-5)