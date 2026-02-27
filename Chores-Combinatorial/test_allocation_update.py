import unittest
from unittest.mock import patch

from allocation_update import allocation_update


class TestAllocationUpdate(unittest.TestCase):
    def test_branch_calls_balance_allocation(self) -> None:
        x = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        p = [1.0, 1.0]
        d = [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
        e = [1.0, 1.0]
        S = {0}
        gamma_set = {0}

        expected = (
            [[0.7, 0.3], [0.3, 0.7]],
            [1.0, 1.0],
            {0, 1},
        )

        with patch("allocation_update.balance_allocation", return_value=expected) as mocked:
            out = allocation_update(x, p, S, d, e, gamma_set)

        mocked.assert_called_once_with(p, 2)
        self.assertEqual(out, expected)

    def test_else_branch_reassigns_j_set_to_agent_in_s(self) -> None:
        x = [
            [0.0, 0.0],
            [0.0, 4.0],
        ]
        p = [1.0, 1.0]
        d = [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
        e = [0.0, 4.0]
        S = {0}
        gamma_set = {0}

        new_x, new_e, new_s = allocation_update(x, p, S, d, e, gamma_set)

        # J should include chore 1, and it should be assigned fully to agent 0.
        self.assertAlmostEqual(new_x[0][1], 1.0, places=10)
        self.assertAlmostEqual(new_x[1][1], 0.0, places=10)
        # Chore 0 is untouched by the else-branch loop.
        self.assertAlmostEqual(new_x[0][0], 0.0, places=10)
        self.assertAlmostEqual(new_x[1][0], 0.0, places=10)

        self.assertAlmostEqual(new_e[0], 1.0, places=10)
        self.assertAlmostEqual(new_e[1], 0.0, places=10)
        self.assertEqual(new_s, {0})

    def test_all_agents_in_s_returns_unchanged(self) -> None:
        x = [
            [0.5, 0.5],
            [0.5, 0.5],
        ]
        p = [1.0, 2.0]
        d = [
            [1.0, 2.0],
            [2.0, 1.0],
        ]
        e = [1.5, 1.5]
        S = {0, 1}
        gamma_set = {0}

        new_x, new_e, new_s = allocation_update(x, p, S, d, e, gamma_set)

        self.assertEqual(new_x, x)
        self.assertAlmostEqual(new_e[0], 1.5, places=10)
        self.assertAlmostEqual(new_e[1], 1.5, places=10)
        self.assertEqual(new_s, S)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            allocation_update([[1.0]], [1.0], set(), [[1.0]], [1.0], set())
        with self.assertRaises(ValueError):
            allocation_update([[1.0]], [0.0], {0}, [[1.0]], [1.0], set())
        with self.assertRaises(ValueError):
            allocation_update([[1.0]], [1.0], {0}, [[1.0]], [1.0], {2})


if __name__ == "__main__":
    unittest.main()
