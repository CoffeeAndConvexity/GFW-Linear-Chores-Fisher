import math
import unittest

from balance_allocation import balance_allocation


class TestBalanceAllocation(unittest.TestCase):
    def test_feasibility_and_optimality_positive_prices(self) -> None:
        p = [2.0, 3.0, 1.0, 4.0]
        n_agents = 3
        x, e, s_set = balance_allocation(p, n_agents)

        n_chores = len(p)
        self.assertEqual(len(x), n_agents)
        self.assertEqual(len(e), n_agents)
        self.assertTrue(all(len(row) == n_chores for row in x))

        # Feasibility: x_ij >= 0 and sum_i x_ij = 1.
        for row in x:
            for value in row:
                self.assertGreaterEqual(value, -1e-8)
        for j in range(n_chores):
            col_sum = sum(x[i][j] for i in range(n_agents))
            self.assertAlmostEqual(col_sum, 1.0, places=5)

        # Surplus consistency: e_i = sum_j x_ij * p_j.
        for i in range(n_agents):
            computed = sum(x[i][j] * p[j] for j in range(n_chores))
            self.assertAlmostEqual(e[i], computed, places=6)

        # Optimality check:
        # For fixed sum(e_i), max product occurs when e_i are equal.
        target = sum(p) / n_agents
        for value in e:
            self.assertAlmostEqual(value, target, places=4)

        product_e = math.prod(e)
        theoretical_max = target ** n_agents
        self.assertAlmostEqual(product_e, theoretical_max, places=4)

        # S must match the minimum-surplus agents.
        min_surplus = min(e)
        expected_s = {i for i, value in enumerate(e) if abs(value - min_surplus) <= 1e-8}
        self.assertEqual(s_set, expected_s)

    def test_zero_price_chores(self) -> None:
        p = [0.0, 0.0, 5.0]
        n_agents = 2
        x, e, s_set = balance_allocation(p, n_agents)

        for j in range(len(p)):
            col_sum = sum(x[i][j] for i in range(n_agents))
            self.assertAlmostEqual(col_sum, 1.0, places=5)

        target = sum(p) / n_agents
        for value in e:
            self.assertAlmostEqual(value, target, places=4)
        min_surplus = min(e)
        expected_s = {i for i, value in enumerate(e) if abs(value - min_surplus) <= 1e-8}
        self.assertEqual(s_set, expected_s)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            balance_allocation([1.0, 2.0], 0)
        with self.assertRaises(ValueError):
            balance_allocation([], 2)
        with self.assertRaises(ValueError):
            balance_allocation([1.0, -1.0], 2)


if __name__ == "__main__":
    unittest.main()
