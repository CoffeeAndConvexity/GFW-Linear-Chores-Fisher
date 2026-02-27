import unittest

from price_update import price_update


class TestPriceUpdate(unittest.TestCase):
    def test_price_update_basic_case(self) -> None:
        # Agent 0 has MPB at chore 0 only, so Gamma(S) = {0}.
        e = [10.0, 20.0]
        p = [1.0, 1.0, 1.0]
        d = [
            [1.0, 2.0, 4.0],
            [2.0, 1.0, 5.0],
        ]
        S = {0}

        new_p, new_e, gamma, gamma_set = price_update(e, p, S, d)

        self.assertEqual(gamma_set, {0})
        self.assertAlmostEqual(gamma, 0.25, places=10)
        self.assertAlmostEqual(new_p[0], 0.25, places=10)
        self.assertAlmostEqual(new_p[1], 1.0, places=10)
        self.assertAlmostEqual(new_p[2], 1.0, places=10)
        self.assertAlmostEqual(new_e[0], 2.5, places=10)
        self.assertAlmostEqual(new_e[1], 20.0, places=10)

    def test_gamma_undefined_when_all_chores_in_gamma_set(self) -> None:
        # All ratios are identical, so Gamma(S) contains all chores.
        e = [1.0, 2.0]
        p = [1.0, 1.0]
        d = [
            [1.0, 1.0],
            [2.0, 3.0],
        ]

        with self.assertRaises(ValueError):
            price_update(e, p, {0}, d)

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            price_update([1.0], [1.0], set(), [[1.0]])
        with self.assertRaises(ValueError):
            price_update([1.0], [1.0], {1}, [[1.0]])
        with self.assertRaises(ValueError):
            price_update([1.0], [-1.0], {0}, [[1.0]])


if __name__ == "__main__":
    unittest.main()
