from __future__ import annotations

import random

from combinatorial_algorithm import solve_chores_ceei


def main() -> None:
    random.seed(42)
    for t in range(1, 11):
        n_agents = random.randint(2, 4)
        n_chores = random.randint(2, 4)
        d = [
            [round(random.uniform(0.5, 5.0), 3) for _ in range(n_chores)]
            for _ in range(n_agents)
        ]

        print(f"Instance {t}")
        print(f"n_agents={n_agents}, n_chores={n_chores}")
        print("d=")
        for row in d:
            print("  ", row)

        try:
            out = solve_chores_ceei(d)
            print("status=solved")
            print("p=", [round(v, 6) for v in out.p])
            print("x=")
            for row in out.x:
                print("  ", [round(v, 6) for v in row])
            print("e=", [round(v, 6) for v in out.e])
            print(f"S={sorted(out.s_set)}, iters={out.iterations}")
        except Exception as exc:  # noqa: BLE001
            print("status=failed")
            print(f"reason={type(exc).__name__}: {exc}")

        print("-" * 60)


if __name__ == "__main__":
    main()

