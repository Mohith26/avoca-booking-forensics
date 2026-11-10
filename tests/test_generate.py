"""Run: python3 -m unittest discover -s tests   (stdlib only)"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
import generate as g  # noqa: E402


def calls(n=300, seed=1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        cust = g.CUSTOMERS[i % 3]
        out.append((g.make_call(rng, cust, i, 1), cust))
    return out


class GeneratorTests(unittest.TestCase):
    def test_deterministic(self):
        a = [g.simulate(c, cu, "baseline")[0] for c, cu in calls(seed=3)]
        b = [g.simulate(c, cu, "baseline")[0] for c, cu in calls(seed=3)]
        self.assertEqual(a, b)

    def test_same_acts_same_outcome(self):
        # common random numbers: if a policy takes the same acts, the caller
        # must behave identically (this is what makes replays comparable)
        for c, cu in calls():
            b0, e0, _, _ = g.simulate(c, cu, "baseline")
            for cand in g.CANDIDATES:
                b1, e1, _, _ = g.simulate(c, cu, cand["id"])
                if [a for _, a, _ in e0] == [a for _, a, _ in e1]:
                    self.assertEqual(b0, b1)

    def test_exante_probability_in_range_and_restores_state(self):
        for c, cu in calls(50):
            before = dict(c["u"])
            p = g.exante(c, cu, "baseline")
            self.assertTrue(0.0 < p <= 1.0)
            self.assertEqual(before, c["u"])

    def test_transcript_marks_fatal_turn_on_agent(self):
        for c, cu in calls():
            booked, ev, _, mode = g.simulate(c, cu, "baseline")
            turns, idx = g.render(c, cu, ev, booked, mode, random.Random(c["render_seed"]))
            if booked:
                self.assertIsNone(idx)
            elif mode != "CALLER_SHOPPING":
                self.assertEqual(turns[idx][0], "agent")


if __name__ == "__main__":
    unittest.main()
