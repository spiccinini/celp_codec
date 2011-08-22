import unittest

from adaptive_codebook import AdaptiveCodebook

class AdaptiveCBTest(unittest.TestCase):

    def test_empty(self):
        cb_size = 20
        vector_size = 10
        ac = AdaptiveCodebook(vector_size=vector_size, cb_size=cb_size, min_period=5)
        self.assertEqual(len(ac), cb_size)
        for i in range(cb_size):
            self.assertEqual(len(ac[i]), vector_size)

    def test_values(self):
        cb_size = 20
        vector_size = 10
        ac = AdaptiveCodebook(vector_size=vector_size, cb_size=cb_size, min_period=5)

        ac.samples = range(-ac.max_period, 0)

        self.assertEqual(ac[0], [-5, -4, -3, -2, -1, -5, -4, -3, -2, -1])
        self.assertEqual(ac[1], [-6, -5, -4, -3, -2, -1, -6, -5, -4, -3])
        self.assertEqual(ac[5], [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1])
        self.assertEqual(ac[19], [-24, -23, -22, -21, -20, -19, -18, -17, -16, -15])

    def test_add_vector(self):
        cb_size = 30
        vector_size = 10
        ac = AdaptiveCodebook(vector_size=vector_size, cb_size=cb_size, min_period=5)

        vector = [1]*vector_size
        ac.add_vector(vector)
        self.assertEqual(ac[0], vector[-ac.min_period:]+vector[-ac.min_period:])




