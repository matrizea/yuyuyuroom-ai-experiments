import unittest
import numpy as np
from experiment_depth import near_display,MODELS,REVISIONS

class DisplayTests(unittest.TestCase):
    def test_order(self):
        a=np.arange(100,dtype=np.float32).reshape(10,10)
        self.assertGreater(near_display(a,'v1')[-1,-1],near_display(a,'v1')[0,0])
        self.assertLess(near_display(a,'v3')[-1,-1],near_display(a,'v3')[0,0])
    def test_constant(self):
        a=near_display(np.ones((5,5)),'v3')
        self.assertTrue(np.isfinite(a).all());self.assertTrue((a==0).all())
    def test_limits(self):
        a=near_display(np.arange(100,dtype=np.float32),'v2')
        self.assertGreaterEqual(float(a.min()),0);self.assertLessEqual(float(a.max()),1)
    def test_revisions(self):
        self.assertEqual(set(MODELS),set(REVISIONS))
        self.assertTrue(all(len(x)==40 for x in REVISIONS.values()))

if __name__=='__main__':unittest.main()
