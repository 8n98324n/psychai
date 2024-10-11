# tests/test_hello_world_hello_world.py

import unittest
from psychai.hello_world.hello_world import hello_world

class TestHello_World(unittest.TestCase):
    def test_add(self):
        self.assertEqual(hello_world(2, 3), 5)

if __name__ == "__main__":
    unittest.main()