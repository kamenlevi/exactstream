import unittest
from textutil import title_case


class T(unittest.TestCase):
    def test_title(self):
        self.assertEqual(title_case('a b'), 'A B')
