import unittest
from textutil import title_case, slugify


class T(unittest.TestCase):
    def test_title(self):
        self.assertEqual(title_case('a b'), 'A B')

    def test_slug(self):
        self.assertEqual(slugify('Hello, World!'), 'hello-world')
