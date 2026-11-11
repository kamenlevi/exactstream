import unittest
from app.users import fetch_user


class T(unittest.TestCase):
    def test_get(self):
        self.assertEqual(fetch_user(1), 'ann')
