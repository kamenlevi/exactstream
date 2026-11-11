import unittest
from app.users import get_user


class T(unittest.TestCase):
    def test_get(self):
        self.assertEqual(get_user(1), 'ann')
