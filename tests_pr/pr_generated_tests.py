import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from main import add_numbers, multiply_numbers


class TestGeneratedCode(unittest.TestCase):
    def setUp(self):
        pass

    def test_add_numbers_normal_cases(self):
        self.assertEqual(add_numbers(1, 2), 3)
        self.assertEqual(add_numbers(-1, 1), 0)
        self.assertEqual(add_numbers(-1, -1), -2)

    def test_add_numbers_edge_cases(self):
        self.assertEqual(add_numbers(0, 0), 0)
        self.assertEqual(add_numbers(0, 1), 1)
        self.assertEqual(add_numbers(1, 0), 1)

    def test_add_numbers_error_cases(self):
        with self.assertRaises(TypeError):
            add_numbers('a', 1)
        with self.assertRaises(TypeError):
            add_numbers(1, 'a')

    def test_multiply_numbers_normal_cases(self):
        self.assertEqual(multiply_numbers(1, 2), 2)
        self.assertEqual(multiply_numbers(-1, 1), -1)
        self.assertEqual(multiply_numbers(-1, -1), 1)

    def test_multiply_numbers_edge_cases(self):
        self.assertEqual(multiply_numbers(0, 0), 0)
        self.assertEqual(multiply_numbers(0, 1), 0)
        self.assertEqual(multiply_numbers(1, 0), 0)

    def test_multiply_numbers_error_cases(self):
        with self.assertRaises(TypeError):
            multiply_numbers('a', 1)
        with self.assertRaises(TypeError):
            multiply_numbers(1, 'a')

    def test_add_numbers_type_validation(self):
        self.assertIsInstance(add_numbers(1, 2), int)
        self.assertIsInstance(add_numbers(1.0, 2.0), float)

    def test_multiply_numbers_type_validation(self):
        self.assertIsInstance(multiply_numbers(1, 2), int)
        self.assertIsInstance(multiply_numbers(1.0, 2.0), float)


if __name__ == '__main__':
    unittest.main()
