import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
from utils import sum_three_numbers


class TestGeneratedCode(unittest.TestCase):
    def setUp(self):
        pass

    def test_sum_three_numbers_normal_case(self):
        self.assertEqual(sum_three_numbers(1, 2, 3), 6)
        self.assertEqual(sum_three_numbers(1.5, 2.5, 3.5), 7.5)

    def test_sum_three_numbers_edge_cases(self):
        self.assertEqual(sum_three_numbers(0, 0, 0), 0)
        self.assertEqual(sum_three_numbers(-1, -2, -3), -6)
        self.assertEqual(sum_three_numbers(-1.5, -2.5, -3.5), -7.5)

    def test_sum_three_numbers_type_validation(self):
        self.assertIsInstance(sum_three_numbers(1, 2, 3), int)
        self.assertIsInstance(sum_three_numbers(1.5, 2.5, 3.5), float)

    def test_sum_three_numbers_error_cases(self):
        with self.assertRaises(TypeError):
            sum_three_numbers('a', 2, 3)
        with self.assertRaises(TypeError):
            sum_three_numbers(1, 'b', 3)
        with self.assertRaises(TypeError):
            sum_three_numbers(1, 2, 'c')

    def test_sum_three_numbers_none_values(self):
        with self.assertRaises(TypeError):
            sum_three_numbers(None, 2, 3)
        with self.assertRaises(TypeError):
            sum_three_numbers(1, None, 3)
        with self.assertRaises(TypeError):
            sum_three_numbers(1, 2, None)


if __name__ == '__main__':
    unittest.main()
