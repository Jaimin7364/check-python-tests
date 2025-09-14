import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, patch
from pathlib import Path
from utils import sum_three_numbers
from main import add_numbers, is_prime, multiply_numbers


class TestGeneratedCode(unittest.TestCase):
    def setUp(self):
        pass

    def test_add_numbers_normal_cases(self):
        self.assertEqual(add_numbers(1, 2), 3)
        self.assertEqual(add_numbers(1.5, 2.5), 4.0)
        self.assertEqual(add_numbers(-1, -2), -3)
        self.assertEqual(add_numbers(-1.5, -2.5), -4.0)

    def test_add_numbers_edge_cases(self):
        self.assertEqual(add_numbers(0, 0), 0)
        self.assertEqual(add_numbers(1, 0), 1)
        self.assertEqual(add_numbers(0, 1), 1)

    def test_add_numbers_error_cases(self):
        with self.assertRaises(TypeError):
            add_numbers('a', 2)
        with self.assertRaises(TypeError):
            add_numbers(1, 'b')

    def test_multiply_numbers_normal_cases(self):
        self.assertEqual(multiply_numbers(1, 2), 2)
        self.assertEqual(multiply_numbers(1.5, 2.5), 3.75)
        self.assertEqual(multiply_numbers(-1, -2), 2)
        self.assertEqual(multiply_numbers(-1.5, -2.5), 3.75)

    def test_multiply_numbers_edge_cases(self):
        self.assertEqual(multiply_numbers(0, 0), 0)
        self.assertEqual(multiply_numbers(1, 0), 0)
        self.assertEqual(multiply_numbers(0, 1), 0)

    def test_multiply_numbers_error_cases(self):
        with self.assertRaises(TypeError):
            multiply_numbers('a', 2)
        with self.assertRaises(TypeError):
            multiply_numbers(1, 'b')

    def test_is_prime_normal_cases(self):
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertTrue(is_prime(7))

    def test_is_prime_edge_cases(self):
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(6))
        self.assertFalse(is_prime(8))

    def test_is_prime_error_cases(self):
        with self.assertRaises(TypeError):
            is_prime('a')
        with self.assertRaises(TypeError):
            is_prime(1.5)

    def test_sum_three_numbers_normal_cases(self):
        self.assertEqual(sum_three_numbers(1, 2, 3), 6)
        self.assertEqual(sum_three_numbers(1.5, 2.5, 3.5), 7.5)
        self.assertEqual(sum_three_numbers(-1, -2, -3), -6)
        self.assertEqual(sum_three_numbers(-1.5, -2.5, -3.5), -7.5)

    def test_sum_three_numbers_edge_cases(self):
        self.assertEqual(sum_three_numbers(0, 0, 0), 0)
        self.assertEqual(sum_three_numbers(1, 0, 0), 1)
        self.assertEqual(sum_three_numbers(0, 1, 0), 1)
        self.assertEqual(sum_three_numbers(0, 0, 1), 1)

    def test_sum_three_numbers_error_cases(self):
        with self.assertRaises(TypeError):
            sum_three_numbers('a', 2, 3)
        with self.assertRaises(TypeError):
            sum_three_numbers(1, 'b', 3)
        with self.assertRaises(TypeError):
            sum_three_numbers(1, 2, 'c')

    def test_sum_three_numbers_type_validation(self):
        self.assertIsInstance(sum_three_numbers(1, 2, 3), (int, float))
        self.assertIsInstance(sum_three_numbers(1.5, 2.5, 3.5), (int, float))


if __name__ == '__main__':
    unittest.main()
