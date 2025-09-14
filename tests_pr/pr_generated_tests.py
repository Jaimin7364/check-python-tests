import unittest
from utils import subtract_three_numbers
from main import Calculator, add_numbers, capitalize_string, is_prime


class TestFunctions(unittest.TestCase):
    """
    A test class for the functions in main.py and utils.py.
    """

    def setUp(self):
        """
        Instantiate the Calculator class for testing its methods.
        """
        self.calc = Calculator()

    def test_add_numbers(self):
        """
        Test the add_numbers function with normal behavior.
        """
        self.assertEqual(add_numbers(1, 2), 3)
        self.assertEqual(add_numbers(-1, 1), 0)
        self.assertEqual(add_numbers(1.5, 2.5), 4)

    def test_add_numbers_error(self):
        """
        Test the add_numbers function with error conditions.
        """
        with self.assertRaises(TypeError):
            add_numbers('a', 2)

    def test_is_prime(self):
        """
        Test the is_prime function with normal behavior.
        """
        self.assertTrue(is_prime(7))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(1))

    def test_is_prime_error(self):
        """
        Test the is_prime function with error conditions.
        """
        with self.assertRaises(TypeError):
            is_prime('a')

    def test_capitalize_string(self):
        """
        Test the capitalize_string function with normal behavior.
        """
        self.assertEqual(capitalize_string('hello world'), 'Hello World')
        self.assertEqual(capitalize_string('HELLO WORLD'), 'Hello World')
        self.assertEqual(capitalize_string(''), '')

    def test_multiply(self):
        """
        Test the multiply method of the Calculator class.
        """
        self.assertEqual(self.calc.multiply(2, 3), 6)
        self.assertEqual(self.calc.multiply(-2, 3), -6)
        self.assertEqual(self.calc.multiply(2.5, 3), 7.5)

    def test_multiply_error(self):
        """
        Test the multiply method of the Calculator class with error conditions.
        """
        with self.assertRaises(TypeError):
            self.calc.multiply('a', 2)

    def test_power(self):
        """
        Test the power method of the Calculator class.
        """
        self.assertEqual(self.calc.power(2, 3), 8)
        self.assertEqual(self.calc.power(-2, 3), -8)
        self.assertEqual(self.calc.power(2.5, 3), 15.625)

    def test_power_error(self):
        """
        Test the power method of the Calculator class with error conditions.
        """
        with self.assertRaises(TypeError):
            self.calc.power('a', 2)

    def test_subtract_three_numbers(self):
        """
        Test the subtract_three_numbers function with normal behavior.
        """
        self.assertEqual(subtract_three_numbers(10, 3, 2), 5)
        self.assertEqual(subtract_three_numbers(-10, 3, 2), -15)
        self.assertEqual(subtract_three_numbers(10.5, 3.5, 2), 5)

    def test_subtract_three_numbers_error(self):
        """
        Test the subtract_three_numbers function with error conditions.
        """
        with self.assertRaises(TypeError):
            subtract_three_numbers('a', 2, 3)


if __name__ == '__main__':
    unittest.main()
