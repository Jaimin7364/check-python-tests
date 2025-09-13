from main import Calculator, add_numbers, capitalize_string, is_prime
import unittest


class TestFunctions(unittest.TestCase):
    """
    A test class for the functions in main.py.
    """

    def setUp(self):
        """
        Instantiate the Calculator class for testing class methods.
        """
        self.calc = Calculator()

    def test_add_numbers(self):
        """
        Test the add_numbers function with normal behavior.
        """
        self.assertEqual(add_numbers(1, 2), 3)
        self.assertEqual(add_numbers(-1, 1), 0)
        self.assertEqual(add_numbers(1.5, 2.5), 4.0)

    def test_is_prime(self):
        """
        Test the is_prime function with normal behavior and edge cases.
        """
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertTrue(is_prime(5))
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(4))
        self.assertFalse(is_prime(6))

    def test_capitalize_string(self):
        """
        Test the capitalize_string function with normal behavior.
        """
        self.assertEqual(capitalize_string("hello world"), "Hello World")
        self.assertEqual(capitalize_string("HELLO WORLD"), "Hello World")
        self.assertEqual(capitalize_string("hElLo WoRlD"), "Hello World")

    def test_multiply(self):
        """
        Test the multiply method with normal behavior.
        """
        self.assertEqual(self.calc.multiply(2, 3), 6)
        self.assertEqual(self.calc.multiply(-2, 3), -6)
        self.assertEqual(self.calc.multiply(2, -3), -6)
        self.assertEqual(self.calc.multiply(2.5, 3), 7.5)

    def test_divide(self):
        """
        Test the divide method with normal behavior and error conditions.
        """
        self.assertEqual(self.calc.divide(6, 3), 2)
        self.assertEqual(self.calc.divide(-6, 3), -2)
        self.assertEqual(self.calc.divide(6, -3), -2)
        self.assertEqual(self.calc.divide(6.0, 3), 2.0)
        with self.assertRaises(ValueError):
            self.calc.divide(6, 0)

    def test_power(self):
        """
        Test the power method with normal behavior.
        """
        self.assertEqual(self.calc.power(2, 3), 8)
        self.assertEqual(self.calc.power(-2, 3), -8)
        self.assertEqual(self.calc.power(2, -3), 1/8)
        self.assertEqual(self.calc.power(2.5, 3), 15.625)


if __name__ == '__main__':
    unittest.main()
