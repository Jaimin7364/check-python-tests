import unittest
from main import add_numbers, capitalize_string, is_prime
from main import Calculator

class TestMainFunctions(unittest.TestCase):
    
    def test_add_numbers(self):
        self.assertEqual(add_numbers(2, 3), 5)
        self.assertEqual(add_numbers(-1, 1), 0)
        self.assertEqual(add_numbers(0, 0), 0)
        self.assertEqual(add_numbers(2.5, 3.5), 6.0)
    
    def test_is_prime(self):
        self.assertTrue(is_prime(2))
        self.assertTrue(is_prime(3))
        self.assertFalse(is_prime(1))
        self.assertFalse(is_prime(0))
        self.assertFalse(is_prime(-1))
        self.assertFalse(is_prime(4))
        self.assertTrue(is_prime(29))
    
    def test_capitalize_string(self):
        self.assertEqual(capitalize_string("hello world"), "Hello World")
        self.assertEqual(capitalize_string("PYTHON"), "Python")
        self.assertEqual(capitalize_string(""), "")
        self.assertEqual(capitalize_string("a"), "A")
    
    def setUp(self):
        self.calc = Calculator()
    
    def test_multiply(self):
        self.assertEqual(self.calc.multiply(2, 3), 6)
        self.assertEqual(self.calc.multiply(-1, 1), -1)
        self.assertEqual(self.calc.multiply(0, 0), 0)
        self.assertEqual(self.calc.multiply(2.5, 3.5), 8.75)
    
    def test_divide(self):
        self.assertEqual(self.calc.divide(6, 3), 2)
        self.assertEqual(self.calc.divide(-1, 1), -1)
        self.assertEqual(self.calc.divide(0, 1), 0)
        self.assertEqual(self.calc.divide(5, 2), 2.5)
        with self.assertRaises(ValueError):
            self.calc.divide(5, 0)
    
    def test_power(self):
        self.assertEqual(self.calc.power(2, 3), 8)
        self.assertEqual(self.calc.power(0, 0), 1)
        self.assertEqual(self.calc.power(2, 0), 1)
        self.assertEqual(self.calc.power(2.5, 2), 6.25)
    
    def test_subtract(self):
        self.assertEqual(self.calc.subtract(5, 3), 2)
        self.assertEqual(self.calc.subtract(-1, 1), -2)
        self.assertEqual(self.calc.subtract(0, 0), 0)
        self.assertEqual(self.calc.subtract(2.5, 1.5), 1.0)

if __name__ == '__main__':
    unittest.main()