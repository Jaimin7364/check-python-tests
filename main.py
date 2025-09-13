# main.py
# A simple file with functions to demonstrate the CI/CD test runner.

def add_numbers(a, b):
    """
    Adds two numbers together.

    Args:
        a (int or float): The first number.
        b (int or float): The second number.

    Returns:
        int or float: The sum of the two numbers.
    """
    return a + b

def is_prime(n):
    """
    Checks if a number is a prime number.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def capitalize_string(s):
    """
    Capitalizes the first letter of each word in a string.
    
    Args:
        s (str): The input string.

    Returns:
        str: The string with each word capitalized.
    """
    return s.title()

class Calculator:
    """A simple calculator class."""
    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero.")
        return a / b

    def power(self, base, exp):
        """
        Calculates the power of a base to an exponent.

        Args:
            base (int or float): The base number.
            exp (int): The exponent.

        Returns:
            int or float: The result of base ** exp.
        """
        return base ** exp
