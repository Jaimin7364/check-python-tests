# utils.py
# A new file with a simple utility function to demonstrate the CI/CD test runner.

def reverse_string(s):
    """
    Reverses a given string.

    Args:
        s (str): The string to be reversed.

    Returns:
        str: The reversed string.
    
    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError(f"Expected string, got {type(s).__name__}")
    return s[::-1]

# Example usage (for demonstration and local testing)
if __name__ == "__main__":
    original_str = "Hello, World!"
    reversed_str = reverse_string(original_str)
    print(f"Original: {original_str}")
    print(f"Reversed: {reversed_str}")
