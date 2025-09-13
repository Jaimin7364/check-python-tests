import os
import subprocess
import json
import sys
import requests
import re
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

# --- Configuration ---
DEFAULT_TEST_DIR = "tests_pr"

# Free LLM Provider Configuration
LLM_PROVIDERS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    },
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "models": [
            "llama-3.3-70b-versatile",  # Primary model
            "llama-3.1-8b-instant",     # Fallback 1
            "gemma2-9b-it",            # Fallback 2
            "llama3-70b-8192"          # Fallback 3
        ],
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    },
    "together": {
        "url": "https://api.together.xyz/v1/chat/completions",
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    },
    "huggingface": {
        "url": "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
        "model": "microsoft/DialoGPT-medium",
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    }
}

@dataclass
class FunctionInfo:
    """Information about a function in the codebase"""
    name: str
    file_path: str
    code: str
    line_start: int
    line_end: int
    is_method: bool = False
    class_name: Optional[str] = None

class ChangeAnalyzerAndTester:
    """
    Analyzes code changes in a PR and generates/runs tests using free LLM APIs.
    """
    def __init__(self):
        self.project_root = Path(os.getcwd())
        # Add project root to the Python path
        sys.path.insert(0, str(self.project_root))
        
        self.test_dir = self.project_root / DEFAULT_TEST_DIR
        self.test_dir.mkdir(exist_ok=True)
        
        # Configure LLM provider
        self.llm_provider = os.environ.get("LLM_PROVIDER", "openrouter").lower()
        self.api_key = self._get_api_key()
        
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _get_api_key(self) -> str:
        """Get API key based on the selected provider."""
        provider_keys = {
            "openrouter": "OPENROUTER_API_KEY",
            "groq": "GROQ_API_KEY", 
            "together": "TOGETHER_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY"
        }
        
        key_name = provider_keys.get(self.llm_provider)
        if not key_name:
            raise ValueError(f"Unknown provider: {self.llm_provider}")
            
        api_key = os.environ.get(key_name)
        if not api_key:
            raise ValueError(f"❌ API key required! Set {key_name} environment variable for {self.llm_provider} provider.")
        
        return api_key

    def _get_changed_files(self) -> List[str]:
        """Gets changed files from the GITHUB_ENV variable."""
        changed_files_str = os.environ.get("CHANGED_FILES", "")
        if not changed_files_str.strip():
            print("ℹ️ No changed files detected in this PR.")
            return []
        return changed_files_str.split()

    def _extract_functions_from_file(self, file_path: Path) -> List[FunctionInfo]:
        """Extracts function and method definitions from Python code using AST."""
        functions = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=file_path.name)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    is_method = False
                    class_name = None
                    # A simple check for methods by inspecting the parent
                    parent = next((n for n in ast.walk(tree) if hasattr(n, 'body') and node in n.body), None)
                    if isinstance(parent, ast.ClassDef):
                        is_method = True
                        class_name = parent.name
                    
                    # Get the source code for the function
                    start_line = node.lineno
                    end_line = node.end_lineno
                    lines = content.splitlines()
                    func_code = "\n".join(lines[start_line-1:end_line])

                    functions.append(FunctionInfo(
                        name=node.name,
                        file_path=str(file_path.relative_to(self.project_root)),
                        code=func_code,
                        line_start=start_line,
                        line_end=end_line,
                        is_method=is_method,
                        class_name=class_name
                    ))
        except SyntaxError as e:
            print(f"⚠️ Syntax error in {file_path}: {e}")
        return functions

    def _invoke_llm_for_generation(self, prompt: str) -> str:
        """Invokes the free LLM API for code generation."""
        provider_config = LLM_PROVIDERS[self.llm_provider]
        
        # Get models (handle both single model and multiple models)
        if "models" in provider_config:
            models_to_try = provider_config["models"]
        else:
            models_to_try = [provider_config["model"]]
        
        headers = provider_config["headers"](self.api_key)
        
        # Try each model until one works
        last_error = None
        for model_name in models_to_try:
            print(f"🤖 Calling {self.llm_provider} API with model {model_name}...")
            
            # Prepare the request payload based on provider
            if self.llm_provider == "huggingface":
                payload = {"inputs": prompt}
            else:
                # OpenAI-compatible format for openrouter, groq, together
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are an expert Python unit testing engineer. Generate comprehensive, well-structured unit tests using Python's unittest framework."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.1,
                    "stream": False
                }
            
            try:
                response = requests.post(
                    provider_config["url"],
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code != 200:
                    print(f"❌ Model {model_name} returned status {response.status_code}")
                    if response.status_code == 400:
                        error_data = response.json()
                        if "model_decommissioned" in str(error_data) or "not supported" in str(error_data):
                            print(f"⚠️ Model {model_name} is decommissioned, trying next model...")
                            continue
                    print(f"Response: {response.text}")
                    response.raise_for_status()
                
                response_data = response.json()
                
                # Extract generated text based on provider response format
                if self.llm_provider == "huggingface":
                    generated_code = response_data[0].get("generated_text", "").strip()
                else:
                    # OpenAI-compatible format
                    generated_code = response_data["choices"][0]["message"]["content"].strip()
                
                # Clean up code blocks
                if "```python" in generated_code:
                    code_start = generated_code.find("```python") + len("```python")
                    code_end = generated_code.rfind("```")
                    if code_end > code_start:
                        generated_code = generated_code[code_start:code_end].strip()
                elif "```" in generated_code:
                    # Handle cases where just ``` is used without python
                    code_start = generated_code.find("```") + 3
                    code_end = generated_code.rfind("```")
                    if code_end > code_start:
                        generated_code = generated_code[code_start:code_end].strip()
                
                if not generated_code or len(generated_code.strip()) < 50:
                    raise ValueError("Generated code is too short or empty")
                
                print(f"✅ Successfully generated {len(generated_code)} characters of test code using {model_name}")
                return generated_code
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error with model {model_name}: {e}")
                last_error = e
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        print(f"API Error Details: {error_detail}")
                    except:
                        print(f"Response text: {e.response.text}")
                
                # If this is the last model, raise the error
                if model_name == models_to_try[-1]:
                    break
                else:
                    print(f"⚠️ Trying next model...")
                    continue
                    
            except Exception as e:
                print(f"❌ Error with model {model_name}: {e}")
                last_error = e
                # If this is the last model, raise the error
                if model_name == models_to_try[-1]:
                    break
                else:
                    print(f"⚠️ Trying next model...")
                    continue
        
        # If we get here, all models failed
        raise Exception(f"Failed to generate tests using {self.llm_provider} API. Tried models: {', '.join(models_to_try)}") from last_error

    def _generate_mock_test_code(self, prompt: str) -> str:
        """Generate a basic test template when LLM is not available."""
        # Try to extract function names from the prompt to make better mock tests
        function_names = []
        if "Function/Method:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if line.startswith("Function/Method:"):
                    func_name = line.split(":")[1].strip()
                    function_names.append(func_name)
        
        if not function_names:
            # Fallback generic test
            return '''import unittest

class TestGeneratedCode(unittest.TestCase):
    
    def test_placeholder(self):
        """Placeholder test - replace with actual tests."""
        self.assertTrue(True, "Placeholder test")

if __name__ == '__main__':
    unittest.main()
'''
        
        # Generate specific tests for detected functions
        test_code = "import unittest\n\n"
        
        # Extract imports from prompt
        if "from " in prompt and " import " in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if line.strip().startswith("from ") and " import " in line:
                    test_code += line.strip() + "\n"
            test_code += "\n"
        
        test_code += "class TestGeneratedCode(unittest.TestCase):\n\n"
        
        for func_name in function_names[:3]:  # Limit to first 3 functions
            test_code += f'''    def test_{func_name}(self):
        """Test for {func_name} function."""
        # TODO: Add actual test implementation
        self.assertTrue(True, "Placeholder test for {func_name}")

'''
        
    def _generate_test_suite(self, functions: List[FunctionInfo]) -> str:
        """Generates a combined test file for multiple functions/methods."""
        if not functions:
            raise ValueError("No functions provided for test generation")

        functions_info = []
        imports_by_file = {}
        
        for func in functions:
            functions_info.append(f"Function/Method: {func.name}\nFile: {func.file_path}\nCode: {func.code}")
            
            # Get module name from file path (remove .py extension)
            module_name = Path(func.file_path).stem
            
            if module_name not in imports_by_file:
                imports_by_file[module_name] = {'functions': set(), 'classes': set()}
            
            # Collect imports properly by file
            if func.class_name:
                imports_by_file[module_name]['classes'].add(func.class_name)
            else:
                imports_by_file[module_name]['functions'].add(func.name)
        
        # Generate imports grouped by file
        imports_set = set()
        for module_name, items in imports_by_file.items():
            all_items = list(items['functions']) + list(items['classes'])
            if all_items:
                imports_set.add(f"from {module_name} import {', '.join(sorted(all_items))}")
        
        all_imports = "\n".join(imports_set)
        
        prompt = f"""
You are an expert Python unit testing engineer. Generate comprehensive, production-quality unit tests using Python's `unittest` framework.

FUNCTIONS/METHODS TO TEST:
{chr(10).join(functions_info)}

REQUIRED IMPORTS (use these exact imports):
{all_imports}

CRITICAL INSTRUCTIONS:
1. Use ONLY the imports provided above - do not make up class names or imports
2. Create comprehensive `unittest.TestCase` classes for the tests
3. For each function/method, create multiple test methods covering:
   - Normal/happy path scenarios with various valid inputs
   - Edge cases (empty inputs, boundary values, special characters)
   - Error conditions with appropriate `assertRaises` tests
   - Different data types if applicable
4. For class methods, instantiate the class properly in setUp() or within test methods
5. Use descriptive test method names like `test_function_name_with_valid_input`
6. Include meaningful docstrings for each test method
7. Use appropriate assertions: `assertEqual`, `assertTrue`, `assertFalse`, `assertRaises`, `assertIn`, etc.
8. Test both expected outputs and side effects
9. For functions with complex logic, test multiple execution paths
10. Add setUp() and tearDown() methods if needed for test fixtures

EXAMPLE TEST STRUCTURE:
```python
import unittest
from module import function_name, ClassName

class TestFunctionName(unittest.TestCase):
    
    def test_function_name_with_valid_input(self):
        \"\"\"Test function with normal valid input.\"\"\"
        result = function_name("valid_input")
        self.assertEqual(result, expected_output)
    
    def test_function_name_with_empty_input(self):
        \"\"\"Test function behavior with empty input.\"\"\"
        # Test implementation
    
    def test_function_name_raises_error_for_invalid_input(self):
        \"\"\"Test that function raises appropriate error for invalid input.\"\"\"
        with self.assertRaises(ExpectedError):
            function_name(invalid_input)

if __name__ == '__main__':
    unittest.main()
```

Generate ONLY the complete, runnable Python code for the test file. No explanations or markdown formatting.
"""
        return self._invoke_llm_for_generation(prompt)

    def _format_python_code(self, file_path: Path):
        """Formats a Python file using autopep8."""
        try:
            subprocess.run(
                ['autopep8', '--in-place', str(file_path)],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✔️ Formatted: {file_path.name}")
        except FileNotFoundError:
            print("⚠️ autopep8 not found. Skipping code formatting.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to format {file_path.name}: {e.stderr}")

    def _execute_tests(self, test_file_path: Path) -> Dict[str, any]:
        """Executes the generated test file."""
        print(f"🧪 Executing tests from {test_file_path}")
        try:
            # Get the current Python executable instead of hardcoded 'python'
            python_executable = sys.executable
            # Set PYTHONPATH to include the project root for imports
            env = os.environ.copy()
            current_pythonpath = env.get('PYTHONPATH', '')
            if current_pythonpath:
                env['PYTHONPATH'] = f"{str(self.project_root)}:{current_pythonpath}"
            else:
                env['PYTHONPATH'] = str(self.project_root)
            
            run_result = subprocess.run(
                [python_executable, str(test_file_path)],
                capture_output=True, text=True, check=True,
                cwd=str(self.project_root),
                env=env
            )
            print("✅ Tests passed.")
            return {"status": "success", "output": run_result.stdout}
        except subprocess.CalledProcessError as e:
            print("❌ Tests failed.")
            print("--- Test Output ---")
            print(e.stdout)
            print(e.stderr)
            print("-------------------")
            return {"status": "failure", "output": e.stdout + e.stderr}

    def run(self):
        """Main runner for the CI script."""
        changed_files = self._get_changed_files()
        if not changed_files:
            print("No Python files changed. Exiting CI run.")
            return

        all_changed_functions = []
        for file_path_str in changed_files:
            file_path = self.project_root / file_path_str
            if file_path.exists() and file_path.suffix == '.py':
                print(f"🔍 Analyzing changed file: {file_path_str}")
                functions = self._extract_functions_from_file(file_path)
                all_changed_functions.extend(functions)

        if not all_changed_functions:
            print("No functions found in changed Python files. Exiting.")
            return

        print(f"🤖 Found {len(all_changed_functions)} functions/methods to test.")
        
        test_content = self._generate_test_suite(all_changed_functions)
        if not test_content:
            print("❌ Failed to generate test cases. Aborting.")
            return

        test_file_name = "pr_generated_tests.py"
        test_file_path = self.test_dir / test_file_name
        try:
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            self._format_python_code(test_file_path)
            print(f"✅ Generated and saved tests to {test_file_path}")
        except Exception as e:
            print(f"❌ Failed to save test file: {e}")
            return

        self._execute_tests(test_file_path)

if __name__ == "__main__":
    runner = ChangeAnalyzerAndTester()
    runner.run()
