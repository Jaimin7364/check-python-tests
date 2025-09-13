import os
import subprocess
import json
import sys
import re
import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

from groq import Groq

# --- Configuration ---
# Use an environment variable for the Groq API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY environment variable not set.")
    sys.exit(1)

# Use the Llama 3.1 70B model
MODEL_ID = "llama-3.3-70b-versatile"
DEFAULT_TEST_DIR = "tests_pr"

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
    Analyzes code changes in a PR and generates/runs tests using a Groq LLM.
    """
    def __init__(self):
        self.project_root = Path(os.getcwd())
        # Add project root to the Python path
        sys.path.insert(0, str(self.project_root))
        
        self.test_dir = self.project_root / DEFAULT_TEST_DIR
        self.test_dir.mkdir(exist_ok=True)
        # Initialize the Groq client
        self.client = Groq(api_key=GROQ_API_KEY)

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
        """Invokes the Groq LLM for code generation."""
        try:
            # Groq API call
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt},
                ],
                model=MODEL_ID,
                temperature=0.1,
            )
            
            generated_code = chat_completion.choices[0].message.content.strip()

            # Clean up markdown code fences
            if "```python" in generated_code:
                code_start = generated_code.find("```python") + len("```python")
                code_end = generated_code.rfind("```")
                if code_end > code_start:
                    generated_code = generated_code[code_start:code_end].strip()
                else:
                    generated_code = ""
            return generated_code
        except Exception as e:
            print(f"❌ Error generating test code with Groq LLM: {e}")
            return ""

    def _generate_test_suite(self, functions: List[FunctionInfo]) -> str:
        """Generates a combined test file for multiple functions/methods."""
        if not functions:
            return ""

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
You are an expert Python unit testing engineer. Generate a comprehensive test file using Python's `unittest` framework.

FUNCTIONS/METHODS TO TEST:
{chr(10).join(functions_info)}

REQUIRED IMPORTS (use these exact imports):
{all_imports}

CRITICAL INSTRUCTIONS:
1. Use ONLY the imports provided above - do not make up class names or imports
2. Create a `unittest.TestCase` class for the tests.
3. For each function or method, create a test method (e.g., `test_function_name`).
4. For class methods, instantiate the class in setUp() or within test methods
5. Include test cases for normal behavior, edge cases, and error conditions.
6. Use `self.assertEqual`, `self.assertTrue`, `self.assertRaises`, etc.
7. Provide meaningful docstrings.
8. Use a `if __name__ == '__main__':` block to run the tests.
9. IMPORTANT: Only test the functions/methods that are explicitly provided in the function list above

Generate ONLY the complete, runnable Python code for the test file. No explanations.
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
        """Executes the generated test file with code coverage."""
        print(f"🧪 Executing tests from {test_file_path}")
        
        # Get the current Python executable instead of hardcoded 'python'
        python_executable = sys.executable
        
        # Set PYTHONPATH to include the project root for imports
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{str(self.project_root)}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(self.project_root)
        
        try:
            # Run tests with coverage
            coverage_file = self.project_root / ".coverage"
            
            # First, run the tests with coverage
            run_result = subprocess.run(
                [python_executable, "-m", "coverage", "run", "--source=.", str(test_file_path)],
                capture_output=True, text=True, check=True,
                cwd=str(self.project_root),
                env=env
            )
            
            # Generate coverage report
            coverage_result = subprocess.run(
                [python_executable, "-m", "coverage", "report", "--format=text"],
                capture_output=True, text=True, check=True,
                cwd=str(self.project_root),
                env=env
            )
            
            # Generate detailed coverage report
            coverage_html_result = subprocess.run(
                [python_executable, "-m", "coverage", "html", "-d", "htmlcov"],
                capture_output=True, text=True,
                cwd=str(self.project_root),
                env=env
            )
            
            print("✅ Tests passed.")
            print("\n📊 Code Coverage Report:")
            print(coverage_result.stdout)
            
            if coverage_html_result.returncode == 0:
                print("📄 Detailed HTML coverage report generated in 'htmlcov/' directory")
            
            return {
                "status": "success", 
                "output": run_result.stdout,
                "coverage_report": coverage_result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            print("❌ Tests failed.")
            print("--- Test Output ---")
            print(e.stdout)
            print(e.stderr)
            print("-------------------")
            return {"status": "failure", "output": e.stdout + e.stderr}

    def _parse_coverage_metrics(self, coverage_output: str) -> Dict[str, str]:
        """Parse coverage output to extract key metrics."""
        metrics = {}
        lines = coverage_output.strip().split('\n')
        
        for line in lines:
            if 'TOTAL' in line:
                # Extract total coverage percentage
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        coverage_percent = parts[-1].rstrip('%')
                        metrics['total_coverage'] = coverage_percent
                    except (ValueError, IndexError):
                        pass
                break
        
        return metrics

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

        # Execute tests and get coverage results
        test_results = self._execute_tests(test_file_path)
        
        # Display coverage summary
        if test_results["status"] == "success" and "coverage_report" in test_results:
            coverage_metrics = self._parse_coverage_metrics(test_results["coverage_report"])
            if "total_coverage" in coverage_metrics:
                coverage_percent = coverage_metrics["total_coverage"]
                print(f"\n🎯 Total Code Coverage: {coverage_percent}%")
                
                # Provide coverage feedback
                try:
                    coverage_float = float(coverage_percent)
                    if coverage_float >= 90:
                        print("🟢 Excellent coverage!")
                    elif coverage_float >= 75:
                        print("🟡 Good coverage, consider adding more tests")
                    elif coverage_float >= 50:
                        print("🟠 Moderate coverage, more tests recommended")
                    else:
                        print("🔴 Low coverage, significant testing gaps detected")
                except ValueError:
                    pass

if __name__ == "__main__":
    runner = ChangeAnalyzerAndTester()
    runner.run()