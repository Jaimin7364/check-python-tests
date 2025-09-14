import os
import subprocess
import json
import sys
import re
import ast
import hashlib
import xml.etree.ElementTree as ET
import tempfile
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

from groq import Groq

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY environment variable not set.")
    sys.exit(1)

MODEL_ID = "llama-3.3-70b-versatile"
DEFAULT_TEST_DIR = "tests_pr"

@dataclass
class TypeInfo:
    """Type information for function parameters and return types"""
    param_types: Dict[str, str] = field(default_factory=dict)
    return_type: Optional[str] = None
    has_self: bool = False
    has_cls: bool = False

@dataclass
class ImportInfo:
    """Information about imports needed for a function"""
    module_imports: Set[str] = field(default_factory=set)
    from_imports: Dict[str, Set[str]] = field(default_factory=dict)
    local_classes: Set[str] = field(default_factory=set)
    third_party_imports: Set[str] = field(default_factory=set)

@dataclass
class FunctionInfo:
    """Enhanced information about a function in the codebase"""
    name: str
    file_path: str
    code: str
    line_start: int
    line_end: int
    is_method: bool = False
    class_name: Optional[str] = None
    docstring: Optional[str] = None
    type_info: TypeInfo = field(default_factory=TypeInfo)
    import_info: ImportInfo = field(default_factory=ImportInfo)
    dependencies: Set[str] = field(default_factory=set)
    complexity_score: int = 1

@dataclass
class TestCaseResult:
    """Individual test case result"""
    name: str
    status: str  # "PASS", "FAIL", "ERROR", "SKIP"
    execution_time: float = 0.0
    error_message: str = ""
    failure_reason: str = ""
    test_method: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "name": self.name,
            "status": self.status,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "failure_reason": self.failure_reason,
            "test_method": self.test_method
        }

@dataclass
class TestReport:
    """Data structure for test execution report"""
    timestamp: str
    model_name: str
    changed_files: List[str]
    analyzed_functions: List[FunctionInfo]
    test_results: Dict[str, Any]
    coverage_metrics: Dict[str, str]
    status: str
    execution_time: float
    logs: List[str] = field(default_factory=list)
    syntax_validation: Dict[str, bool] = field(default_factory=dict)
    test_cases: List[TestCaseResult] = field(default_factory=list)

class CodeAnalyzer:
    """Enhanced code analyzer for better function understanding"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.builtin_types = {'int', 'str', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'bytes', 'None'}
    
    def extract_type_info(self, node: ast.FunctionDef) -> TypeInfo:
        """Extract comprehensive type information from function definition"""
        type_info = TypeInfo()
        
        # Extract parameter types
        for arg in node.args.args:
            param_name = arg.arg
            if param_name == 'self':
                type_info.has_self = True
            elif param_name == 'cls':
                type_info.has_cls = True
            
            if arg.annotation:
                type_info.param_types[param_name] = ast.unparse(arg.annotation)
        
        # Extract return type
        if node.returns:
            type_info.return_type = ast.unparse(node.returns)
        
        return type_info
    
    def extract_imports_from_file(self, file_path: Path) -> ImportInfo:
        """Extract all imports from a Python file"""
        import_info = ImportInfo()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=file_path.name)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_info.module_imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if module not in import_info.from_imports:
                        import_info.from_imports[module] = set()
                    for alias in node.names:
                        import_info.from_imports[module].add(alias.name)
                elif isinstance(node, ast.ClassDef):
                    import_info.local_classes.add(node.name)
        
        except (SyntaxError, FileNotFoundError):
            pass
        
        return import_info
    
    def calculate_complexity_score(self, node: ast.FunctionDef) -> int:
        """Calculate a simple complexity score for the function"""
        score = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                score += 1
            elif isinstance(child, ast.ExceptHandler):
                score += 1
            elif isinstance(child, (ast.And, ast.Or)):
                score += 1
        
        return min(score, 10)  # Cap at 10

class TestValidator:
    """Validates generated test code for syntax and basic correctness"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def validate_syntax(self, test_code: str) -> Tuple[bool, List[str]]:
        """Validate Python syntax of generated test code"""
        errors = []
        
        try:
            # First, try to parse the AST
            ast.parse(test_code)
            
            # Try to compile the code
            compile(test_code, '<test_code>', 'exec')
            
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax Error: {e.msg} at line {e.lineno}")
        except Exception as e:
            errors.append(f"Compilation Error: {str(e)}")
        
        return False, errors
    
    def validate_imports(self, test_code: str, available_modules: Set[str]) -> Tuple[bool, List[str]]:
        """Validate that all imports in test code are available"""
        errors = []
        
        try:
            tree = ast.parse(test_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split('.')[0]
                        if module_name not in available_modules and module_name not in {'unittest', 'unittest.mock', 'mock'}:
                            errors.append(f"Unavailable import: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split('.')[0]
                        if module_name not in available_modules and module_name not in {'unittest', 'unittest.mock', 'mock'}:
                            errors.append(f"Unavailable from-import: {node.module}")
        
        except SyntaxError:
            errors.append("Cannot parse test code for import validation")
        
        return len(errors) == 0, errors

class ChangeAnalyzerAndTester:
    """
    Enhanced analyzer and tester with improved code generation and validation
    """
    def __init__(self):
        self.project_root = Path(os.getcwd())
        sys.path.insert(0, str(self.project_root))
        
        self.test_dir = self.project_root / DEFAULT_TEST_DIR
        self.test_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.code_analyzer = CodeAnalyzer(self.project_root)
        self.test_validator = TestValidator(self.project_root)
        
        self.start_time = datetime.now()
        self.logs = []
        self.model_name = MODEL_ID

    def _log(self, message: str, level: str = "INFO"):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        print(message)

    def _get_changed_files(self) -> List[str]:
        """Gets changed files from the GITHUB_ENV variable."""
        changed_files_str = os.environ.get("CHANGED_FILES", "")
        if not changed_files_str.strip():
            self._log("ℹ️ No changed files detected in this PR.", "INFO")
            return []
        
        changed_files = changed_files_str.split()
        self._log(f"📁 Detected {len(changed_files)} changed files: {', '.join(changed_files)}", "INFO")
        return changed_files

    def _extract_functions_from_file(self, file_path: Path) -> List[FunctionInfo]:
        """Enhanced function extraction with detailed analysis"""
        functions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content, filename=file_path.name)
            
            # Get file-level imports
            file_imports = self.code_analyzer.extract_imports_from_file(file_path)
            
            # Find all classes first
            classes = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes[node.name] = node
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Determine if it's a method and get class context
                    is_method = False
                    class_name = None
                    parent_class = None
                    
                    for class_node in classes.values():
                        if node in class_node.body:
                            is_method = True
                            class_name = class_node.name
                            parent_class = class_node
                            break
                    
                    # Extract function code
                    start_line = node.lineno
                    end_line = node.end_lineno
                    lines = content.splitlines()
                    func_code = "\n".join(lines[start_line-1:end_line])
                    
                    # Extract docstring
                    docstring = ast.get_docstring(node)
                    
                    # Extract type information
                    type_info = self.code_analyzer.extract_type_info(node)
                    
                    # Calculate complexity
                    complexity = self.code_analyzer.calculate_complexity_score(node)
                    
                    # Create function info
                    func_info = FunctionInfo(
                        name=node.name,
                        file_path=str(file_path.relative_to(self.project_root)),
                        code=func_code,
                        line_start=start_line,
                        line_end=end_line,
                        is_method=is_method,
                        class_name=class_name,
                        docstring=docstring,
                        type_info=type_info,
                        import_info=file_imports,
                        complexity_score=complexity
                    )
                    
                    functions.append(func_info)
        
        except SyntaxError as e:
            self._log(f"⚠️ Syntax error in {file_path}: {e}", "WARNING")
        except Exception as e:
            self._log(f"⚠️ Error analyzing {file_path}: {e}", "WARNING")
        
        return functions

    def _build_comprehensive_prompt(self, functions: List[FunctionInfo]) -> str:
        """Build a comprehensive, structured prompt for better test generation"""
        
        # Analyze all functions to determine imports and dependencies
        all_imports = set()
        all_from_imports = {}
        all_classes = set()
        
        for func in functions:
            all_imports.update(func.import_info.module_imports)
            for module, items in func.import_info.from_imports.items():
                if module not in all_from_imports:
                    all_from_imports[module] = set()
                all_from_imports[module].update(items)
            all_classes.update(func.import_info.local_classes)
        
        # Build import statements
        import_statements = []
        
        # Add standard imports
        import_statements.extend([
            "import unittest",
            "from unittest.mock import Mock, patch, MagicMock",
            "import sys",
            "from pathlib import Path"
        ])
        
        # Add project-specific imports
        for module in sorted(all_imports):
            if not module.startswith('__'):
                import_statements.append(f"import {module}")
        
        for module, items in all_from_imports.items():
            if module and not module.startswith('__'):
                items_str = ', '.join(sorted(items))
                import_statements.append(f"from {module} import {items_str}")
        
        # Create function analysis section
        function_analyses = []
        for i, func in enumerate(functions, 1):
            analysis = [
                f"## Function {i}: {func.name}",
                f"**Type**: {'Method' if func.is_method else 'Function'}",
                f"**File**: {func.file_path}",
                f"**Lines**: {func.line_start}-{func.line_end}",
                f"**Complexity Score**: {func.complexity_score}/10"
            ]
            
            if func.class_name:
                analysis.append(f"**Class**: {func.class_name}")
            
            if func.type_info.param_types:
                param_info = []
                for param, ptype in func.type_info.param_types.items():
                    param_info.append(f"{param}: {ptype}")
                analysis.append(f"**Parameters**: {', '.join(param_info)}")
            
            if func.type_info.return_type:
                analysis.append(f"**Returns**: {func.type_info.return_type}")
            
            if func.docstring:
                analysis.append(f"**Docstring**: {func.docstring[:200]}...")
            
            analysis.append("**Code**:")
            analysis.append("```python")
            analysis.append(func.code)
            analysis.append("```")
            
            function_analyses.append("\n".join(analysis))
        
        # Build the comprehensive prompt
        prompt = f"""You are an expert Python unit testing engineer. Generate comprehensive, syntactically correct unit tests using Python's unittest framework.

## CRITICAL REQUIREMENTS FOR CODE GENERATION:

### 1. SYNTAX AND STRUCTURE:
- Generate ONLY valid Python code with no syntax errors
- Use proper indentation (4 spaces)
- Include proper class and method definitions
- Ensure all parentheses, brackets, and quotes are balanced
- Use proper exception handling syntax

### 2. IMPORTS (Use these exact imports):
```python
{chr(10).join(import_statements)}
```

### 3. TEST CLASS STRUCTURE:
```python
class TestGeneratedCode(unittest.TestCase):
    def setUp(self):
        # Setup code here if needed
        pass
    
    def test_function_name(self):
        # Test implementation
        pass
```

### 4. FUNCTIONS TO TEST:

{chr(10).join(function_analyses)}

## GENERATION RULES:

### A. For Regular Functions:
- Import the function directly
- Test with various input types and edge cases
- Use appropriate assertions (assertEqual, assertTrue, assertRaises, etc.)

### B. For Class Methods:
- Create class instances in setUp() or individual tests
- Test both instance and class methods appropriately
- Mock external dependencies when necessary

### C. Test Coverage Strategy:
- **Normal cases**: Test expected functionality
- **Edge cases**: Empty inputs, None values, boundary conditions
- **Error cases**: Invalid inputs, exception scenarios
- **Type validation**: Test with correct and incorrect types

### D. Mock Strategy:
- Mock file I/O operations
- Mock network calls
- Mock external dependencies
- Use unittest.mock.patch for system calls

### E. Assertions to Use:
- assertEqual(a, b) - for exact matches
- assertTrue(condition) - for boolean conditions
- assertFalse(condition) - for negative boolean conditions
- assertRaises(Exception, callable) - for exception testing
- assertIsInstance(obj, type) - for type checking
- assertIn(item, container) - for membership testing

## OUTPUT FORMAT:
Generate ONLY the complete Python test file with:
1. All necessary imports at the top
2. One TestCase class
3. setUp method if needed
4. Individual test methods for each function
5. if __name__ == '__main__': unittest.main() at the end

## VALIDATION CHECKLIST:
Before outputting, ensure:
- [ ] No syntax errors
- [ ] All imports are valid
- [ ] All test methods start with 'test_'
- [ ] Proper exception handling
- [ ] Balanced parentheses and quotes
- [ ] Correct indentation
- [ ] Valid Python identifiers

Generate the complete test file now:"""

        return prompt

    def _invoke_llm_for_generation(self, prompt: str, max_retries: int = 3) -> str:
        """Enhanced LLM invocation with retry logic and validation"""
        
        for attempt in range(max_retries):
            try:
                self._log(f"🤖 Generating test code (attempt {attempt + 1}/{max_retries})...", "INFO")
                
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a Python testing expert. Generate only syntactically correct, well-structured unit tests. Never include explanations or markdown formatting in your response - only pure Python code."},
                        {"role": "user", "content": prompt}
                    ],
                    model=MODEL_ID,
                    temperature=0.1,
                    max_tokens=4000,
                )
                
                generated_code = chat_completion.choices[0].message.content.strip()
                
                # Clean up markdown code fences if present
                if "```python" in generated_code:
                    code_start = generated_code.find("```python") + len("```python")
                    code_end = generated_code.rfind("```")
                    if code_end > code_start:
                        generated_code = generated_code[code_start:code_end].strip()
                
                if "```" in generated_code and "```python" not in generated_code:
                    # Handle generic code fences
                    code_start = generated_code.find("```") + 3
                    code_end = generated_code.rfind("```")
                    if code_end > code_start:
                        generated_code = generated_code[code_start:code_end].strip()
                
                # Validate syntax
                is_valid, syntax_errors = self.test_validator.validate_syntax(generated_code)
                
                if is_valid:
                    self._log("✅ Generated syntactically valid test code", "SUCCESS")
                    return generated_code
                else:
                    self._log(f"⚠️ Syntax validation failed (attempt {attempt + 1}): {'; '.join(syntax_errors)}", "WARNING")
                    if attempt < max_retries - 1:
                        # Add syntax error feedback to prompt for retry
                        prompt += f"\n\nPREVIOUS ATTEMPT HAD SYNTAX ERRORS:\n{'; '.join(syntax_errors)}\n\nPlease fix these issues and generate valid Python code."
                    
            except Exception as e:
                self._log(f"❌ Error in LLM generation (attempt {attempt + 1}): {e}", "ERROR")
                if attempt == max_retries - 1:
                    return ""
        
        self._log("❌ Failed to generate valid test code after all retries", "ERROR")
        return ""

    def _generate_test_suite(self, functions: List[FunctionInfo]) -> str:
        """Generate test suite with enhanced validation"""
        if not functions:
            return ""

        # Build comprehensive prompt
        prompt = self._build_comprehensive_prompt(functions)
        
        # Generate test code with retries
        test_code = self._invoke_llm_for_generation(prompt)
        
        if not test_code:
            return ""
        
        # Additional post-processing validation
        is_valid, errors = self.test_validator.validate_syntax(test_code)
        if not is_valid:
            self._log(f"⚠️ Final validation failed: {'; '.join(errors)}", "WARNING")
            # Try to fix common issues
            test_code = self._fix_common_syntax_issues(test_code)
        
        return test_code

    def _fix_common_syntax_issues(self, code: str) -> str:
        """Fix common syntax issues in generated code"""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix common indentation issues
            if line.strip() and not line.startswith(' ') and not line.startswith('import') and not line.startswith('from') and not line.startswith('class') and not line.startswith('def ') and not line.startswith('if __name__'):
                if fixed_lines and (fixed_lines[-1].strip().endswith(':') or 'def ' in fixed_lines[-1]):
                    line = '    ' + line
            
            # Fix missing colons after class/def/if statements
            stripped = line.strip()
            if (stripped.startswith(('def ', 'class ', 'if ', 'elif ', 'else', 'try', 'except', 'finally', 'for ', 'while ')) 
                and not stripped.endswith(':') 
                and not stripped.endswith(':\\') 
                and '#' not in stripped):
                line = line + ':'
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)

    def _format_python_code(self, file_path: Path):
        """Enhanced code formatting with validation"""
        try:
            # First, validate the file can be parsed
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                ast.parse(content)
            except SyntaxError as e:
                self._log(f"⚠️ Syntax error in generated test file: {e}", "WARNING")
                return False
            
            # Try to format with autopep8
            result = subprocess.run(
                ['autopep8', '--in-place', '--aggressive', '--aggressive', str(file_path)],
                check=True,
                capture_output=True,
                text=True
            )
            self._log(f"✔️ Formatted: {file_path.name}", "INFO")
            return True
            
        except FileNotFoundError:
            self._log("⚠️ autopep8 not found. Skipping code formatting.", "WARNING")
            return True
        except subprocess.CalledProcessError as e:
            self._log(f"❌ Failed to format {file_path.name}: {e.stderr}", "ERROR")
            return False

    def _parse_test_output(self, test_output: str, test_stderr: str) -> List[TestCaseResult]:
        """Parse unittest output to extract individual test case results"""
        test_cases = []
        
        # Combine output for parsing
        full_output = test_output + "\n" + test_stderr
        lines = full_output.split('\n')
        
        # Look for verbose unittest output patterns
        test_pattern = re.compile(r'^(\w+\.)?test_\w+.*\.\.\.')
        result_pattern = re.compile(r'^(test_\w+).*?\.\.\.\s+(ok|FAIL|ERROR|SKIP)')
        
        # Also look for test method names in failure/error blocks
        failure_test_pattern = re.compile(r'^(FAIL|ERROR):\s+(test_\w+)')
        
        current_failures = {}  # Store failure info by test name
        in_failure_block = False
        current_failure_test = None
        failure_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check for test execution results (verbose format)
            result_match = result_pattern.match(line_stripped)
            if result_match:
                test_name = result_match.group(1)
                status = result_match.group(2)
                
                # Map unittest results to our format
                if status == 'ok':
                    final_status = 'PASS'
                elif status == 'FAIL':
                    final_status = 'FAIL'
                elif status == 'ERROR':
                    final_status = 'ERROR'
                elif status == 'SKIP':
                    final_status = 'SKIP'
                else:
                    final_status = 'UNKNOWN'
                
                test_case = TestCaseResult(
                    name=test_name,
                    status=final_status,
                    test_method=test_name,
                    failure_reason=current_failures.get(test_name, "")
                )
                test_cases.append(test_case)
                continue
            
            # Check for failure/error headers
            failure_match = failure_test_pattern.match(line_stripped)
            if failure_match:
                current_failure_test = failure_match.group(2)
                in_failure_block = True
                failure_lines = []
                continue
            
            # Check for end of failure block
            if in_failure_block and line_stripped.startswith('='):
                if current_failure_test and failure_lines:
                    # Take first few lines of failure message
                    current_failures[current_failure_test] = '\n'.join(failure_lines[:3])
                in_failure_block = False
                current_failure_test = None
                failure_lines = []
                continue
            
            # Collect failure details
            if in_failure_block and line_stripped:
                failure_lines.append(line_stripped)
        
        # If no tests were found in output, try to extract from test file
        if not test_cases:
            test_cases = self._extract_test_methods_from_file()
            
            # Try to determine status from overall output
            if "FAILED" in full_output or "ERROR" in full_output:
                for test_case in test_cases:
                    test_case.status = "FAIL"
            elif "OK" in full_output or test_cases:
                for test_case in test_cases:
                    test_case.status = "PASS"
        
        return test_cases
    
    def _extract_test_methods_from_file(self) -> List[TestCaseResult]:
        """Extract test method names from the generated test file"""
        test_cases = []
        test_file_path = self.test_dir / "pr_generated_tests.py"
        
        if test_file_path.exists():
            try:
                with open(test_file_path, 'r') as f:
                    content = f.read()
                
                # Parse the AST to find test methods
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_cases.append(TestCaseResult(
                            name=node.name,
                            status="PASS",  # Assume PASS if we can't determine otherwise
                            test_method=node.name
                        ))
            except Exception as e:
                self._log(f"Warning: Could not parse test file for method names: {e}", "WARNING")
        
        return test_cases

    def _execute_tests(self, test_file_path: Path, changed_files: List[str]) -> Dict[str, Any]:
        """Enhanced test execution with better error handling"""
        self._log(f"🧪 Executing tests from {test_file_path}", "INFO")
        
        python_executable = sys.executable
        
        # Set up environment
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{str(self.project_root)}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(self.project_root)
        
        # Filter Python files
        python_changed_files = [
            file_path_str for file_path_str in changed_files
            if (self.project_root / file_path_str).exists() 
            and (self.project_root / file_path_str).suffix == '.py'
        ]
        
        try:
            # First, try to run the test file directly to check for import errors
            dry_run = subprocess.run(
                [python_executable, "-m", "py_compile", str(test_file_path)],
                capture_output=True, text=True,
                cwd=str(self.project_root),
                env=env
            )
            
            if dry_run.returncode != 0:
                self._log(f"❌ Test file compilation failed: {dry_run.stderr}", "ERROR")
                return {"status": "failure", "output": f"Compilation failed: {dry_run.stderr}", "test_cases": []}
            
            # Run tests with coverage and verbose output
            if python_changed_files:
                include_patterns = ','.join(python_changed_files)
                coverage_cmd = [
                    python_executable, "-m", "coverage", "run", 
                    f"--include={include_patterns}",
                    str(test_file_path), "-v"
                ]
            else:
                coverage_cmd = [
                    python_executable, "-m", "coverage", "run", 
                    "--source=.",
                    str(test_file_path), "-v"
                ]
            
            # Execute tests with timeout
            run_result = subprocess.run(
                coverage_cmd,
                capture_output=True, text=True, 
                timeout=300,  # 5 minute timeout
                cwd=str(self.project_root),
                env=env
            )
            
            # Generate coverage report
            coverage_result = subprocess.run(
                [python_executable, "-m", "coverage", "report", "--format=text"],
                capture_output=True, text=True,
                cwd=str(self.project_root),
                env=env
            )
            
            if run_result.returncode == 0:
                self._log("✅ Tests passed.", "SUCCESS")
                self._log(f"\n📊 Code Coverage Report:", "INFO")
                self._log(coverage_result.stdout, "INFO")
                
                # Parse individual test case results
                test_cases = self._parse_test_output(run_result.stdout, run_result.stderr)
                
                return {
                    "status": "success", 
                    "output": run_result.stdout,
                    "coverage_report": coverage_result.stdout,
                    "stderr": run_result.stderr,
                    "test_cases": test_cases
                }
            else:
                self._log("❌ Tests failed.", "ERROR")
                self._log("--- Test Output ---", "ERROR")
                self._log(run_result.stdout, "ERROR")
                self._log(run_result.stderr, "ERROR")
                self._log("-------------------", "ERROR")
                
                # Parse individual test case results even for failures
                test_cases = self._parse_test_output(run_result.stdout, run_result.stderr)
                
                return {
                    "status": "failure", 
                    "output": run_result.stdout + run_result.stderr,
                    "coverage_report": coverage_result.stdout if coverage_result.returncode == 0 else "",
                    "test_cases": test_cases
                }
                
        except subprocess.TimeoutExpired:
            self._log("❌ Test execution timed out after 5 minutes", "ERROR")
            return {"status": "failure", "output": "Test execution timed out", "test_cases": []}
        except Exception as e:
            self._log(f"❌ Test execution failed: {e}", "ERROR")
            return {"status": "failure", "output": f"Execution error: {str(e)}", "test_cases": []}

    def _parse_coverage_metrics(self, coverage_output: str) -> Dict[str, str]:
        """Enhanced coverage metrics parsing"""
        metrics = {}
        if not coverage_output:
            return metrics
            
        lines = coverage_output.strip().split('\n')
        
        for line in lines:
            if 'TOTAL' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        # Extract various metrics
                        statements = parts[1] if len(parts) > 1 else "0"
                        missing = parts[2] if len(parts) > 2 else "0"
                        coverage_percent = parts[-1].rstrip('%')
                        
                        metrics['total_coverage'] = coverage_percent
                        metrics['total_statements'] = statements
                        metrics['missing_statements'] = missing
                    except (ValueError, IndexError):
                        pass
                break
        
        return metrics

    def _generate_json_report(self, report: TestReport) -> str:
        """Enhanced JSON report generation"""
        report_dict = {
            "title": "🚀 Python Project Test Automation Report",
            "generated": report.timestamp,
            "model_name": report.model_name,
            "execution_time_seconds": report.execution_time,
            "status": report.status,
            "changed_files": report.changed_files,
            "syntax_validation": report.syntax_validation,
            "analyzed_functions": [
                {
                    "name": func.name,
                    "file_path": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "is_method": func.is_method,
                    "class_name": func.class_name,
                    "complexity_score": func.complexity_score,
                    "has_type_hints": bool(func.type_info.param_types or func.type_info.return_type),
                    "has_docstring": bool(func.docstring),
                    "code_snippet": func.code[:200] + "..." if len(func.code) > 200 else func.code
                }
                for func in report.analyzed_functions
            ],
            "test_results": report.test_results,
            "test_cases": [test_case.to_dict() for test_case in report.test_cases],
            "coverage_metrics": report.coverage_metrics,
            "execution_logs": report.logs
        }
        
        return json.dumps(report_dict, indent=2, ensure_ascii=False)

    def _generate_xml_report(self, report: TestReport) -> str:
        """Enhanced XML report generation"""
        root = ET.Element("test_automation_report")
        
        # Header with additional metadata
        header = ET.SubElement(root, "header")
        ET.SubElement(header, "title").text = "🚀 Python Project Test Automation Report"
        ET.SubElement(header, "generated").text = report.timestamp
        ET.SubElement(header, "model_name").text = report.model_name
        ET.SubElement(header, "execution_time_seconds").text = str(report.execution_time)
        ET.SubElement(header, "status").text = report.status
        
        # Syntax validation results
        syntax_elem = ET.SubElement(root, "syntax_validation")
        for key, value in report.syntax_validation.items():
            validation_elem = ET.SubElement(syntax_elem, "validation")
            validation_elem.set("test", key)
            validation_elem.text = str(value)
        
        # Enhanced function information
        functions_elem = ET.SubElement(root, "analyzed_functions")
        ET.SubElement(functions_elem, "count").text = str(len(report.analyzed_functions))
        for func in report.analyzed_functions:
            func_elem = ET.SubElement(functions_elem, "function")
            ET.SubElement(func_elem, "name").text = func.name
            ET.SubElement(func_elem, "file_path").text = func.file_path
            ET.SubElement(func_elem, "line_start").text = str(func.line_start)
            ET.SubElement(func_elem, "line_end").text = str(func.line_end)
            ET.SubElement(func_elem, "is_method").text = str(func.is_method)
            ET.SubElement(func_elem, "complexity_score").text = str(func.complexity_score)
            if func.class_name:
                ET.SubElement(func_elem, "class_name").text = func.class_name
            code_snippet = func.code[:200] + "..." if len(func.code) > 200 else func.code
            ET.SubElement(func_elem, "code_snippet").text = code_snippet
        
        # Test results with enhanced details
        test_results_elem = ET.SubElement(root, "test_results")
        ET.SubElement(test_results_elem, "status").text = report.test_results.get("status", "unknown")
        if "output" in report.test_results:
            ET.SubElement(test_results_elem, "output").text = report.test_results["output"]
        if "coverage_report" in report.test_results:
            ET.SubElement(test_results_elem, "coverage_report").text = report.test_results["coverage_report"]
        if "stderr" in report.test_results:
            ET.SubElement(test_results_elem, "stderr").text = report.test_results["stderr"]
        
        # Individual test cases
        test_cases_elem = ET.SubElement(root, "test_cases")
        for test_case in report.test_cases:
            case_elem = ET.SubElement(test_cases_elem, "test_case")
            case_elem.set("name", test_case.name)
            case_elem.set("status", test_case.status)
            if test_case.execution_time > 0:
                case_elem.set("execution_time", str(test_case.execution_time))
            if test_case.failure_reason:
                ET.SubElement(case_elem, "failure_reason").text = test_case.failure_reason
            if test_case.test_method:
                ET.SubElement(case_elem, "test_method").text = test_case.test_method
        
        # Coverage metrics
        coverage_elem = ET.SubElement(root, "coverage_metrics")
        for key, value in report.coverage_metrics.items():
            metric_elem = ET.SubElement(coverage_elem, "metric")
            metric_elem.set("name", key)
            metric_elem.text = value
        
        # Execution logs
        logs_elem = ET.SubElement(root, "execution_logs")
        ET.SubElement(logs_elem, "count").text = str(len(report.logs))
        for log_entry in report.logs:
            log_elem = ET.SubElement(logs_elem, "log")
            log_elem.text = log_entry
        
        return ET.tostring(root, encoding='unicode', method='xml')

    def _generate_text_report(self, report: TestReport) -> str:
        """Enhanced human-readable text format report"""
        lines = [
            "🚀 Python Project Test Automation Report",
            f"Generated: {report.timestamp}",
            f"Model Used: {report.model_name}",
            f"Execution Time: {report.execution_time:.2f} seconds",
            f"Status: {report.status}",
            "",
            "=" * 60,
            "",
            "📁 CHANGED FILES:",
            ""
        ]
        
        for i, file_path in enumerate(report.changed_files, 1):
            lines.append(f"  {i}. {file_path}")
        
        # Syntax validation results
        if report.syntax_validation:
            lines.extend([
                "",
                "🔍 SYNTAX VALIDATION:",
                ""
            ])
            for test_name, is_valid in report.syntax_validation.items():
                status = "✅ PASS" if is_valid else "❌ FAIL"
                lines.append(f"  {test_name}: {status}")
        
        lines.extend([
            "",
            f"🔍 ANALYZED FUNCTIONS/METHODS ({len(report.analyzed_functions)} total):",
            ""
        ])
        
        for i, func in enumerate(report.analyzed_functions, 1):
            func_type = "Method" if func.is_method else "Function"
            class_info = f" (Class: {func.class_name})" if func.class_name else ""
            complexity_info = f" [Complexity: {func.complexity_score}/10]"
            type_hints = " [Type Hints: ✅]" if func.type_info.param_types or func.type_info.return_type else " [Type Hints: ❌]"
            docstring_info = " [Docstring: ✅]" if func.docstring else " [Docstring: ❌]"
            
            lines.append(f"  {i}. {func_type}: {func.name}{class_info}{complexity_info}{type_hints}{docstring_info}")
            lines.append(f"     File: {func.file_path} (lines {func.line_start}-{func.line_end})")
            lines.append("")
        
        lines.extend([
            "🧪 TEST RESULTS:",
            f"  Status: {report.test_results.get('status', 'unknown').upper()}",
            ""
        ])
        
        # Add individual test case results
        if report.test_cases:
            lines.extend([
                "📋 INDIVIDUAL TEST CASES:",
                ""
            ])
            
            for i, test_case in enumerate(report.test_cases, 1):
                status_emoji = "✅" if test_case.status == "PASS" else "❌" if test_case.status in ["FAIL", "ERROR"] else "⚠️"
                lines.append(f"  {i}. {test_case.name}")
                lines.append(f"     Status: {status_emoji} {test_case.status}")
                
                if test_case.failure_reason and test_case.status in ["FAIL", "ERROR"]:
                    lines.append(f"     Reason: {test_case.failure_reason}")
                
                if test_case.execution_time > 0:
                    lines.append(f"     Time: {test_case.execution_time:.3f}s")
                
                lines.append("")
        
        if "output" in report.test_results and report.test_results["output"]:
            lines.extend([
                "  Test Output:",
                "  " + "─" * 40,
                *[f"  {line}" for line in report.test_results["output"].split('\n')[:15]],
                "  " + "─" * 40,
                ""
            ])
        
        lines.extend([
            "📊 COVERAGE METRICS:",
            ""
        ])
        
        for key, value in report.coverage_metrics.items():
            lines.append(f"  {key.replace('_', ' ').title()}: {value}")
        
        if report.coverage_metrics:
            lines.append("")
        
        lines.extend([
            "📝 EXECUTION LOGS (Last 25 entries):",
            ""
        ])
        
        for log_entry in report.logs[-25:]:
            lines.append(f"  {log_entry}")
        
        lines.extend([
            "",
            "=" * 60,
            f"Report completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return '\n'.join(lines)

    def _save_reports(self, report: TestReport):
        """Save enhanced reports in multiple formats"""
        timestamp_str = report.timestamp.replace(":", "-").replace(" ", "_")
        base_filename = f"test_automation_report_{timestamp_str.split('_')[0]}_{timestamp_str.split('_')[1]}"
        
        # Save JSON report
        json_content = self._generate_json_report(report)
        json_path = self.reports_dir / f"{base_filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_content)
        self._log(f"📄 JSON report saved: {json_path}", "INFO")
        
        # Save XML report
        xml_content = self._generate_xml_report(report)
        xml_path = self.reports_dir / f"{base_filename}.xml"
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        self._log(f"📄 XML report saved: {xml_path}", "INFO")
        
        # Save text report
        text_content = self._generate_text_report(report)
        text_path = self.reports_dir / f"{base_filename}.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        self._log(f"📄 Text report saved: {text_path}", "INFO")

    def run(self):
        """Enhanced main runner with comprehensive validation"""
        self._log("🚀 Starting Enhanced Python Project Test Automation", "INFO")
        
        changed_files = self._get_changed_files()
        if not changed_files:
            self._log("No Python files changed. Exiting CI run.", "INFO")
            return

        all_changed_functions = []
        syntax_validation = {}
        
        for file_path_str in changed_files:
            file_path = self.project_root / file_path_str
            if file_path.exists() and file_path.suffix == '.py':
                self._log(f"🔍 Analyzing changed file: {file_path_str}", "INFO")
                
                # Validate file syntax first
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    ast.parse(content, filename=file_path.name)
                    syntax_validation[file_path_str] = True
                    
                    functions = self._extract_functions_from_file(file_path)
                    all_changed_functions.extend(functions)
                    self._log(f"✅ Found {len(functions)} functions/methods in {file_path_str}", "INFO")
                    
                except SyntaxError as e:
                    syntax_validation[file_path_str] = False
                    self._log(f"❌ Syntax error in {file_path_str}: {e}", "ERROR")
                except Exception as e:
                    syntax_validation[file_path_str] = False
                    self._log(f"❌ Error analyzing {file_path_str}: {e}", "ERROR")

        if not all_changed_functions:
            self._log("No valid functions found in changed Python files. Exiting.", "WARNING")
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            report = TestReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=self.model_name,
                changed_files=changed_files,
                analyzed_functions=all_changed_functions,
                test_results={"status": "skipped", "output": "No valid functions found"},
                coverage_metrics={},
                status="SKIPPED",
                execution_time=execution_time,
                logs=self.logs,
                syntax_validation=syntax_validation
            )
            self._save_reports(report)
            return

        # Log function analysis summary
        total_functions = len(all_changed_functions)
        methods_count = sum(1 for f in all_changed_functions if f.is_method)
        functions_count = total_functions - methods_count
        avg_complexity = sum(f.complexity_score for f in all_changed_functions) / total_functions
        
        self._log(f"📊 Analysis Summary:", "INFO")
        self._log(f"   Total Functions/Methods: {total_functions}", "INFO")
        self._log(f"   Functions: {functions_count}, Methods: {methods_count}", "INFO")
        self._log(f"   Average Complexity: {avg_complexity:.1f}/10", "INFO")
        
        # Generate test suite using enhanced LLM
        self._log(f"🧠 Generating comprehensive test cases using {self.model_name}...", "INFO")
        test_content = self._generate_test_suite(all_changed_functions)
        
        if not test_content:
            self._log("❌ Failed to generate test cases. Aborting.", "ERROR")
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            report = TestReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=self.model_name,
                changed_files=changed_files,
                analyzed_functions=all_changed_functions,
                test_results={"status": "failure", "output": "Failed to generate test cases"},
                coverage_metrics={},
                status="FAILED",
                execution_time=execution_time,
                logs=self.logs,
                syntax_validation=syntax_validation
            )
            self._save_reports(report)
            return

        # Save and validate test file
        test_file_name = "pr_generated_tests.py"
        test_file_path = self.test_dir / test_file_name
        
        try:
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            # Validate and format the generated test file
            is_valid_syntax = self._format_python_code(test_file_path)
            syntax_validation['generated_test_file'] = is_valid_syntax
            
            if is_valid_syntax:
                self._log(f"✅ Generated and saved syntactically valid tests to {test_file_path}", "SUCCESS")
            else:
                self._log(f"⚠️ Generated tests have syntax issues, but saved to {test_file_path}", "WARNING")
                
        except Exception as e:
            self._log(f"❌ Failed to save test file: {e}", "ERROR")
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            report = TestReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=self.model_name,
                changed_files=changed_files,
                analyzed_functions=all_changed_functions,
                test_results={"status": "failure", "output": f"Failed to save test file: {e}"},
                coverage_metrics={},
                status="FAILED",
                execution_time=execution_time,
                logs=self.logs,
                syntax_validation=syntax_validation
            )
            self._save_reports(report)
            return

        # Execute tests with enhanced error handling
        self._log("🏃 Executing generated tests with comprehensive coverage analysis...", "INFO")
        test_results = self._execute_tests(test_file_path, changed_files)
        
        # Parse coverage metrics
        coverage_metrics = {}
        if test_results.get("coverage_report"):
            coverage_metrics = self._parse_coverage_metrics(test_results["coverage_report"])
            if "total_coverage" in coverage_metrics:
                coverage_percent = coverage_metrics["total_coverage"]
                self._log(f"\n🎯 Code Coverage: {coverage_percent}%", "INFO")
                
                try:
                    coverage_float = float(coverage_percent)
                    if coverage_float >= 90:
                        self._log("🟢 Excellent coverage!", "SUCCESS")
                    elif coverage_float >= 75:
                        self._log("🟡 Good coverage, consider adding edge case tests", "INFO")
                    elif coverage_float >= 50:
                        self._log("🟠 Moderate coverage, more comprehensive tests recommended", "WARNING")
                    else:
                        self._log("🔴 Low coverage, significant testing improvements needed", "WARNING")
                except ValueError:
                    pass

        # Calculate execution time and determine status
        execution_time = (datetime.now() - self.start_time).total_seconds()
        overall_status = "SUCCESS" if test_results["status"] == "success" else "FAILED"
        
        # Create comprehensive report
        report = TestReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name=self.model_name,
            changed_files=changed_files,
            analyzed_functions=all_changed_functions,
            test_results=test_results,
            coverage_metrics=coverage_metrics,
            status=overall_status,
            execution_time=execution_time,
            logs=self.logs,
            syntax_validation=syntax_validation,
            test_cases=test_results.get("test_cases", [])
        )
        
        # Save comprehensive reports
        self._save_reports(report)
        
        # Final summary
        self._log(f"✨ Enhanced test automation completed in {execution_time:.2f} seconds", "INFO")
        self._log(f"📊 Final Status: {overall_status}", "INFO")
        self._log(f"🔧 Syntax Validation: {sum(syntax_validation.values())}/{len(syntax_validation)} files passed", "INFO")
        
        if coverage_metrics.get("total_coverage"):
            self._log(f"📈 Coverage Achieved: {coverage_metrics['total_coverage']}%", "INFO")


if __name__ == "__main__":
    runner = ChangeAnalyzerAndTester()
    runner.run()
