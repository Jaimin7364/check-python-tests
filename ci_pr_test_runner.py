import os
import subprocess
import json
import sys
import re
import ast
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
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

@dataclass
class TestReport:
    """Data structure for test execution report"""
    timestamp: str
    model_name: str
    changed_files: List[str]
    analyzed_functions: List[FunctionInfo]
    test_results: Dict[str, any]
    coverage_metrics: Dict[str, str]
    status: str
    execution_time: float
    logs: List[str] = field(default_factory=list)

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
        
        # Create reports directory
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize the Groq client
        self.client = Groq(api_key=GROQ_API_KEY)
        
        # Initialize reporting variables
        self.start_time = datetime.now()
        self.logs = []
        self.model_name = MODEL_ID

    def _log(self, message: str, level: str = "INFO"):
        """Add a log entry with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        print(message)  # Still print to console

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
            self._log(f"⚠️ Syntax error in {file_path}: {e}", "WARNING")
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
            self._log(f"❌ Error generating test code with Groq LLM: {e}", "ERROR")
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
            self._log(f"✔️ Formatted: {file_path.name}", "INFO")
        except FileNotFoundError:
            self._log("⚠️ autopep8 not found. Skipping code formatting.", "WARNING")
        except subprocess.CalledProcessError as e:
            self._log(f"❌ Failed to format {file_path.name}: {e.stderr}", "ERROR")

    def _execute_tests(self, test_file_path: Path, changed_files: List[str]) -> Dict[str, any]:
        """Executes the generated test file with code coverage for only the changed files."""
        self._log(f"🧪 Executing tests from {test_file_path}", "INFO")
        
        # Get the current Python executable instead of hardcoded 'python'
        python_executable = sys.executable
        
        # Set PYTHONPATH to include the project root for imports
        env = os.environ.copy()
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{str(self.project_root)}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(self.project_root)
        
        # Filter only Python files from changed files
        python_changed_files = []
        for file_path_str in changed_files:
            file_path = self.project_root / file_path_str
            if file_path.exists() and file_path.suffix == '.py':
                python_changed_files.append(file_path_str)
        
        try:
            # Run tests with coverage only for changed Python files
            coverage_file = self.project_root / ".coverage"
            
            if python_changed_files:
                # Build coverage command with specific source files
                # Use --include pattern to only measure coverage for changed files
                include_patterns = ','.join(python_changed_files)
                coverage_cmd = [
                    python_executable, "-m", "coverage", "run", 
                    f"--include={include_patterns}",
                    str(test_file_path)
                ]
            else:
                # Fallback to general coverage if no Python files changed
                coverage_cmd = [
                    python_executable, "-m", "coverage", "run", 
                    "--source=.",
                    str(test_file_path)
                ]
            
            # First, run the tests with coverage
            run_result = subprocess.run(
                coverage_cmd,
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
            
            self._log("✅ Tests passed.", "SUCCESS")
            self._log(f"\n📊 Code Coverage Report (Changed Files Only):", "INFO")
            self._log(coverage_result.stdout, "INFO")
            
            if coverage_html_result.returncode == 0:
                self._log("📄 Detailed HTML coverage report generated in 'htmlcov/' directory", "INFO")
            
            return {
                "status": "success", 
                "output": run_result.stdout,
                "coverage_report": coverage_result.stdout
            }
            
        except subprocess.CalledProcessError as e:
            self._log("❌ Tests failed.", "ERROR")
            self._log("--- Test Output ---", "ERROR")
            self._log(e.stdout, "ERROR")
            self._log(e.stderr, "ERROR")
            self._log("-------------------", "ERROR")
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

    def _generate_json_report(self, report: TestReport) -> str:
        """Generate JSON format report"""
        # Convert dataclass to dictionary for JSON serialization
        report_dict = {
            "title": "🚀 Python Project Test Automation Report",
            "generated": report.timestamp,
            "model_name": report.model_name,
            "execution_time_seconds": report.execution_time,
            "status": report.status,
            "changed_files": report.changed_files,
            "analyzed_functions": [
                {
                    "name": func.name,
                    "file_path": func.file_path,
                    "line_start": func.line_start,
                    "line_end": func.line_end,
                    "is_method": func.is_method,
                    "class_name": func.class_name,
                    "code_snippet": func.code[:200] + "..." if len(func.code) > 200 else func.code
                }
                for func in report.analyzed_functions
            ],
            "test_results": report.test_results,
            "coverage_metrics": report.coverage_metrics,
            "execution_logs": report.logs
        }
        
        return json.dumps(report_dict, indent=2, ensure_ascii=False)

    def _generate_xml_report(self, report: TestReport) -> str:
        """Generate XML format report"""
        root = ET.Element("test_automation_report")
        
        # Header
        header = ET.SubElement(root, "header")
        ET.SubElement(header, "title").text = "🚀 Python Project Test Automation Report"
        ET.SubElement(header, "generated").text = report.timestamp
        ET.SubElement(header, "model_name").text = report.model_name
        ET.SubElement(header, "execution_time_seconds").text = str(report.execution_time)
        ET.SubElement(header, "status").text = report.status
        
        # Changed files
        changed_files_elem = ET.SubElement(root, "changed_files")
        ET.SubElement(changed_files_elem, "count").text = str(len(report.changed_files))
        for file_path in report.changed_files:
            file_elem = ET.SubElement(changed_files_elem, "file")
            file_elem.text = file_path
        
        # Analyzed functions
        functions_elem = ET.SubElement(root, "analyzed_functions")
        ET.SubElement(functions_elem, "count").text = str(len(report.analyzed_functions))
        for func in report.analyzed_functions:
            func_elem = ET.SubElement(functions_elem, "function")
            ET.SubElement(func_elem, "name").text = func.name
            ET.SubElement(func_elem, "file_path").text = func.file_path
            ET.SubElement(func_elem, "line_start").text = str(func.line_start)
            ET.SubElement(func_elem, "line_end").text = str(func.line_end)
            ET.SubElement(func_elem, "is_method").text = str(func.is_method)
            if func.class_name:
                ET.SubElement(func_elem, "class_name").text = func.class_name
            code_snippet = func.code[:200] + "..." if len(func.code) > 200 else func.code
            ET.SubElement(func_elem, "code_snippet").text = code_snippet
        
        # Test results
        test_results_elem = ET.SubElement(root, "test_results")
        ET.SubElement(test_results_elem, "status").text = report.test_results.get("status", "unknown")
        if "output" in report.test_results:
            ET.SubElement(test_results_elem, "output").text = report.test_results["output"]
        if "coverage_report" in report.test_results:
            ET.SubElement(test_results_elem, "coverage_report").text = report.test_results["coverage_report"]
        
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
        """Generate human-readable text format report"""
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
        
        lines.extend([
            "",
            f"🔍 ANALYZED FUNCTIONS/METHODS ({len(report.analyzed_functions)} total):",
            ""
        ])
        
        for i, func in enumerate(report.analyzed_functions, 1):
            func_type = "Method" if func.is_method else "Function"
            class_info = f" (Class: {func.class_name})" if func.class_name else ""
            lines.append(f"  {i}. {func_type}: {func.name}{class_info}")
            lines.append(f"     File: {func.file_path} (lines {func.line_start}-{func.line_end})")
            lines.append("")
        
        lines.extend([
            "🧪 TEST RESULTS:",
            f"  Status: {report.test_results.get('status', 'unknown').upper()}",
            ""
        ])
        
        if "output" in report.test_results and report.test_results["output"]:
            lines.extend([
                "  Test Output:",
                "  " + "─" * 40,
                *[f"  {line}" for line in report.test_results["output"].split('\n')[:10]],
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
            "📝 EXECUTION LOGS:",
            ""
        ])
        
        for log_entry in report.logs[-20:]:  # Show last 20 log entries
            lines.append(f"  {log_entry}")
        
        lines.extend([
            "",
            "=" * 60,
            f"Report completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return '\n'.join(lines)

    def _save_reports(self, report: TestReport):
        """Save reports in multiple formats"""
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
        """Main runner for the CI script."""
        self._log("🚀 Starting Python Project Test Automation", "INFO")
        
        changed_files = self._get_changed_files()
        if not changed_files:
            self._log("No Python files changed. Exiting CI run.", "INFO")
            return

        all_changed_functions = []
        for file_path_str in changed_files:
            file_path = self.project_root / file_path_str
            if file_path.exists() and file_path.suffix == '.py':
                self._log(f"🔍 Analyzing changed file: {file_path_str}", "INFO")
                functions = self._extract_functions_from_file(file_path)
                all_changed_functions.extend(functions)

        if not all_changed_functions:
            self._log("No functions found in changed Python files. Exiting.", "WARNING")
            return

        self._log(f"🤖 Found {len(all_changed_functions)} functions/methods to test using {self.model_name}", "INFO")
        
        # Generate test suite using LLM
        self._log(f"🧠 Generating test cases using {self.model_name}...", "INFO")
        test_content = self._generate_test_suite(all_changed_functions)
        if not test_content:
            self._log("❌ Failed to generate test cases. Aborting.", "ERROR")
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            # Create report for failed generation
            report = TestReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=self.model_name,
                changed_files=changed_files,
                analyzed_functions=all_changed_functions,
                test_results={"status": "failure", "output": "Failed to generate test cases"},
                coverage_metrics={},
                status="FAILED",
                execution_time=execution_time,
                logs=self.logs
            )
            self._save_reports(report)
            return

        test_file_name = "pr_generated_tests.py"
        test_file_path = self.test_dir / test_file_name
        try:
            with open(test_file_path, 'w', encoding='utf-8') as f:
                f.write(test_content)
            self._format_python_code(test_file_path)
            self._log(f"✅ Generated and saved tests to {test_file_path}", "SUCCESS")
        except Exception as e:
            self._log(f"❌ Failed to save test file: {e}", "ERROR")
            execution_time = (datetime.now() - self.start_time).total_seconds()
            
            # Create report for failed save
            report = TestReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_name=self.model_name,
                changed_files=changed_files,
                analyzed_functions=all_changed_functions,
                test_results={"status": "failure", "output": f"Failed to save test file: {e}"},
                coverage_metrics={},
                status="FAILED",
                execution_time=execution_time,
                logs=self.logs
            )
            self._save_reports(report)
            return

        # Execute tests and get coverage results
        self._log("🏃 Executing generated tests with coverage analysis...", "INFO")
        test_results = self._execute_tests(test_file_path, changed_files)
        
        # Parse coverage metrics
        coverage_metrics = {}
        if test_results["status"] == "success" and "coverage_report" in test_results:
            coverage_metrics = self._parse_coverage_metrics(test_results["coverage_report"])
            if "total_coverage" in coverage_metrics:
                coverage_percent = coverage_metrics["total_coverage"]
                self._log(f"\n🎯 Changed Files Code Coverage: {coverage_percent}%", "INFO")
                
                # Provide coverage feedback
                try:
                    coverage_float = float(coverage_percent)
                    if coverage_float >= 90:
                        self._log("🟢 Excellent coverage!", "SUCCESS")
                    elif coverage_float >= 75:
                        self._log("🟡 Good coverage, consider adding more tests", "INFO")
                    elif coverage_float >= 50:
                        self._log("🟠 Moderate coverage, more tests recommended", "WARNING")
                    else:
                        self._log("🔴 Low coverage, significant testing gaps detected", "WARNING")
                except ValueError:
                    pass

        # Calculate execution time
        execution_time = (datetime.now() - self.start_time).total_seconds()
        
        # Determine overall status
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
            logs=self.logs
        )
        
        # Save reports in multiple formats
        self._save_reports(report)
        self._log(f"✨ Test automation completed in {execution_time:.2f} seconds with status: {overall_status}", "INFO")

if __name__ == "__main__":
    runner = ChangeAnalyzerAndTester()
    runner.run()