#!/usr/bin/env python3
"""
Test suite for code execution functionality in the Anthropic provider.
"""

import os
import sys
import time
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_providers.anthropic_provider import AnthropicProvider
from utils.sandbox import CodeExecutionSandbox, DockerSandbox, create_sandbox
from utils.config import LLMConfig

class TestCodeExecutionSandbox(unittest.TestCase):
    """Test cases for the CodeExecutionSandbox class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sandbox = CodeExecutionSandbox(
            timeout=10,
            max_memory_mb=256,
            enable_network=False
        )
    
    def test_validate_code_python_valid(self):
        """Test code validation for valid Python code."""
        code = """
import math
x = 5
y = math.sqrt(x)
print(f"Square root of {x} is {y}")
"""
        is_valid, error = self.sandbox.validate_code(code, "python")
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
    
    def test_validate_code_python_invalid_import(self):
        """Test code validation for invalid Python imports."""
        code = """
import os
os.system("echo 'This should not run'")
"""
        is_valid, error = self.sandbox.validate_code(code, "python")
        self.assertFalse(is_valid)
        self.assertIn("Restricted import detected: os", error)
    
    def test_validate_code_dangerous_patterns(self):
        """Test code validation for dangerous patterns."""
        dangerous_codes = [
            "exec('malicious code')",
            "eval('dangerous expression')",
            "__import__('os').system('bad command')",
            "open('/etc/passwd', 'r').read()"
        ]
        
        for code in dangerous_codes:
            is_valid, error = self.sandbox.validate_code(code, "python")
            self.assertFalse(is_valid, f"Code should be invalid: {code}")
            self.assertIn("Potentially dangerous code pattern", error)
    
    def test_validate_code_unsupported_language(self):
        """Test code validation for unsupported languages."""
        code = "console.log('Hello, World!');"
        is_valid, error = self.sandbox.validate_code(code, "unsupported")
        self.assertFalse(is_valid)
        self.assertIn("Language 'unsupported' not allowed", error)
    
    def test_execute_code_python_success(self):
        """Test successful Python code execution."""
        code = """
import math
result = math.sqrt(16)
print(f"Result: {result}")
"""
        result = self.sandbox.execute_code(code, "python")
        
        self.assertTrue(result["success"])
        self.assertIn("Result: 4.0", result["stdout"])
        self.assertEqual(result["return_code"], 0)
        self.assertGreater(result["execution_time"], 0)
    
    def test_execute_code_python_error(self):
        """Test Python code execution with runtime error."""
        code = """
x = 1 / 0  # Division by zero
print("This should not print")
"""
        result = self.sandbox.execute_code(code, "python")
        
        self.assertFalse(result["success"])
        self.assertIn("ZeroDivisionError", result["stderr"])
        self.assertNotEqual(result["return_code"], 0)
    
    def test_execute_code_timeout(self):
        """Test code execution timeout."""
        code = """
import time
time.sleep(15)  # Sleep longer than timeout
print("This should not print")
"""
        # Use a short timeout for testing
        short_timeout_sandbox = CodeExecutionSandbox(timeout=2)
        result = short_timeout_sandbox.execute_code(code, "python")
        
        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])
    
    def test_execute_code_javascript_success(self):
        """Test successful JavaScript code execution."""
        code = """
const result = Math.sqrt(25);
console.log(`Result: ${result}`);
"""
        result = self.sandbox.execute_code(code, "javascript")
        
        self.assertTrue(result["success"])
        self.assertIn("Result: 5", result["stdout"])
        self.assertEqual(result["return_code"], 0)
    
    def test_execute_code_security_validation_failure(self):
        """Test that security validation prevents execution."""
        code = """
import subprocess
subprocess.run(['echo', 'This should not run'])
"""
        result = self.sandbox.execute_code(code, "python")
        
        self.assertFalse(result["success"])
        self.assertIn("Security validation failed", result["error"])
        self.assertIn("Restricted import detected", result["error"])


class TestDockerSandbox(unittest.TestCase):
    """Test cases for the DockerSandbox class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sandbox = DockerSandbox(
            timeout=10,
            memory_limit="256m",
            network_disabled=True
        )
    
    def test_docker_availability_check(self):
        """Test Docker availability checking."""
        # This test depends on whether Docker is actually installed
        # We'll just ensure the method doesn't crash
        available = self.sandbox._check_docker_available()
        self.assertIsInstance(available, bool)
    
    @unittest.skipUnless(os.system("docker --version >/dev/null 2>&1") == 0, 
                        "Docker not available")
    def test_execute_code_docker_python(self):
        """Test Python code execution in Docker."""
        code = """
import math
result = math.factorial(5)
print(f"5! = {result}")
"""
        result = self.sandbox.execute_code(code, "python")
        
        self.assertTrue(result["success"])
        self.assertIn("5! = 120", result["stdout"])
        self.assertEqual(result["return_code"], 0)
    
    def test_execute_code_docker_unavailable(self):
        """Test behavior when Docker is not available."""
        # Mock Docker as unavailable
        with patch.object(self.sandbox, 'docker_available', False):
            result = self.sandbox.execute_code("print('hello')", "python")
            
            self.assertFalse(result["success"])
            self.assertIn("Docker not available", result["error"])


class TestSandboxFactory(unittest.TestCase):
    """Test cases for the sandbox factory function."""
    
    def test_create_sandbox_default(self):
        """Test creating default sandbox."""
        sandbox = create_sandbox()
        self.assertIsInstance(sandbox, CodeExecutionSandbox)
    
    def test_create_sandbox_with_docker(self):
        """Test creating sandbox with Docker preference."""
        sandbox = create_sandbox(use_docker=True)
        # Should return either DockerSandbox or fallback to CodeExecutionSandbox
        self.assertTrue(isinstance(sandbox, (DockerSandbox, CodeExecutionSandbox)))
    
    def test_create_sandbox_with_config(self):
        """Test creating sandbox with custom configuration."""
        sandbox = create_sandbox(
            timeout=60,
            max_memory_mb=1024,
            enable_network=True
        )
        
        self.assertIsInstance(sandbox, CodeExecutionSandbox)
        self.assertEqual(sandbox.timeout, 60)
        self.assertEqual(sandbox.max_memory_mb, 1024)
        self.assertTrue(sandbox.enable_network)


class TestAnthropicProviderCodeExecution(unittest.TestCase):
    """Test cases for code execution in AnthropicProvider."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the Anthropic client to avoid actual API calls
        with patch('llm_providers.anthropic_provider.Anthropic'):
            self.provider = AnthropicProvider(
                api_key="test_key",
                enable_code_execution=True,
                enable_mcp=False,
                enable_web_search=False
            )
    
    def test_supports_code_execution(self):
        """Test that provider reports code execution support."""
        self.assertTrue(self.provider.supports_code_execution)
    
    def test_execute_code_success(self):
        """Test successful code execution through provider."""
        code = """
result = 2 + 3
print(f"2 + 3 = {result}")
"""
        result = self.provider.execute_code(code, "python")
        
        self.assertIsNotNone(result)
        self.assertIn("2 + 3 = 5", result)
    
    def test_execute_code_with_error(self):
        """Test code execution with runtime error."""
        code = """
undefined_variable = some_nonexistent_variable
"""
        result = self.provider.execute_code(code, "python")
        
        self.assertIsNotNone(result)
        self.assertIn("Execution failed", result)
    
    def test_execute_code_disabled(self):
        """Test code execution when disabled."""
        # Create provider with code execution disabled
        with patch('llm_providers.anthropic_provider.Anthropic'):
            disabled_provider = AnthropicProvider(
                api_key="test_key",
                enable_code_execution=False
            )
        
        result = disabled_provider.execute_code("print('hello')", "python")
        self.assertIsNone(result)
    
    def test_execute_code_sandbox_failure(self):
        """Test behavior when sandbox initialization fails."""
        # Mock sandbox creation to fail
        with patch('llm_providers.anthropic_provider.create_sandbox', 
                  side_effect=Exception("Sandbox creation failed")):
            with patch('llm_providers.anthropic_provider.Anthropic'):
                provider = AnthropicProvider(
                    api_key="test_key",
                    enable_code_execution=True
                )
            
            result = provider.execute_code("print('hello')", "python")
            self.assertIsNone(result)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete code execution pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig()
    
    def test_config_code_execution_settings(self):
        """Test configuration system for code execution."""
        # Test getting code execution status
        status = self.config.get_code_execution_status()
        self.assertIsInstance(status, dict)
        self.assertIn("anthropic", status)
        
        # Test enabling/disabling code execution
        self.config.set_code_execution_enabled("anthropic", True)
        self.assertTrue(self.config.get_code_execution_enabled("anthropic"))
        
        self.config.set_code_execution_enabled("anthropic", False)
        self.assertFalse(self.config.get_code_execution_enabled("anthropic"))
    
    def test_provider_capabilities(self):
        """Test provider capabilities reporting."""
        capabilities = self.config.get_provider_capabilities("anthropic")
        self.assertIsInstance(capabilities, dict)
        self.assertIn("code_execution", capabilities)
        self.assertIn("mcp", capabilities)
        self.assertIn("web_search", capabilities)
        self.assertIn("pdf_upload", capabilities)


def run_performance_tests():
    """Run performance tests for code execution."""
    print("Running performance tests...")
    
    sandbox = CodeExecutionSandbox(timeout=30)
    
    # Test execution time for simple operations
    start_time = time.time()
    result = sandbox.execute_code("print('Hello, World!')", "python")
    execution_time = time.time() - start_time
    
    print(f"Simple Python execution: {execution_time:.3f}s")
    print(f"Sandbox overhead: {execution_time - result['execution_time']:.3f}s")
    
    # Test memory usage (approximate)
    memory_test_code = """
import sys
data = [i for i in range(1000000)]  # Create a large list
print(f"Memory usage test completed. List length: {len(data)}")
"""
    
    start_time = time.time()
    result = sandbox.execute_code(memory_test_code, "python")
    execution_time = time.time() - start_time
    
    print(f"Memory test execution: {execution_time:.3f}s")
    print(f"Success: {result['success']}")


if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "="*50)
    run_performance_tests()
    
    print("\nAll tests completed!")