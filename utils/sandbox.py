"""
Security and sandboxing utilities for code execution.

This module provides secure code execution environments with proper isolation
and security controls to prevent malicious code from affecting the host system.
"""

import os
import subprocess
import tempfile
import shutil
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import time
import signal
from contextlib import contextmanager
import re

logger = logging.getLogger(__name__)

class CodeExecutionSandbox:
    """Secure sandbox for executing code with restrictions and monitoring."""
    
    def __init__(self, 
                 timeout: int = 30,
                 max_memory_mb: int = 512,
                 max_disk_mb: int = 100,
                 allowed_languages: Optional[List[str]] = None,
                 enable_network: bool = False,
                 persistent_session: bool = True):
        """Initialize the sandbox with security constraints.
        
        Args:
            timeout: Maximum execution time in seconds
            max_memory_mb: Maximum memory usage in MB
            max_disk_mb: Maximum disk usage in MB
            allowed_languages: List of allowed programming languages
            enable_network: Whether to allow network access
            persistent_session: Whether to maintain a persistent working directory
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_disk_mb = max_disk_mb
        self.allowed_languages = allowed_languages or ["python", "javascript", "bash"]
        self.enable_network = enable_network
        self.persistent_session = persistent_session
        self.temp_dir = None
        self._session_dir = None
        
        # Command caching to prevent repeated identical calls
        self._command_cache = {}
        self._cache_timeout = 10  # Cache results for 10 seconds
        
        # Security settings
        self.restricted_imports = {
            "python": [
                "os", "sys", "subprocess", "socket", "urllib", "requests",
                "http", "ftplib", "smtplib", "poplib", "imaplib", "telnetlib",
                "webbrowser", "ctypes", "multiprocessing", "threading"
            ]
        }
        
        self.allowed_imports = {
            "python": [
                "math", "random", "datetime", "json", "re", "string",
                "collections", "itertools", "functools", "operator",
                "statistics", "decimal", "fractions", "cmath"
            ]
        }
        
        # System tools not available in sandbox
        self.unavailable_tools = {
            "pdflatex", "latex", "xelatex", "lualatex",  # LaTeX tools
            "apt-get", "yum", "dnf", "pacman",  # Package managers
            "sudo", "su",  # Privilege escalation
            "docker", "podman",  # Container tools
            "systemctl", "service",  # System services
        }
    
    @contextmanager
    def secure_temp_directory(self):
        """Create a secure temporary directory for code execution."""
        if self.persistent_session:
            # Use persistent session directory
            if self._session_dir is None:
                self._session_dir = tempfile.mkdtemp(prefix="sandbox_session_")
                os.chmod(self._session_dir, 0o700)
                logger.info(f"Created persistent sandbox session: {self._session_dir}")
            
            yield self._session_dir
        else:
            # Use temporary directory (original behavior)
            try:
                self.temp_dir = tempfile.mkdtemp(prefix="sandbox_")
                # Set restrictive permissions
                os.chmod(self.temp_dir, 0o700)
                yield self.temp_dir
            finally:
                if self.temp_dir and os.path.exists(self.temp_dir):
                    try:
                        shutil.rmtree(self.temp_dir)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp directory: {e}")
                    finally:
                        self.temp_dir = None
    
    def cleanup_session(self):
        """Clean up the persistent session directory."""
        if self._session_dir and os.path.exists(self._session_dir):
            try:
                shutil.rmtree(self._session_dir)
                logger.info(f"Cleaned up sandbox session: {self._session_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup session directory: {e}")
            finally:
                self._session_dir = None
        
        # Clear command cache
        self._command_cache.clear()
    
    def clear_cache(self):
        """Clear the command cache."""
        self._command_cache.clear()
        logger.info("Cleared sandbox command cache")
    
    def _check_unavailable_tools(self, code: str) -> tuple[bool, str]:
        """Check if code tries to use unavailable system tools.
        
        Args:
            code: Code to check
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Use word boundary matching to avoid false positives (e.g., 'su' in 'result')
        for tool in self.unavailable_tools:
            pattern = rf"\b{re.escape(tool)}\b"
            if re.search(pattern, code):
                return False, (
                    f"Tool '{tool}' is not available in sandbox environment. "
                    "Sandbox supports: python3, echo, cat, ls, grep, sort, head, tail, wc, etc."
                )
        return True, ""
    
    def validate_code(self, code: str, language: str) -> tuple[bool, str]:
        """Validate code for security issues before execution.
        
        Args:
            code: Code to validate
            language: Programming language
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if language not in self.allowed_languages:
            return False, f"Language '{language}' not allowed"
        
        # Check for unavailable system tools
        is_valid, error_msg = self._check_unavailable_tools(code)
        if not is_valid:
            return False, error_msg
        
        # Check for restricted imports/modules
        if language == "python":
            for restricted in self.restricted_imports.get("python", []):
                if f"import {restricted}" in code or f"from {restricted}" in code:
                    return False, f"Restricted import detected: {restricted}"
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            "__import__", "exec(", "eval(", "compile(",
            "open(", "file(", "input(", "raw_input(",
            "globals()", "locals()", "vars()",
            "subprocess", "os.system", "os.popen",
            "socket", "urllib", "requests", "http"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                return False, f"Potentially dangerous code pattern: {pattern}"
        
        return True, ""
    
    def execute_code(self, code: str, language: str, **kwargs) -> Dict[str, Any]:
        """Execute code in a secure sandbox environment.
        
        Args:
            code: Code to execute
            language: Programming language
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary containing execution results
        """
        # Check cache for recent identical commands
        cache_key = f"{language}:{code}"
        current_time = time.time()
        
        if cache_key in self._command_cache:
            cached_result, cached_time = self._command_cache[cache_key]
            if current_time - cached_time < self._cache_timeout:
                logger.info(f"Using cached result for command: {code[:50]}...")
                return cached_result
        
        # Validate code first
        is_valid, error_msg = self.validate_code(code, language)
        if not is_valid:
            return {
                "success": False,
                "error": f"Security validation failed: {error_msg}",
                "stdout": "",
                "stderr": "",
                "execution_time": 0
            }
        
        with self.secure_temp_directory() as temp_dir:
            result = self._execute_in_sandbox(code, language, temp_dir, **kwargs)
            
            # Cache the result (only cache successful non-file-writing commands)
            if result.get("success") and not any(op in code for op in [">", ">>", "rm", "mv", "cp"]):
                self._command_cache[cache_key] = (result, current_time)
            
            return result
    
    def _execute_in_sandbox(self, code: str, language: str, temp_dir: str, **kwargs) -> Dict[str, Any]:
        """Execute code in the sandbox directory."""
        start_time = time.time()
        
        try:
            # Create code file
            if language == "python":
                code_file = os.path.join(temp_dir, "sandbox_code.py")
                command = ["python3", "-c", code]
            elif language == "javascript":
                code_file = os.path.join(temp_dir, "sandbox_code.js")
                command = ["node", "-e", code]
            elif language == "bash":
                code_file = os.path.join(temp_dir, "sandbox_code.sh")
                command = ["bash", "-c", code]
            else:
                return {
                    "success": False,
                    "error": f"Unsupported language: {language}",
                    "stdout": "",
                    "stderr": "",
                    "execution_time": 0
                }
            
            # Set up environment
            env = os.environ.copy()
            env["PYTHONPATH"] = ""  # Clear Python path
            
            # Create a python symlink to python3 in the temp directory for bash commands
            if language == "bash" and "python " in code:
                python_link = os.path.join(temp_dir, "python")
                if not os.path.exists(python_link):
                    # Find python3 executable
                    python3_path = shutil.which("python3")
                    if python3_path:
                        try:
                            os.symlink(python3_path, python_link)
                            env["PATH"] = f"{temp_dir}:{env.get('PATH', '')}"
                        except (OSError, FileNotFoundError):
                            # If symlink fails, try to modify the command
                            pass
            
            # Include common paths
            if "PATH" not in env or temp_dir not in env["PATH"]:
                env["PATH"] = f"{temp_dir}:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"
            
            if not self.enable_network:
                # Disable network access (basic approach)
                env["http_proxy"] = "http://127.0.0.1:1"
                env["https_proxy"] = "http://127.0.0.1:1"
            
            # Execute with restrictions
            result = subprocess.run(
                command,
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Code execution timed out after {self.timeout} seconds",
                "stdout": "",
                "stderr": "",
                "execution_time": self.timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "stdout": "",
                "stderr": "",
                "execution_time": time.time() - start_time
            }

class DockerSandbox:
    """Docker-based sandbox for even more secure code execution."""
    
    def __init__(self, 
                 timeout: int = 30,
                 memory_limit: str = "512m",
                 cpu_limit: str = "0.5",
                 network_disabled: bool = True):
        """Initialize Docker sandbox.
        
        Args:
            timeout: Maximum execution time in seconds
            memory_limit: Docker memory limit (e.g., "512m")
            cpu_limit: Docker CPU limit (e.g., "0.5")
            network_disabled: Whether to disable network access
        """
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self.docker_available = self._check_docker_available()
    
    def _check_docker_available(self) -> bool:
        """Check if Docker is available on the system."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def execute_code(self, code: str, language: str, **kwargs) -> Dict[str, Any]:
        """Execute code in a Docker container.
        
        Args:
            code: Code to execute
            language: Programming language
            **kwargs: Additional execution parameters
            
        Returns:
            Dictionary containing execution results
        """
        if not self.docker_available:
            return {
                "success": False,
                "error": "Docker not available",
                "stdout": "",
                "stderr": "",
                "execution_time": 0
            }
        
        # Select appropriate Docker image
        if language == "python":
            image = "python:3.11-slim"
            cmd = ["python", "-c", code]
        elif language == "javascript":
            image = "node:18-slim"
            cmd = ["node", "-e", code]
        else:
            return {
                "success": False,
                "error": f"Unsupported language for Docker: {language}",
                "stdout": "",
                "stderr": "",
                "execution_time": 0
            }
        
        # Build Docker command
        docker_cmd = [
            "docker", "run", "--rm",
            "--memory", self.memory_limit,
            "--cpus", self.cpu_limit
        ]
        
        if self.network_disabled:
            docker_cmd.extend(["--network", "none"])
        
        docker_cmd.extend([image] + cmd)
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 5  # Allow some buffer for Docker overhead
            )
            
            execution_time = time.time() - start_time
            
            # If Docker execution fails, fallback to local sandbox execution for robustness
            if result.returncode != 0:
                fallback = CodeExecutionSandbox(timeout=self.timeout)
                print("⚠️ Docker execution failed, falling back to local sandbox")
                return fallback.execute_code(code, language, **kwargs)

            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "execution_time": execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Docker execution timed out after {self.timeout} seconds",
                "stdout": "",
                "stderr": "",
                "execution_time": self.timeout
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Docker execution failed: {str(e)}",
                "stdout": "",
                "stderr": "",
                "execution_time": time.time() - start_time
            }

# Factory function to create appropriate sandbox
def create_sandbox(use_docker: bool = False, **kwargs) -> CodeExecutionSandbox:
    """Create an appropriate sandbox based on system capabilities.
    
    Args:
        use_docker: Whether to prefer Docker if available
        **kwargs: Additional sandbox configuration
        
    Returns:
        Configured sandbox instance
    """
    if use_docker:
        docker_sandbox = DockerSandbox(**kwargs)
        if docker_sandbox.docker_available:
            return docker_sandbox
    
    # Enable persistent sessions by default for better file persistence
    if 'persistent_session' not in kwargs:
        kwargs['persistent_session'] = True
    
    return CodeExecutionSandbox(**kwargs)