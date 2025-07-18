"""
MCP Server with Calculator and Math Tools for Problem Set Solving

This module provides a comprehensive set of mathematical tools that can be used
by LLMs to solve mathematical problems more accurately.
"""

import math
import statistics
from typing import List, Union, Dict, Any
from mcp.server.fastmcp import FastMCP
import sympy as sp
import numpy as np

# Initialize the MCP server
mcp = FastMCP("Math Tools Server")

# No longer needed - simple arithmetic should be handled by the LLM directly
# @mcp.tool()
# def add(a: float, b: float) -> float:
#     """Add two numbers together."""
#     return a + b
#
# @mcp.tool()
# def subtract(a: float, b: float) -> float:
#     """Subtract one number from another."""
#     return a - b
#
# @mcp.tool()
# def multiply(a: float, b: float) -> float:
#     """Multiply two numbers."""
#     return a * b
#
# @mcp.tool()
# def divide(a: float, b: float) -> float:
#     """Divide one number by another."""
#     if b == 0:
#         raise ValueError("Cannot divide by zero")
#     return a / b

@mcp.tool()
def power(base: float, exponent: float) -> float:
    """Raise a number to a power."""
    return base ** exponent

@mcp.tool()
def square_root(x: float) -> float:
    """Calculate the square root of a number."""
    if x < 0:
        raise ValueError("Cannot take square root of negative number")
    return math.sqrt(x)

@mcp.tool()
def logarithm(x: float, base: float = math.e) -> float:
    """Calculate the logarithm of a number with specified base (default: natural log)."""
    if x <= 0:
        raise ValueError("Cannot take logarithm of non-positive number")
    if base <= 0 or base == 1:
        raise ValueError("Base must be positive and not equal to 1")
    return math.log(x, base)

@mcp.tool()
def sin(x: float) -> float:
    """Calculate the sine of a number (in radians)."""
    return math.sin(x)

@mcp.tool()
def cos(x: float) -> float:
    """Calculate the cosine of a number (in radians)."""
    return math.cos(x)

@mcp.tool()
def tan(x: float) -> float:
    """Calculate the tangent of a number (in radians)."""
    return math.tan(x)

@mcp.tool()
def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return math.factorial(n)

@mcp.tool()
def combination(n: int, r: int) -> int:
    """Calculate nCr (combinations)."""
    if n < 0 or r < 0 or r > n:
        raise ValueError("Invalid values for combination")
    return math.comb(n, r)

@mcp.tool()
def permutation(n: int, r: int) -> int:
    """Calculate nPr (permutations)."""
    if n < 0 or r < 0 or r > n:
        raise ValueError("Invalid values for permutation")
    return math.perm(n, r)

# evaluate_expression removed - simple expressions should be handled by the LLM directly
# Complex expressions should use specific tools like solve_equation, differentiate, integrate, etc.

@mcp.tool()
def solve_equation(equation: str, variable: str = "x") -> List[str]:
    """Solve an algebraic equation for a given variable."""
    try:
        # Parse the equation
        var = sp.Symbol(variable)
        eq = sp.Eq(*[sp.sympify(side) for side in equation.split("=")])
        
        # Solve the equation
        solutions = sp.solve(eq, var)
        
        # Convert solutions to strings for JSON serialization
        return [str(sol) for sol in solutions]
    except Exception as e:
        raise ValueError(f"Could not solve equation: {e}")

@mcp.tool()
def differentiate(expression: str, variable: str = "x") -> str:
    """Calculate the derivative of an expression with respect to a variable."""
    try:
        var = sp.Symbol(variable)
        expr = sp.sympify(expression)
        derivative = sp.diff(expr, var)
        return str(derivative)
    except Exception as e:
        raise ValueError(f"Could not differentiate: {e}")

@mcp.tool()
def integrate(expression: str, variable: str = "x") -> str:
    """Calculate the indefinite integral of an expression with respect to a variable."""
    try:
        var = sp.Symbol(variable)
        expr = sp.sympify(expression)
        integral = sp.integrate(expr, var)
        return str(integral)
    except Exception as e:
        raise ValueError(f"Could not integrate: {e}")

@mcp.tool()
def definite_integral(expression: str, variable: str = "x", lower_limit: float = 0, upper_limit: float = 1) -> float:
    """Calculate the definite integral of an expression over a specified range."""
    try:
        var = sp.Symbol(variable)
        expr = sp.sympify(expression)
        result = sp.integrate(expr, (var, lower_limit, upper_limit))
        return float(result.evalf())
    except Exception as e:
        raise ValueError(f"Could not calculate definite integral: {e}")

@mcp.tool()
def matrix_multiply(matrix1: List[List[float]], matrix2: List[List[float]]) -> List[List[float]]:
    """Multiply two matrices."""
    try:
        m1 = np.array(matrix1)
        m2 = np.array(matrix2)
        result = np.dot(m1, m2)
        return result.tolist()
    except Exception as e:
        raise ValueError(f"Could not multiply matrices: {e}")

@mcp.tool()
def matrix_determinant(matrix: List[List[float]]) -> float:
    """Calculate the determinant of a square matrix."""
    try:
        m = np.array(matrix)
        if m.shape[0] != m.shape[1]:
            raise ValueError("Matrix must be square")
        return float(np.linalg.det(m))
    except Exception as e:
        raise ValueError(f"Could not calculate determinant: {e}")

@mcp.tool()
def matrix_inverse(matrix: List[List[float]]) -> List[List[float]]:
    """Calculate the inverse of a square matrix."""
    try:
        m = np.array(matrix)
        if m.shape[0] != m.shape[1]:
            raise ValueError("Matrix must be square")
        result = np.linalg.inv(m)
        return result.tolist()
    except Exception as e:
        raise ValueError(f"Could not calculate matrix inverse: {e}")

@mcp.tool()
def calculate_mean(numbers: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list")
    return statistics.mean(numbers)

@mcp.tool()
def calculate_median(numbers: List[float]) -> float:
    """Calculate the median of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot calculate median of empty list")
    return statistics.median(numbers)

@mcp.tool()
def calculate_mode(numbers: List[float]) -> float:
    """Calculate the mode of a list of numbers."""
    if not numbers:
        raise ValueError("Cannot calculate mode of empty list")
    try:
        return statistics.mode(numbers)
    except statistics.StatisticsError:
        raise ValueError("No unique mode found")

@mcp.tool()
def calculate_standard_deviation(numbers: List[float]) -> float:
    """Calculate the standard deviation of a list of numbers."""
    if len(numbers) < 2:
        raise ValueError("Need at least 2 numbers for standard deviation")
    return statistics.stdev(numbers)

@mcp.tool()
def calculate_variance(numbers: List[float]) -> float:
    """Calculate the variance of a list of numbers."""
    if len(numbers) < 2:
        raise ValueError("Need at least 2 numbers for variance")
    return statistics.variance(numbers)

@mcp.tool()
def linear_regression(x_values: List[float], y_values: List[float]) -> Dict[str, float]:
    """Perform linear regression on two sets of data points."""
    if len(x_values) != len(y_values):
        raise ValueError("x and y values must have the same length")
    if len(x_values) < 2:
        raise ValueError("Need at least 2 data points")
    
    try:
        # Calculate regression coefficients
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            raise ValueError("Cannot perform regression: all x values are the same")
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = [slope * x + intercept for x in x_values]
        ss_res = sum((y_values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((y_values[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1
        
        return {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared
        }
    except Exception as e:
        raise ValueError(f"Could not perform linear regression: {e}")

@mcp.tool()
def gcd(a: int, b: int) -> int:
    """Calculate the greatest common divisor of two integers."""
    return math.gcd(a, b)

@mcp.tool()
def lcm(a: int, b: int) -> int:
    """Calculate the least common multiple of two integers."""
    return abs(a * b) // math.gcd(a, b)

@mcp.tool()
def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

@mcp.tool()
def prime_factors(n: int) -> List[int]:
    """Find the prime factors of a number."""
    if n <= 1:
        return []
    
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    
    return factors

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()