"""
Network connectivity checker utility for web search functionality.
"""

import requests
import socket
from typing import Dict, List, Optional
from urllib.parse import urlparse

class NetworkConnectivityChecker:
    """Utility class for checking network connectivity to various services."""
    
    # Timeout for connectivity checks (in seconds)
    TIMEOUT = 5
    
    # Test endpoints for different services
    TEST_ENDPOINTS = {
        'anthropic_web_search': [
            'https://api.anthropic.com',
            'https://claude.ai'
        ],
        'gemini_web_search': [
            'https://googleapis.com',
            'https://generativelanguage.googleapis.com',
            'https://google.com'
        ],
        'openai_web_search': [
            'https://api.openai.com',
            'https://openai.com'
        ],
        'general_web': [
            'https://google.com',
            'https://cloudflare.com',
            'https://github.com'
        ]
    }
    
    def __init__(self):
        self._connectivity_cache = {}
    
    def check_dns_resolution(self, hostname: str) -> bool:
        """Check if DNS resolution works for a hostname."""
        try:
            socket.gethostbyname(hostname)
            return True
        except (socket.gaierror, socket.timeout):
            return False
    
    def check_https_connection(self, url: str) -> bool:
        """Check if HTTPS connection works to a URL."""
        try:
            response = requests.get(
                url, 
                timeout=self.TIMEOUT,
                headers={'User-Agent': 'Auto-PSet-Solver/1.0'}
            )
            # Any HTTP response (including 404, 403, etc.) indicates connectivity
            return True
        except (requests.exceptions.RequestException, 
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.SSLError):
            return False
    
    def check_service_connectivity(self, service_name: str) -> Dict[str, bool]:
        """Check connectivity to all endpoints for a specific service."""
        if service_name not in self.TEST_ENDPOINTS:
            raise ValueError(f"Unknown service: {service_name}")
        
        results = {}
        endpoints = self.TEST_ENDPOINTS[service_name]
        
        for endpoint in endpoints:
            hostname = urlparse(endpoint).hostname
            
            # Check DNS first
            dns_ok = self.check_dns_resolution(hostname)
            results[f"{hostname}_dns"] = dns_ok
            
            # Check HTTPS connection if DNS works
            if dns_ok:
                https_ok = self.check_https_connection(endpoint)
                results[f"{hostname}_https"] = https_ok
            else:
                results[f"{hostname}_https"] = False
        
        return results
    
    def is_web_search_available(self, provider_type: str) -> bool:
        """
        Check if web search is likely to work for a specific provider.
        
        Args:
            provider_type: One of 'anthropic', 'gemini', 'openai', 'deepseek'
            
        Returns:
            True if web search is likely to work, False otherwise
        """
        # Use cached result if available and recent
        cache_key = f"{provider_type}_web_search"
        if cache_key in self._connectivity_cache:
            return self._connectivity_cache[cache_key]
        
        # Map provider types to service names
        service_mapping = {
            'anthropic': 'anthropic_web_search',
            'gemini': 'gemini_web_search', 
            'openai': 'openai_web_search',
            'deepseek': 'general_web'  # DeepSeek doesn't have web search yet
        }
        
        service_name = service_mapping.get(provider_type, 'general_web')
        
        try:
            results = self.check_service_connectivity(service_name)
            
            # Consider web search available if at least one endpoint is reachable
            https_results = [v for k, v in results.items() if k.endswith('_https')]
            is_available = any(https_results)
            
            # Cache the result
            self._connectivity_cache[cache_key] = is_available
            
            return is_available
            
        except Exception as e:
            print(f"⚠️ Network connectivity check failed for {provider_type}: {e}")
            # Default to False if connectivity check fails
            return False
    
    def get_connectivity_report(self, provider_type: str) -> str:
        """Get a detailed connectivity report for debugging."""
        service_mapping = {
            'anthropic': 'anthropic_web_search',
            'gemini': 'gemini_web_search',
            'openai': 'openai_web_search', 
            'deepseek': 'general_web'
        }
        
        service_name = service_mapping.get(provider_type, 'general_web')
        
        try:
            results = self.check_service_connectivity(service_name)
            
            report_lines = [f"🔍 Connectivity Report for {provider_type}:"]
            
            for endpoint_test, result in results.items():
                status = "✅" if result else "❌"
                report_lines.append(f"  {status} {endpoint_test}")
            
            # Summary
            https_results = [v for k, v in results.items() if k.endswith('_https')]
            overall_status = "✅ AVAILABLE" if any(https_results) else "❌ UNAVAILABLE"
            report_lines.append(f"  Overall: {overall_status}")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            return f"❌ Failed to generate connectivity report for {provider_type}: {e}"
    
    def clear_cache(self):
        """Clear the connectivity cache to force fresh checks."""
        self._connectivity_cache.clear()

# Global instance for easy access
network_checker = NetworkConnectivityChecker() 