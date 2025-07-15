"""
MCP Client for connecting to the math tools server
"""

import json
import logging
import asyncio
import threading
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Global singleton instance and lock
_mcp_client_instance = None
_mcp_client_lock = threading.Lock()

class SyncMCPClient:
    """Synchronous wrapper for MCPClient."""
    
    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.available_tools = {}
        self.is_connected = False
        self._mcp_module = None
        self._loop = None
        
    def connect(self) -> bool:
        """Connect to the MCP server synchronously."""
        try:
            # Import math tools directly to get available tools
            from mcp_server.math_tools import mcp
            self._mcp_module = mcp
            
            # Use asyncio.run() safely - check if we're already in an event loop
            try:
                # Try to get current event loop
                current_loop = asyncio.get_running_loop()
                # If we're in a running loop, use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    self._mcp_module.list_tools(), current_loop
                )
                tools = future.result(timeout=10)
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                tools = asyncio.run(self._mcp_module.list_tools())
            
            self.available_tools = {tool.name: tool for tool in tools}
            self.is_connected = True
            logger.info(f"Connected to MCP server with {len(self.available_tools)} tools")
            return True
        except Exception as e:
            logger.error(f"Failed to connect synchronously: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the MCP server."""
        self.is_connected = False
        self.available_tools = {}
        self._mcp_module = None
        if self._loop and not self._loop.is_running():
            self._loop.close()
        self._loop = None
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool synchronously."""
        if not self.is_connected or not self._mcp_module:
            logger.error("Not connected to MCP server")
            return None
            
        try:
            # Use asyncio.run() safely - check if we're already in an event loop
            try:
                # Try to get current event loop
                current_loop = asyncio.get_running_loop()
                # If we're in a running loop, use run_coroutine_threadsafe
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    self._mcp_module.call_tool(tool_name, arguments), current_loop
                )
                result = future.result(timeout=30)
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                result = asyncio.run(self._mcp_module.call_tool(tool_name, arguments))
            
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error calling tool synchronously: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_available_tools(self) -> List[str]:
        """Get available tools synchronously."""
        return list(self.available_tools.keys())
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool schema synchronously."""
        if tool_name not in self.available_tools:
            return None
            
        tool = self.available_tools[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": getattr(tool, 'input_schema', {})
        }


def get_mcp_client(server_command: Optional[List[str]] = None) -> Optional[SyncMCPClient]:
    """Get singleton MCP client instance."""
    global _mcp_client_instance
    
    with _mcp_client_lock:
        if _mcp_client_instance is None:
            if server_command is None:
                import os
                server_command = ['python', os.path.join(os.path.dirname(__file__), 'math_tools.py')]
            
            _mcp_client_instance = SyncMCPClient(server_command)
            if not _mcp_client_instance.connect():
                _mcp_client_instance = None
                return None
        
        return _mcp_client_instance


def disconnect_mcp_client():
    """Disconnect and cleanup singleton MCP client."""
    global _mcp_client_instance
    
    with _mcp_client_lock:
        if _mcp_client_instance:
            _mcp_client_instance.disconnect()
            _mcp_client_instance = None