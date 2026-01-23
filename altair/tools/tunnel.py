"""Ngrok tunnel management tools for Altair.

Allows Altair to expose local dev servers via ngrok for remote access.
"""

import asyncio
import logging
import subprocess
import json
import httpx
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from shared.llm.types import ToolParameter
from shared.tools.interface import Tool, ToolContext

logger = logging.getLogger(__name__)


@dataclass
class TunnelInfo:
    """Information about an active ngrok tunnel."""

    port: int
    public_url: str
    process: subprocess.Popen
    started_at: datetime = field(default_factory=datetime.now)
    protocol: str = "http"
    name: Optional[str] = None  # Optional friendly name

    @property
    def age_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def is_alive(self) -> bool:
        return self.process.poll() is None


class TunnelManager:
    """
    Manages ngrok tunnel processes.

    Singleton-ish manager that tracks all active tunnels.
    """

    _instance: Optional["TunnelManager"] = None

    def __init__(self):
        self._tunnels: Dict[int, TunnelInfo] = {}  # port -> TunnelInfo
        self._ngrok_api_port = 4040  # Default ngrok API port

    @classmethod
    def get_instance(cls) -> "TunnelManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start_tunnel(
        self,
        port: int,
        protocol: str = "http",
        name: Optional[str] = None,
    ) -> TunnelInfo:
        """
        Start an ngrok tunnel for a given port.

        Args:
            port: Local port to expose
            protocol: Protocol (http, tcp)
            name: Optional friendly name for the tunnel

        Returns:
            TunnelInfo with public URL

        Raises:
            Exception if tunnel creation fails
        """
        # Check if tunnel already exists for this port
        if port in self._tunnels and self._tunnels[port].is_alive:
            return self._tunnels[port]

        # Build ngrok command
        cmd = ["ngrok", protocol, str(port), "--log=stdout", "--log-format=json"]

        if name:
            cmd.extend(["--name", name])

        logger.info(f"Starting ngrok tunnel for port {port}: {' '.join(cmd)}")

        # Start ngrok process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Wait for tunnel to establish and get URL
        public_url = await self._wait_for_tunnel_url(port, process)

        if not public_url:
            process.kill()
            raise Exception(f"Failed to get public URL for tunnel on port {port}")

        tunnel_info = TunnelInfo(
            port=port,
            public_url=public_url,
            process=process,
            protocol=protocol,
            name=name,
        )

        self._tunnels[port] = tunnel_info
        logger.info(f"Tunnel created: {public_url} -> localhost:{port}")

        return tunnel_info

    async def _wait_for_tunnel_url(
        self,
        port: int,
        process: subprocess.Popen,
        timeout: float = 10.0,
    ) -> Optional[str]:
        """
        Wait for ngrok to establish tunnel and return the public URL.

        Uses ngrok's local API to get tunnel info.
        """
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check if process died
            if process.poll() is not None:
                logger.error("ngrok process exited unexpectedly")
                return None

            # Try to get tunnel URL from ngrok API
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://localhost:{self._ngrok_api_port}/api/tunnels",
                        timeout=2.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        tunnels = data.get("tunnels", [])

                        for tunnel in tunnels:
                            # Find tunnel for our port
                            tunnel_config = tunnel.get("config", {})
                            tunnel_addr = tunnel_config.get("addr", "")

                            if str(port) in tunnel_addr:
                                public_url = tunnel.get("public_url", "")
                                if public_url:
                                    # Prefer https URL
                                    if public_url.startswith("https://"):
                                        return public_url
                                    elif "https://" not in str(tunnels):
                                        return public_url

                        # Check again for https variant
                        for tunnel in tunnels:
                            tunnel_config = tunnel.get("config", {})
                            tunnel_addr = tunnel_config.get("addr", "")
                            if str(port) in tunnel_addr:
                                return tunnel.get("public_url", "")

            except Exception as e:
                # ngrok API not ready yet
                logger.debug(f"Waiting for ngrok API: {e}")

            await asyncio.sleep(0.5)

        return None

    def stop_tunnel(self, port: int) -> bool:
        """
        Stop a tunnel for a given port.

        Returns:
            True if tunnel was stopped, False if not found
        """
        if port not in self._tunnels:
            return False

        tunnel = self._tunnels[port]

        if tunnel.is_alive:
            tunnel.process.terminate()
            try:
                tunnel.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel.process.kill()

        del self._tunnels[port]
        logger.info(f"Stopped tunnel for port {port}")
        return True

    def stop_all_tunnels(self) -> int:
        """
        Stop all active tunnels.

        Returns:
            Number of tunnels stopped
        """
        ports = list(self._tunnels.keys())
        count = 0
        for port in ports:
            if self.stop_tunnel(port):
                count += 1
        return count

    def get_tunnel(self, port: int) -> Optional[TunnelInfo]:
        """Get tunnel info for a port."""
        tunnel = self._tunnels.get(port)
        if tunnel and not tunnel.is_alive:
            # Clean up dead tunnel
            del self._tunnels[port]
            return None
        return tunnel

    def list_tunnels(self) -> List[TunnelInfo]:
        """List all active tunnels."""
        # Clean up dead tunnels
        dead_ports = [p for p, t in self._tunnels.items() if not t.is_alive]
        for port in dead_ports:
            del self._tunnels[port]

        return list(self._tunnels.values())


class StartTunnelTool(Tool):
    """Start an ngrok tunnel to expose a local port."""

    @property
    def name(self) -> str:
        return "start_tunnel"

    @property
    def description(self) -> str:
        return (
            "Start an ngrok tunnel to expose a local port to the internet. "
            "Use this when you've started a dev server and want to give the user "
            "a public URL they can access remotely. Returns the public URL."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="port",
                type="integer",
                description="Local port to expose (e.g., 3000, 8000, 5173)",
                required=True,
            ),
            ToolParameter(
                name="protocol",
                type="string",
                description="Protocol: 'http' (default) or 'tcp'",
                required=False,
            ),
            ToolParameter(
                name="name",
                type="string",
                description="Friendly name for this tunnel (e.g., 'frontend', 'api')",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        port: int,
        protocol: str = "http",
        name: Optional[str] = None,
    ) -> str:
        manager = TunnelManager.get_instance()

        try:
            # Check if tunnel already exists
            existing = manager.get_tunnel(port)
            if existing:
                return (
                    f"✅ Tunnel already active for port {port}\n"
                    f"**Public URL:** {existing.public_url}\n"
                    f"Running for: {int(existing.age_seconds)}s"
                )

            tunnel = await manager.start_tunnel(port, protocol, name)

            result_parts = [
                f"✅ Tunnel created for port {port}",
                f"**Public URL:** {tunnel.public_url}",
            ]

            if name:
                result_parts.append(f"Name: {name}")

            result_parts.append(
                f"\nShare this URL to access the local server from anywhere."
            )

            return "\n".join(result_parts)

        except FileNotFoundError:
            return (
                "❌ Error: ngrok not found. "
                "Please install ngrok: https://ngrok.com/download"
            )
        except Exception as e:
            logger.error(f"Failed to start tunnel: {e}", exc_info=True)
            return f"❌ Error starting tunnel: {str(e)}"


class StopTunnelTool(Tool):
    """Stop an ngrok tunnel."""

    @property
    def name(self) -> str:
        return "stop_tunnel"

    @property
    def description(self) -> str:
        return (
            "Stop an ngrok tunnel for a specific port, or stop all tunnels. "
            "Use this when the dev server is no longer needed."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="port",
                type="integer",
                description="Port number of tunnel to stop. If not specified, stops all tunnels.",
                required=False,
            ),
        ]

    async def execute(
        self,
        context: ToolContext,
        port: Optional[int] = None,
    ) -> str:
        manager = TunnelManager.get_instance()

        if port:
            if manager.stop_tunnel(port):
                return f"✅ Stopped tunnel for port {port}"
            else:
                return f"No active tunnel found for port {port}"
        else:
            count = manager.stop_all_tunnels()
            if count > 0:
                return f"✅ Stopped {count} tunnel(s)"
            else:
                return "No active tunnels to stop"


class ListTunnelsTool(Tool):
    """List all active ngrok tunnels."""

    @property
    def name(self) -> str:
        return "list_tunnels"

    @property
    def description(self) -> str:
        return "List all active ngrok tunnels with their public URLs."

    @property
    def parameters(self) -> List[ToolParameter]:
        return []

    async def execute(self, context: ToolContext) -> str:
        manager = TunnelManager.get_instance()
        tunnels = manager.list_tunnels()

        if not tunnels:
            return "No active tunnels."

        lines = [f"**Active Tunnels ({len(tunnels)}):**\n"]

        for tunnel in tunnels:
            name_str = f" ({tunnel.name})" if tunnel.name else ""
            age_mins = int(tunnel.age_seconds / 60)
            lines.append(
                f"- **Port {tunnel.port}**{name_str}: {tunnel.public_url}\n"
                f"  Protocol: {tunnel.protocol} | Running: {age_mins}m"
            )

        return "\n".join(lines)
