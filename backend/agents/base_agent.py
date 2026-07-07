# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

import uuid
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Callable

from backend.ollama_service import ask_qwen_coder
from backend.tools.registry import registry
from backend.memory.project_memory import memory_system

logger = logging.getLogger("ASO.BaseAgent")


class BaseAgent:
    def __init__(
        self,
        name: str,
        role: str,
        skills: List[str],
        tool_registry: Optional[Any] = None,
    ):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.skills = skills
        self.status = "idle"
        self.history = []

        # Dependency Injection
        self.tool_registry = tool_registry or registry

        # Metrics tracking
        self.metrics = {
            "task_count": 0,
            "success_count": 0,
            "error_count": 0,
            "total_execution_time_s": 0.0,
            "retry_count": 0,
            "last_active": None,
        }

        # Event support callbacks
        self.event_handlers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {
            "on_task_start": [],
            "on_task_complete": [],
            "on_task_fail": [],
        }

    def register_event_handler(
        self, event_name: str, handler: Callable[[Dict[str, Any]], None]
    ):
        """Register a callback handler for an agent event."""
        if event_name in self.event_handlers:
            self.event_handlers[event_name].append(handler)

    def _trigger_event(self, event_name: str, data: Dict[str, Any]):
        """Trigger callbacks registered for the specified event."""
        for handler in self.event_handlers.get(event_name, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"[{self.name}] Event handler error on {event_name}: {e}")

    def get_system_prompt(self) -> str:
        tools_list = (
            list(self.tool_registry.tools.keys())
            if hasattr(self.tool_registry, "tools")
            else []
        )
        return (
            f"You are {self.name}, an AI Agent in the AGRISENSE Autonomous Software Organization.\n"
            f"Role: {self.role}\n"
            f"Skills: {', '.join(self.skills)}\n"
            "You MUST output your response in valid JSON format ONLY.\n"
            "Format:\n"
            "{\n"
            '  "thought": "your reasoning step-by-step",\n'
            '  "action": "tool_name or task_complete",\n'
            '  "action_input": {"key": "value"},\n'
            '  "response": "message to user or orchestrator"\n'
            "}\n"
            f"Available tools: {tools_list}"
        )

    async def execute_task(self, task: str) -> Dict[str, Any]:
        """
        Executes a task with integrated retries, metrics tracking, event hooks, and logging.
        """
        self.status = "working"
        self.metrics["task_count"] += 1
        self.metrics["last_active"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        start_time = time.time()

        event_data = {
            "agent_name": self.name,
            "task": task,
            "timestamp": self.metrics["last_active"],
        }
        self._trigger_event("on_task_start", event_data)

        logger.info(f"[{self.name}] Started task: {task}")

        loop_context = f"Task: {task}\n"
        final_result = {}
        max_retries = 3

        for step in range(5):
            prompt = loop_context + "\nWhat is your next action? Respond in JSON only."
            attempt = 0
            success = False
            result = {}

            while attempt < max_retries and not success:
                try:
                    result = await ask_qwen_coder(
                        prompt, system=self.get_system_prompt()
                    )
                    if result:  # Check if response parsed correctly
                        success = True
                    else:
                        raise ValueError(
                            "Empty or malformed JSON returned from local assistant."
                        )
                except Exception as e:
                    attempt += 1
                    self.metrics["retry_count"] += 1
                    logger.warning(
                        f"[{self.name}] LLM invocation attempt {attempt} failed: {e}. Retrying..."
                    )
                    await asyncio.sleep(2**attempt)

            if not success:
                logger.error(
                    f"[{self.name}] LLM invocation permanently failed after {max_retries} attempts."
                )
                final_result = {"error": "Failed to query local LLM service."}
                self.status = "failed"
                break

            action = result.get("action", "task_complete")
            action_input = result.get("action_input", {})
            thought = result.get("thought", "")

            logger.info(f"[{self.name}] Step {step + 1}: {thought} -> Action: {action}")

            if action == "task_complete":
                final_result = result
                break

            # Execute tool with fallback/error handling
            try:
                if hasattr(self.tool_registry, "execute_tool"):
                    tool_output = self.tool_registry.execute_tool(action, action_input)
                else:
                    tool_output = "Tool execution registry unavailable."
                loop_context += f"\nAction: {action}\nInput: {action_input}\nResult: {tool_output}\n"
            except Exception as e:
                logger.error(f"[{self.name}] Tool '{action}' execution error: {e}")
                loop_context += (
                    f"\nAction: {action}\nInput: {action_input}\nError: {str(e)}\n"
                )

        # Calculate execution time
        exec_duration = time.time() - start_time
        self.metrics["total_execution_time_s"] += exec_duration

        # Outcome classification
        if "error" in final_result or self.status == "failed":
            self.metrics["error_count"] += 1
            self.status = "failed"
            event_data["error"] = final_result.get("error", "Unknown error")
            self._trigger_event("on_task_fail", event_data)
        else:
            self.metrics["success_count"] += 1
            self.status = "idle"
            event_data["result"] = final_result
            self._trigger_event("on_task_complete", event_data)

        # Log to memory database
        try:
            memory_system.log_task(self.name, task, final_result)
        except Exception as e:
            logger.error(f"[{self.name}] Memory log write failed: {e}")

        self.history.append(
            {"task": task, "result": final_result, "time": exec_duration}
        )
        return final_result

    async def check_health(self) -> Dict[str, Any]:
        """Return dynamic health status and error rates."""
        error_rate = 0.0
        if self.metrics["task_count"] > 0:
            error_rate = self.metrics["error_count"] / self.metrics["task_count"]

        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "healthy": self.status != "failed" and error_rate < 0.5,
            "metrics": self.metrics,
            "error_rate": round(error_rate, 2),
        }

    async def run_diagnostics(self) -> bool:
        """Run self-diagnostics verification check on registry and memory connectivity."""
        try:
            # Check tool registry connectivity
            has_tools = (
                hasattr(self.tool_registry, "tools")
                and len(self.tool_registry.tools) > 0
            )
            # Check memory database connectivity
            history = memory_system.get_history(limit=1)
            # Check prompt generation
            prompt = self.get_system_prompt()

            ok = has_tools and prompt is not None and isinstance(history, list)
            logger.info(
                f"[{self.name}] Self-diagnostics run completed. Status: {'OK' if ok else 'FAIL'}"
            )
            return ok
        except Exception as e:
            logger.error(f"[{self.name}] Self-diagnostics failed with error: {e}")
            return False
