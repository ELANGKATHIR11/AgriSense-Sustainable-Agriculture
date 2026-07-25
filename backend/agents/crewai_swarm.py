# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

import os
import asyncio
from typing import TypedDict
from pydantic_ai import Agent as PydanticAgent
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Ensure Ollama base URL is configured
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")

# 1. Define State Schema for Persistent Swarm State
class AgentSwarmState(TypedDict):
    query: str
    diagnose_result: str
    market_result: str
    cost_result: str
    final_output: str

# 2. Define PydanticAI Agents
agronomist_agent = PydanticAgent(
    "ollama:qwen2.5:1.5b-instruct",
    system_prompt=(
        "You are a Senior Agronomist. Your goal is to analyze crop disease symptoms and soil parameters "
        "to recommend remedies. You are an expert agronomist with decades of experience diagnosing plant health, "
        "pathology, and crop nutrition issues."
    )
)

market_agent = PydanticAgent(
    "ollama:qwen2.5:1.5b-instruct",
    system_prompt=(
        "You are an Agri Marketplace Analyst. Your goal is to recommend seed varieties, organic pesticides, "
        "and fertilizers based on treatments. You are a market coordinator specialized in agricultural inputs, "
        "cost forecasting, and local organic products."
    )
)

cost_agent = PydanticAgent(
    "ollama:qwen2.5:1.5b-instruct",
    system_prompt=(
        "You are an Agricultural Economist. Your goal is to calculate pricing, costs, and ROI of recommended "
        "treatments for the farmer. You are a financial advisor for farms, specializing in input costs, yield economics, "
        "and crop budget management."
    )
)

# 3. Define State Graph Nodes
async def agronomist_node(state: AgentSwarmState) -> dict:
    prompt = (
        f"Evaluate the following agricultural query and context: {state['query']}. "
        "Identify the core issue or disease and draft an agronomic remedy."
    )
    result = await agronomist_agent.run(prompt)
    return {"diagnose_result": result.data}

async def market_node(state: AgentSwarmState) -> dict:
    prompt = (
        "Review the agronomist's diagnostic/remedy report:\n\n"
        f"{state['diagnose_result']}\n\n"
        "Recommend specific agricultural inputs, seed varieties, or organic treatments available in local markets."
    )
    result = await market_agent.run(prompt)
    return {"market_result": result.data}

async def cost_node(state: AgentSwarmState) -> dict:
    prompt = (
        "Examine the remedies and recommended marketplace products:\n\n"
        f"Diagnosis/Remedy:\n{state['diagnose_result']}\n\n"
        f"Market Recommendations:\n{state['market_result']}\n\n"
        "Estimate input cost limits in Indian Rupees (₹) and draft a brief ROI recommendation for the farmer."
    )
    result = await cost_agent.run(prompt)
    return {"cost_result": result.data, "final_output": result.data}

# 4. Compile the LangGraph Swarm
workflow = StateGraph(AgentSwarmState)
workflow.add_node("agronomist", agronomist_node)
workflow.add_node("market", market_node)
workflow.add_node("cost", cost_node)

workflow.add_edge(START, "agronomist")
workflow.add_edge("agronomist", "market")
workflow.add_edge("market", "cost")
workflow.add_edge("cost", END)

# In-memory checkpointer for persistent state
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# 5. Swarm Execution Interface
async def run_agri_crew(task_description: str) -> str:
    """
    Kicks off the sequential multi-agent LangGraph workflow.
    """
    initial_state = {
        "query": task_description,
        "diagnose_result": "",
        "market_result": "",
        "cost_result": "",
        "final_output": ""
    }
    
    config = {"configurable": {"thread_id": "swarm-thread-default"}}
    final_state = await app.ainvoke(initial_state, config=config)
    return final_state["final_output"]
