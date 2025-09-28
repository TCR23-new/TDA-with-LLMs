"""
TDA Agent v2 - Multi-Agent Topological Data Analysis System

This module implements a multi-agent system for topological data analysis using LangGraph.
Combines computational agents for data processing with research agents for interpretation.
"""

import json
import numpy as np
from typing import TypedDict, Annotated
import operator

# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# Local imports
from LLM_Tools import Tool1, Tool2, Tool3, Tool4, Tool5
from PromptExamples import (
    REP_EX_PROMPT, 
    REP_FILE_EX_PROMPT, 
    MULTI_AGENT_EX_PROMPT, 
    SPK_TDA_PROMPT, 
    REACT_PROMPT
)
from TopoPaperRAG import initialize_rag_chain


class AgentState(TypedDict):
    """
    State definition for the multi-agent workflow.
    
    Attributes:
        input: The user's input query or request
    """
    input: str


def create_toolkit() -> list[Tool]:
    """
    Create the toolkit of available tools for the TDA agent.
    
    Returns:
        List of LangChain Tool objects for data analysis
    """
    tools = [
        Tool(
            name='SC_loader',
            func=Tool1,
            description="Load and format spiking data from pickle files"
        ),
        Tool(
            name='TDA_Calculator',
            func=Tool2,
            description="Compute topological features using persistent homology"
        ),
        Tool(
            name='Rep_Tool',
            func=Tool3,
            description="Generate neural network representations from images"
        ),
        Tool(
            name='File_Writer',
            func=Tool4,
            description="Write data to text files for storage"
        ),
        Tool(
            name='TopoTxtTool',
            func=Tool5,
            description="Compute topology of data stored in text files"
        )
    ]
    return tools


def initialize_state_graph(calc_agent: AgentExecutor, research_agent) -> StateGraph:
    """
    Initialize the multi-agent state graph workflow.
    
    Args:
        calc_agent: Computational agent for data processing
        research_agent: Research agent for topological interpretation
        
    Returns:
        Compiled StateGraph workflow
    """
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("computationalist", calc_agent)
    workflow.add_node("researcher", research_agent)
    
    # Define workflow structure
    workflow.set_entry_point("computationalist")
    workflow.add_edge("computationalist", "researcher")
    workflow.add_edge("researcher", END)
    
    return workflow.compile()


def main():
    """
    Main execution function for the TDA agent system.
    """
    # Configuration
    knowledge_base_id = ""  # Add your AWS Knowledge Base ID
    model_id = "amazon.nova-micro-v1:0"
    
    # Initialize LLM
    llm = ChatBedrock(
        model=model_id,
        beta_use_converse_api=True
    )
    
    # Create prompt template
    prompt_template = PromptTemplate.from_template(REACT_PROMPT)
    
    # Initialize toolkit and agents
    toolkit = create_toolkit()
    
    # Create computational agent
    agent = create_react_agent(llm=llm, tools=toolkit, prompt=prompt_template)
    calc_agent = AgentExecutor(
        agent=agent,
        tools=toolkit,
        verbose=True,
        handling_parsing_errors=True
    )
    
    # Initialize RAG chain for research agent
    rag_chain = initialize_rag_chain(knowledge_base_id, model_id)
    
    # Create and run state graph
    state_graph = initialize_state_graph(calc_agent, rag_chain)
    
    # Execute with example prompt
    result = state_graph.invoke({
        "input": MULTI_AGENT_EX_PROMPT
    })
    
    return result


if __name__ == "__main__":
    result = main()
    print("Analysis complete:", result)
