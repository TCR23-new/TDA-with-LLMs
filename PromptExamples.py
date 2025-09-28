
"""
Prompt Templates and Examples for TDA Agent

This module contains prompt templates and example queries for the topological data analysis agent.
Includes ReAct pattern prompts and various use case examples.
"""

# ReAct Agent Prompt Template
REACT_PROMPT = """
You are a helpful assistant specialized in topological data analysis of neural spiking data.
You have access to these tools:

{tools}

The available tools are: {tool_names}

To use a tool, please use the following format:
'''
Thought: I need to figure out what to do
Action: tool_name
Action Input: the input to the tool
'''

After you use a tool, the observation will be provided to you:
'''
Observation: result of the tool
'''

Then you should continue with the thought-action-observation cycle until you have enough 
information to respond to the user's request directly.

When you have the final answer, respond in this format:
'''
Thought: I know the answer
Final Answer: the final answer to the original query
'''

Begin!

Question: {input}
{agent_scratchpad}
"""

# Example Prompts for Different Use Cases

# Basic spiking data analysis
SPK_TDA_PROMPT = """
I have spiking data saved in:
test_data.pickle
Can you load this spiking data and compute its topology?
"""

# Neural network representation extraction from multiple images
REP_EX_PROMPT = """
Given below is a list of image files:
"image1.png"
"image2.png"
"image3.png"
"image4.png"
"image5.png"
"image6.png"

Can you obtain the neural network representation of all of the images? 
Write the representation to a file immediately after computing the representation 
and then proceed to compute the representation of the next image. 
You do not need to come up with a filename.
"""

# Topology analysis of pre-computed representations
REP_FILE_EX_PROMPT = """
Given below is a text file containing neural network representations:
"nn_representations.txt"

Compute the topology of those representations.
"""

# Multi-agent analysis with topological interpretation
MULTI_AGENT_EX_PROMPT = """
Given below is a text file containing neural network representations:
"nn_representations.txt"

Compute the topology of the representations within the text file. 
Interpret the persistent homology calculations like a topologist.
"""
