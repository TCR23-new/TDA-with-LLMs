# TDA Agent - Topological Data Analysis for Spiking Data

An AI agent that performs topological data analysis (TDA) on neural spiking data using LangChain and AWS Bedrock.

## Overview

This project combines topological data analysis with conversational AI to analyze neural spiking data. The agent can load spiking data from pickle files and compute topological features using persistent homology.

## Features

- **Data Loading**: Load spiking data from pickle files
- **Topological Analysis**: Compute persistent homology using Ripser
- **AI Agent**: Natural language interface for data analysis tasks
- **AWS Integration**: Uses Amazon Bedrock for LLM capabilities

## Requirements

```
langchain-aws
langchain-core
langchain
numpy
ripser
pickle
```

## Installation

1. Install required packages:
```bash
pip install langchain-aws langchain-core langchain numpy ripser
```

2. Configure AWS credentials for Bedrock access

## Project Files

- **tda_agent_v2.py**: Updated version of the main TDA agent implementation
- **LLM_Tools.py**: LLM-related tools and utilities
- **PromptExamples.py**: Collection of example prompts for the agent

## Usage

### Basic Example

```python
from tda_agent_v2 import agent_executor

# Ask the agent to analyze your data
question = """
I have spiking data saved in: my_data.pickle
Can you load this spiking data and compute its topology?
"""

results = agent_executor.invoke({"input": question})
print(results['output'])
```

### Data Format

The agent expects pickle files containing spiking data as numpy arrays with shape `(trials, cells)`.

## Tools Available

- **SC_loader**: Loads and formats spiking data from pickle files
- **TDA_Calculator**: Computes topological features using persistent homology

## Output

The agent provides topological summaries showing birth-death times of topological features, helping identify patterns in neural activity.

## License

MIT License
