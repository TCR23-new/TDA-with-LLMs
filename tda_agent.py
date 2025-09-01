# TDA Agent - Topological Data Analysis for Spiking Data
# This module creates an AI agent that can load neural spiking data and compute topological features

# LangChain imports for AI agent functionality
from langchain_aws import BedrockLLM
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from langchain_core.tools import Tool,tool
from langchain.agents import create_react_agent, AgentExecutor

# Scientific computing and data processing
import numpy as np
import pickle
from ripser import Rips  # For topological data analysis

# Data Loading Functions
def load_data(filename: str):
    """
    Load spiking data from a pickle file.
    
    Args:
        filename (str): Path to pickle file containing spiking data
        
    Returns:
        numpy.ndarray: Spiking data array with shape (trials, cells)
    """
    # Handle case where filename contains newlines (from agent input)
    if "\n" in filename and '.pickle' in filename:
        filename_list = filename.split('\n')
        filename = [f for f in filename_list if '.pickle' in f][0]
    
    # Load pickled data (expects list with data as first element)
    data = pickle.load(open(filename,'rb'))
    sc_data = data[0]
    return sc_data

def format_spk_data(data):
    """
    Convert spiking data array to formatted string representation.
    
    Args:
        data (numpy.ndarray): Spiking data with shape (trials, cells)
        
    Returns:
        str: Formatted string with trial and cell information
    """
    NumCells = data.shape[1]
    NumTrials = data.shape[0]
    data_str = ""
    
    # Format each trial and cell value into readable string
    for t in range(NumTrials):
        data_str += 'Trial {}:'.format(t)
        for i in range(NumCells):
            data_str += 'Cell {}: {}'.format(i+1, data[t,i])
        data_str += "\n"
    return data_str

def unformat(data_str: str):
    """
    Convert formatted string back to numpy array.
    
    Args:
        data_str (str): Formatted string from format_spk_data()
        
    Returns:
        numpy.ndarray: Reconstructed spiking data array
    """
    # Parse trials from formatted string
    tt1 = [s for s in data_str.split('Trial') if 'Cell' in s]
    NumTrials = len(tt1)
    NumCells = len(tt1[0].split('Cell')) - 1
    
    # Initialize output array
    SC = np.zeros((NumTrials, NumCells))
    
    # Extract numerical values from each trial and cell
    for i in range(NumTrials):
        tmp_vect = tt1[i].split('Cell')
        tmp_vect = tmp_vect[1:]  # Skip first empty element
        for j in range(len(tmp_vect)):
            SC[i,j] = float(tmp_vect[j].split(':')[1])
    return SC

def compute_topo_summary(data):
    """
    Compute topological features using persistent homology.
    
    Args:
        data (numpy.ndarray): Input data for topological analysis
        
    Returns:
        numpy.ndarray: Persistence diagram for 1-dimensional features (loops)
    """
    # Initialize Ripser for persistent homology computation
    rips = Rips(verbose=False)
    
    # Compute persistence diagrams (0-dim: components, 1-dim: loops, etc.)
    diagrams = rips.fit_transform(data)
    
    # Return 1-dimensional features (loops/cycles)
    return diagrams[1]

def topo_format(data):
    """
    Format topological features into readable string.
    
    Args:
        data (numpy.ndarray): Persistence diagram with birth-death times
        
    Returns:
        str: Formatted string showing topological features
    """
    NumFeatures = data.shape[0]
    BirthDeathTimes = data.shape[1]  # Typically 2: birth and death times
    data_str = ""
    
    # Format each topological feature with its birth-death times
    for i in range(NumFeatures):
        data_str += "Feat {}: ".format(i+1)
        for j in range(BirthDeathTimes):
            data_str += "{} ".format(data[i,j])
        data_str += "\n"
    return data_str

# Agent Tools
def Tool1(filename: str) -> str:
    """
    Tool for loading and formatting spiking data.
    
    Args:
        filename (str): Path to pickle file containing spiking data
        
    Returns:
        str: Formatted string representation of spiking data
    """
    return format_spk_data(load_data(filename))

def Tool2(data: str) -> str:
    """
    Tool for computing topological analysis on formatted data.
    
    Args:
        data (str): Formatted spiking data string
        
    Returns:
        str: Formatted topological features with birth-death times
    """
    return topo_format(compute_topo_summary(unformat(data)))

# Generate example data for testing
X = np.random.rand(100, 5)  # 100 trials, 5 cells
data_filename = 'test_data.pickle'
pickle.dump([X], open(data_filename, 'wb'))

# Define agent tools
sc_data_tool = Tool(
    name='SC_loader',
    func=Tool1,
    description="Useful for loading spiking data from pickle files"
)

tda_tool = Tool(
    name='TDA_Calculator', 
    func=Tool2,
    description="Useful for computing the topology of spiking data using persistent homology"
)

tools = [sc_data_tool, tda_tool]



# Agent prompt template for ReAct (Reasoning + Acting) pattern
prompt1 = """
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

Then you should continue with the thought-action-observation cycle until you have enough information to respond to the user's request directly.
When you have the final answer, respond in this format:
'''
Thought: I know the answer
Final Answer: the final answer to the original query
'''

Begin!

Question: {input}
{agent_scratchpad}
"""

# Create prompt template from the ReAct prompt
prompt_template = PromptTemplate.from_template(prompt1)

# Initialize AWS Bedrock LLM
llm = ChatBedrock(
    model="amazon.nova-micro-v1:0",
    beta_use_converse_api=True  # Use the Converse API for chat models
)

# Create ReAct agent with tools and prompt
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

# Create agent executor with error handling
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handling_parsing_errors=True
)


# Example usage
if __name__ == "__main__":
    question_of_interest = """
    I have spiking data saved in:
    test_data.pickle
    Can you load this spiking data and compute its topology?
    """
    
    # Execute the agent with the question
    results = agent_executor.invoke({"input": question_of_interest})
    print(results['output'])
