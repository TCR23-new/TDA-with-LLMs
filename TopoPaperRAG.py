"""
Topological Data Analysis Paper RAG System

This module implements a Retrieval-Augmented Generation (RAG) system for topological data analysis
research papers. Uses AWS Knowledge Bases for document retrieval and Bedrock for generation.
"""

# Suppress warnings for cleaner output
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

import os
from langchain import hub
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def initialize_rag_chain(knowledge_base_id: str, model_id: str):
    """
    Initialize RAG chain for topological data analysis research papers.
    
    This function sets up a retrieval-augmented generation system using AWS services
    to provide expert knowledge on topological data analysis concepts and methods.
    
    Args:
        knowledge_base_id: AWS Knowledge Base ID containing TDA research papers
        model_id: AWS Bedrock model identifier (e.g., "amazon.nova-micro-v1:0")
        
    Returns:
        Configured RAG chain for question answering
        
    Note:
        The knowledge base contains the following research papers:
        - Bubenik2015: Persistence landscapes
        - Carlsson2009TopoAndData: Topology and data
        - ChazalRinaldo2018: Persistent homology theory
        - CohenSteiner: Stability of persistence diagrams
        - DukeUniNotes: Educational materials on TDA
        - EdelsbrunnerLit: Computational topology literature
        - KajiAhara: Topological data analysis methods
        - PersistenceImagesAdam: Persistence images approach
        - SinghCarlssson: Topological methods for data analysis
        - Sizemore2017: Applications of TDA
        - TopoDenoising: Topological denoising methods
        - Wasserman2018: Review of topological data analysis
    """
    # Configure retriever with AWS Knowledge Bases
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=knowledge_base_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 4  # Retrieve top 4 most relevant documents
            }
        },
    )
    
    # Initialize Bedrock LLM for generation
    llm = ChatBedrock(
        model=model_id,
        beta_use_converse_api=True  # Use Converse API for better chat performance
    )
    
    # Get pre-built retrieval QA prompt from LangChain Hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Create document combination chain
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    # Create final RAG chain
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return rag_chain


# Example usage:
# rag_chain = initialize_rag_chain(knowledge_base_id, model_id)
# response = rag_chain.invoke({"input": "What is persistent homology?"})
# print(response['answer'])
