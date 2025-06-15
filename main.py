
import os 
from dotenv import load_dotenv
from pprint import pprint
from preprocess import run_preprocess
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage
from langchain.retrievers import ContextualCompressionRetriever
from langgraph.prebuilt import ToolNode
from typing import Literal

load_dotenv()  # take environment variables

class RetrieverInput(BaseModel):
    query: str



@tool(args_schema=RetrieverInput)
def retriever(query:str)-> list:
    """ 
    Function to retreive information about the query regarding Insurance 
    Will retrieve additional information from vector database
    
    Parameters
    ----------
    compress_retriever : ContextualCompressionRetriever
        The retriever used to fetch compressed, relevant context from the database.
    query : str
        The user's query.

    
    Returns
    -------
    list :  
        List of top k retrieved document chunks 
    """
    api_key = os.getenv("OPENAI_API_KEY")
    compress_retriever = run_preprocess("/Users/user/Documents/agent_rag_langgraph/agent_rag_langgraph/data/gels-pdt-gpa-brochure.pdf", api_key)
    result = compress_retriever.invoke(query)
    
    return result






if __name__ == '__main__':
    
    # print(retriever.name)
    # print(retriever.description)
    # print(retriever.args)
    # compress_retriever = run_preprocess("/Users/user/Documents/agent_rag_langgraph/agent_rag_langgraph/data/gels-pdt-gpa-brochure.pdf")
    # result = retriever(compress_retriever, 'why great protector active ?')
    # # pprint(result)
    api_key = os.getenv("OPENAI_API_KEY")
    
    
    tools = [retriever]
    tool_node = ToolNode(tools)

    # # OpenAI LLM model
    model = ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=api_key).bind_tools(tools)
    # res = model.invoke("Hi")
    # pprint(res.content)


    # Function to decide whether to continue or stop the workflow
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        # If the LLM makes a tool call, go to the "tools" node
        if last_message.tool_calls:
            return "tools"
        # Otherwise, finish the workflow
        return END

    # Function that invokes the model
    def call_model(state: MessagesState):
        messages = state['messages']
        response = model.invoke(messages)
        return {"messages": [response]}  # Returns as a list to add to the state

    # Define the workflow with LangGraph
    workflow = StateGraph(MessagesState)

    # Add nodes to the graph
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Connect nodes
    workflow.add_edge(START, "agent")  # Initial entry
    workflow.add_conditional_edges("agent", should_continue)  # Decision after the "agent" node
    workflow.add_edge("tools", "agent")  # Cycle between tools and agent

    # Configure memory to persist the state
    checkpointer = MemorySaver()

    # Compile the graph into a LangChain Runnable application
    app = workflow.compile(checkpointer=checkpointer)

    # Execute the workflow
    final_state = app.invoke(
        {"messages": [HumanMessage(content=""" You are an intelligent assistant. Be friendly and use the tools to help you.. 
                                   The tool contain all the insurance details need to know about. Only use these insurance information.
                                   Do not hallucinate.If you do not know an answer, reply that you do not know.
                                   Question : 
                                   Tell me more about GREAT Protector Active basic plan""")]},
        config={"configurable": {"thread_id": 42}}
    )

    # Show the final response
    print(final_state["messages"][-1].content)
    print("-------------------------------------------------")
    pprint(final_state["messages"])