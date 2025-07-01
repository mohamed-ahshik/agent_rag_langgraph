import asyncio
import glob
import os
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from preprocess import (
    create_vector_store,
    preprocess_document_to_markdown,
    run_preprocess,
)

app = FastAPI()


load_dotenv()  # take environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Allow requests from any frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the templates directory
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_file(file: UploadFile | None = File(None)):
    if not file:
        return {"message": "No upload file sent"}
    else:
        os.makedirs("uploads", exist_ok=True)  # ensure the directory exists
        content = await file.read()
        with open(f"uploads/{file.filename}", "wb") as out_file:
            out_file.write(content)
        return {"filename": file.filename}


pdf_files = []
for name in glob.glob("uploads/*.pdf"):
    pdf_files.append(name)
    print(name)

# markdown_chunks = []
# for file in pdf_files:
#     preprocess_document_to_markdown()


compress_retriever = run_preprocess(
    pdf_files[-1],
    api_key,
)


compress_retriever_tool = create_retriever_tool(
    compress_retriever,
    "uploaded_insurance_document",
    "Details about the uploaded insurance document.",
)
tools = [compress_retriever_tool]


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Edges
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-turbo", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


# Nodes
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]

    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-3.5-turbo")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n
        Look at the input and try to reason about the underlying semantic intent / meaning. Be specific and add additional details if needed \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-4-turbo", streaming=True)
    # model = init_chat_model(model="openai:gpt-4-turbo", tags=['joke_3'], temperature=0)
    response = model.invoke(msg)
    return {"messages": [response]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    # llm = init_chat_model(model="openai:gpt-4.1", tags=['joke_output'], temperature=0)
    llm = ChatOpenAI(model_name="gpt-4.1", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


class Question(BaseModel):
    question: str


@app.post("/chat")
async def main(question: Question):
    vector_db = create_vector_store(api_key)

    pdf_files = []
    for name in glob.glob("uploads/*.pdf"):
        pdf_files.append(name)
        print(name)

    markdown_chunks = []
    for file in pdf_files:
        chunks = preprocess_document_to_markdown(file)
        markdown_chunks.append(chunks)

    for doc in markdown_chunks:
        vector_db.add_documents(documents=doc)

    retriever = vector_db.as_retriever(
        search_type="similarity",
    )

    # compression_retriever = get_compressed_docs(retriever)

    compress_retriever_tool = create_retriever_tool(
        retriever,
        "uploaded_insurance_document",
        "Details about the uploaded insurance document.",
    )
    tools = [compress_retriever_tool]

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", agent)  # agent
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()

    inputs = {
        "messages": [
            ("user", question.question),
        ]
    }
    # async for output, metadata in graph.astream(inputs, stream_mode="messages"):
    #         if output.content:
    #             print(output.content, end="", flush=True)

    async def stream_generator():
        try:
            async for output, metadata in graph.astream(inputs, stream_mode="messages"):
                if (
                    output.content
                    and output.content.strip()
                    and metadata["langgraph_node"] == "generate"
                ):
                    print(f"Yielding: {output.content}")  # Debug
                    yield output.content + " "
                    await asyncio.sleep(0)  # Let event loop continue
                elif (
                    output.content
                    and output.content.strip()
                    and metadata["langgraph_node"] == "agent"
                ):
                    print(f"Yielding: {output.content}")  # Debug
                    yield output.content + " "
                    await asyncio.sleep(0)  # Let event loop continue
                elif (
                    output.content
                    and output.content.strip()
                    and metadata["langgraph_node"] == "retrieve"
                ):
                    print(f"Yielding: {output.content}")  # Debug
                    await asyncio.sleep(0)  # Let event loop continue
                else:
                    print("Skipping empty content")
        except Exception as e:
            print("Stream error:", str(e))
            yield f"Error: {str(e)}\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # pprint(output)
    # for key, value in output.items():
    #     pprint(f"Output from node '{key}':")
    #     pprint("---")
    #     pprint(value, indent=2, width=80, depth=None)
    # pprint("\n---\n")
    # import json
    # def event_generator():
    #     for chunk in graph.stream(inputs):
    #         for key, value in chunk.items():
    #             print(value)
    #             yield json.dumps({key: str(value)}) + "\n"

    # return StreamingResponse(event_generator(), media_type="application/json")


# class RetrieverInput(BaseModel):
#     query: str


# @tool(args_schema=RetrieverInput)
# def retriever(query: str) -> list:
#     """
#     Function to retreive information about the query regarding Insurance
#     Will retrieve additional information from vector database

#     Parameters
#     ----------
#     compress_retriever : ContextualCompressionRetriever
#         The retriever used to fetch compressed, relevant context from the database.
#     query : str
#         The user's query.


#     Returns
#     -------
#     list :
#         List of top k retrieved document chunks
#     """
#     api_key = os.getenv("OPENAI_API_KEY")
#     compress_retriever = run_preprocess(
#         "/Users/user/Documents/agent_rag_langgraph/agent_rag_langgraph/data/gels-pdt-gpa-brochure.pdf",
#         api_key,
#     )
#     result = compress_retriever.invoke(query)

#     return result


if __name__ == "__main__":

    main()
