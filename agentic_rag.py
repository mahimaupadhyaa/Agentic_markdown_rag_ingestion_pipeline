"""
Agentic RAG implementation using LangGraph.
"""

import os
import time
import uuid
from typing import Dict, List, Literal, Any, Optional, Union

# Import configuration
from config import config

# Import LangGraph components
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

# Import LangChain components
from langchain_core.messages import convert_to_messages, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

# Import Milvus store
from qdrant_store import QdrantStore as MilvusStore  # drop-in replacement for MilvusStore


def generate_thread_id() -> str:
    timestamp = str(int(time.time() * 1000))
    random_uuid = str(uuid.uuid4()).replace('-', '')
    return f"{timestamp}_{random_uuid}"


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class AgenticRAG:

    def __init__(
            self,
            vector_stores: Optional[List[Dict[str, Any]]] = None,
            vector_store: Optional[MilvusStore] = None,
            model_name: str = None,
            api_key: str = None,
            temperature: float = 0,
            thread_id: Optional[str] = None,
            checkpointer: Optional[InMemorySaver] = None,
        ):

        self.model_name = model_name or config.get("model", "text_generation", default="gpt-4.1-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.checkpointer = checkpointer or InMemorySaver()

        if thread_id is None:
            self.thread_id = generate_thread_id()
        else:
            self.thread_id = thread_id

        self.vector_stores = []
        self.retriever_tools = []

        if vector_stores:

            for vs_config in vector_stores:
                if not isinstance(vs_config, dict):
                    raise ValueError("Each vector store configuration must be a dictionary")

                required_keys = ['store', 'name', 'description']
                missing_keys = [key for key in required_keys if key not in vs_config]
                if missing_keys:
                    raise ValueError(f"Vector store configuration missing required keys: {missing_keys}")

                store = vs_config['store']
                name = vs_config['name']
                description = vs_config['description']
                k = vs_config.get('k', config.get("retrieval", "k", default=2))
                ranker_weights = vs_config.get('ranker_weights', config.get("retrieval", "weights", default=[0.6, 0.4]))

                retriever = store.as_retriever(k=k, ranker_weights=ranker_weights)

                retriever_tool = create_retriever_tool(retriever, name, description)

                self.vector_stores.append({
                    'store': store,
                    'name': name,
                    'description': description,
                    'retriever': retriever,
                    'tool': retriever_tool,
                    'k': k,
                    'ranker_weights': ranker_weights
                })
                self.retriever_tools.append(retriever_tool)

        elif vector_store:

            retriever = vector_store.as_retriever(
                k=config.get("retrieval", "k", default=2),
                ranker_weights=config.get("retrieval", "weights", default=[0.6, 0.4])
            )

            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and retrieve information from the document collection."
            )

            self.vector_stores.append({
                'store': vector_store,
                'name': 'retrieve_documents',
                'description': 'Search and retrieve information from the document collection.',
                'retriever': retriever,
                'tool': retriever_tool,
                'k': config.get("retrieval", "k", default=2),
                'ranker_weights': config.get("retrieval", "weights", default=[0.6, 0.4])
            })
            self.retriever_tools.append(retriever_tool)

        else:

            default_store = MilvusStore()
            retriever = default_store.as_retriever(
                k=config.get("retrieval", "k", default=2),
                ranker_weights=config.get("retrieval", "weights", default=[0.6, 0.4])
            )

            retriever_tool = create_retriever_tool(
                retriever,
                "retrieve_documents",
                "Search and retrieve information from the document collection."
            )

            self.vector_stores.append({
                'store': default_store,
                'name': 'retrieve_documents',
                'description': 'Search and retrieve information from the document collection.',
                'retriever': retriever,
                'tool': retriever_tool,
                'k': config.get("retrieval", "k", default=2),
                'ranker_weights': config.get("retrieval", "weights", default=[0.6, 0.4])
            })
            self.retriever_tools.append(retriever_tool)

        self.vector_store = self.vector_stores[0]['store'] if self.vector_stores else None
        self.retriever = self.vector_stores[0]['retriever'] if self.vector_stores else None
        self.retriever_tool = self.retriever_tools[0] if self.retriever_tools else None

        self.response_model = init_chat_model(self.model_name, temperature=self.temperature)
        self.grader_model = init_chat_model(self.model_name, temperature=0)

        self.graph = self._build_graph()

        print(f"AgenticRAG initialized with model: {self.model_name} and {len(self.vector_stores)} vector store(s)")

    def add_vector_store(self, store: MilvusStore, name: str, description: str,
                        k: Optional[int] = None, ranker_weights: Optional[List[float]] = None) -> None:

        k = k or config.get("retrieval", "k", default=2)
        ranker_weights = ranker_weights or config.get("retrieval", "weights", default=[0.6, 0.4])

        retriever = store.as_retriever(k=k, ranker_weights=ranker_weights)
        retriever_tool = create_retriever_tool(retriever, name, description)

        vs_config = {
            'store': store,
            'name': name,
            'description': description,
            'retriever': retriever,
            'tool': retriever_tool,
            'k': k,
            'ranker_weights': ranker_weights
        }

        self.vector_stores.append(vs_config)
        self.retriever_tools.append(retriever_tool)

        self.graph = self._build_graph()

        print(f"Added vector store '{name}' to AgenticRAG")

    def remove_vector_store(self, name: str) -> bool:

        for i, vs_config in enumerate(self.vector_stores):
            if vs_config['name'] == name:

                removed_config = self.vector_stores.pop(i)
                self.retriever_tools.pop(i)

                if self.vector_store == removed_config['store']:
                    self.vector_store = self.vector_stores[0]['store'] if self.vector_stores else None
                    self.retriever = self.vector_stores[0]['retriever'] if self.vector_stores else None
                    self.retriever_tool = self.retriever_tools[0] if self.retriever_tools else None

                self.graph = self._build_graph()

                print(f"Removed vector store '{name}' from AgenticRAG")
                return True

        print(f"Vector store '{name}' not found")
        return False

    def get_vector_store_info(self) -> List[Dict[str, Any]]:

        return [{
            'name': vs['name'],
            'description': vs['description'],
            'k': vs['k'],
            'ranker_weights': vs['ranker_weights']
        } for vs in self.vector_stores]

    def _route_tools(self, state: MessagesState) -> str:

        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")

        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:

            tool_name = ai_message.tool_calls[0]["name"]

            valid_tool_names = [vs['name'] for vs in self.vector_stores]
            if tool_name in valid_tool_names:
                return tool_name
            else:
                print(f"Unknown tool name: {tool_name}. Available tools: {valid_tool_names}")
                return END

        return END

    def _build_graph(self) -> StateGraph:

        workflow = StateGraph(MessagesState)

        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond)

        retriever_node_names = []
        for vs_config in self.vector_stores:
            node_name = vs_config['name']
            retriever_node_names.append(node_name)
            workflow.add_node(node_name, ToolNode([vs_config['tool']]))

        workflow.add_node("rewrite_question", self._rewrite_question)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "generate_query_or_respond")

        tools_mapping = {}
        for vs_config in self.vector_stores:
            tool_name = vs_config['name']
            tools_mapping[tool_name] = tool_name
        tools_mapping[END] = END

        workflow.add_conditional_edges(
            "generate_query_or_respond",
            self._route_tools,
            tools_mapping,
        )

        for node_name in retriever_node_names:
            workflow.add_conditional_edges(
                node_name,
                self._grade_documents,
            )

        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")

        graph = workflow.compile(checkpointer=self.checkpointer)

        output_file = "graph.png"
        graph.get_graph().draw_mermaid_png(output_file_path=output_file)

        return graph

    def _generate_query_or_respond(self, state: MessagesState) -> Dict:

        print("Generating query or response")

        response = (
            self.response_model
            .bind_tools(self.retriever_tools)
            .invoke(state["messages"])
        )
        return {"messages": [response]}

    def _grade_documents(
        self,
        state: MessagesState
    ) -> Literal["generate_answer", "rewrite_question"]:

        print("Grading retrieved documents")

        question = state["messages"][0].content
        context = state["messages"][-1].content

        grade_prompt = (
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "Here is the retrieved document:\n\n{context}\n\n"
            "Here is the user question: {question}\n"
            "Give a binary score 'yes' or 'no'."
        )

        prompt = grade_prompt.format(question=question, context=context)

        response = (
            self.grader_model
            .with_structured_output(GradeDocuments)
            .invoke([{"role": "user", "content": prompt}])
        )

        score = response.binary_score

        print(f"Document relevance score: {score}")

        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    def _rewrite_question(self, state: MessagesState) -> Dict:

        print("Rewriting question")

        last_human_message = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                last_human_message = message
                break

        question = last_human_message.content

        rewrite_prompt = (
            "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
            "Here is the initial question:\n ------- \n"
            "{question}\n ------- \n"
            "Formulate an improved question:"
        )

        prompt = rewrite_prompt.format(question=question)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])

        print(f"Original question: {question}")
        print(f"Rewritten question: {response.content}")

        return {"messages": [{"role": "user", "content": response.content}]}

    def _generate_answer(self, state: MessagesState) -> Dict:

        print("Generating answer")

        question = state["messages"][0].content
        context = state["messages"][-1].content

        generate_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question}\n"
            "Context: {context}"
        )

        prompt = generate_prompt.format(question=question, context=context)
        response = self.response_model.invoke([{"role": "user", "content": prompt}])

        return {"messages": [response]}

    def update_thread_id(self, new_thread_id: Optional[str] = None) -> str:

        if new_thread_id is None:
            self.thread_id = generate_thread_id()
        else:
            self.thread_id = new_thread_id

        print(f"Thread ID updated to: {self.thread_id}")
        return self.thread_id

    def get_config(self) -> Dict[str, Any]:

        return {"configurable": {"thread_id": self.thread_id}}

    def run(self, query: str) -> str:

        print(f"Running agentic RAG with query: {query}")

        message = {"messages": [{"role": "user", "content": query}]}

        config = self.get_config()
        result = self.graph.invoke(message, config)

        final_message = result["messages"][-1]
        response = final_message.content

        print("Agentic RAG execution completed")

        return response
