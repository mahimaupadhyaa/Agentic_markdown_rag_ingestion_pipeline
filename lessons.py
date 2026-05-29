# LangGraph Lessons

from typing import TypedDict, Dict, List, Union, Annotated, Sequence, Any
from langgraph.graph import StateGraph, START, END
import operator
import random

from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage

# Simple Conditional Edge Graph

class Agent1State(TypedDict):
    number1: int
    operation1: str
    number2: int
    number3: int
    operation2: str
    number4: int
    finalNumber1: int
    finalNumber2: int
    
def adder1(state: Agent1State) -> dict[str, Any]:
    return {"finalNumber1": state["number1"] + state["number2"]}

def adder2(state: Agent1State) -> dict[str, Any]:
    return {"finalNumber2": state["number3"] + state["number4"]}

def subtractor1(state: Agent1State) -> dict[str, Any]:
    return {"finalNumber1": state["number1"] - state["number2"]}

def subtractor2(state: Agent1State) -> dict[str, Any]:
    return {"finalNumber2": state["number3"] - state["number4"]}

def router1(state: Agent1State) -> str:
    if state["operation1"] == "+":
        return "adder1"
    elif state["operation1"] == "-":
        return "subtractor1"
    else: 
        return "default"

def router2(state: Agent1State) -> str:
    if state["operation2"] == "+":
        return "adder2"
    elif state["operation2"] == "-":
        return "subtractor2"
    else: 
        return "default"

graph1 = StateGraph(Agent1State)

graph1.add_node("adder1", adder1)
graph1.add_node("adder2", adder2)
graph1.add_node("subtractor1", subtractor1)
graph1.add_node("subtractor2", subtractor2)
graph1.add_node("router1", lambda state: state)
graph1.add_node("router2", lambda state: state)

graph1.add_edge(START, "router1")

graph1.add_conditional_edges(
    "router1",
    router1,
    {
        "adder1": "adder1",
        "subtractor1": "subtractor1",
        "default": "adder1"
    }
)

graph1.add_edge("adder1", "router2")
graph1.add_edge("subtractor1", "router2")

graph1.add_conditional_edges(
    "router2",
    router2,
    {
        "adder2": "adder2",
        "subtractor2": "subtractor2",
        "default": "adder2"
    }
)

graph1.add_edge("adder2", END)
graph1.add_edge("subtractor2", END)

agent1 = graph1.compile()

png_bytes = agent1.get_graph().draw_mermaid_png()

with open("graph1.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph1.png")

agent1_shared_state = Agent1State(
    number1=10,
    operation1="+",
    number2=10,
    number3=11,
    operation2="-",
    number4=23
)

response = agent1.invoke(agent1_shared_state)
print(response)


# Simple Looping Graph1

class Agent2State(TypedDict):
    name: str
    numbers: Annotated[List[int], operator.add]
    counter: int

def greeter(state: Agent2State) -> dict[str, Any]:
    """Greeting Node which greets the person"""
    return { "name": f"Hi there, {state['name']}!", "counter": 0 }

def random_number_adder(state: Agent2State) -> dict[str, Any]:
    """Random Number Adder Node which adds a random number to the list"""
    new_number = random.randint(0,10)
    return { "numbers": [new_number], "counter": state["counter"] + 1 }

def should_continue(state: Agent2State) -> str:
    """Function to determine if the loop should continue"""
    if state["counter"] < 5:
        return "loop"
    else:
        return "exit"

graph2 = StateGraph(Agent2State)

graph2.add_node("greeter", greeter)
graph2.add_node("random_number_adder", random_number_adder)

graph2.add_edge(START, "greeter")
graph2.add_edge("greeter", "random_number_adder")

graph2.add_conditional_edges(
    "random_number_adder",
    should_continue,
    {
        "loop": "random_number_adder",
        "exit": END
    }
)

agent2 = graph2.compile()

png_bytes = agent2.get_graph().draw_mermaid_png()

with open("graph2.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph2.png")

agent2_shared_state = Agent2State(
    name="John",
    numbers=[],
    counter=10
)

response = agent2.invoke(agent2_shared_state)
print(response)


# Simple Looping Graph2

class Agent3State(TypedDict):
    player_name: str
    player_number: int
    guesses: Annotated[List[int], operator.add]
    max_attempts: int
    lower_bound: int
    upper_bound: int

def setup_guesser(state: Agent3State) -> dict[str, Any]:
    """Setup Guesser Node which sets up the guesser"""
    return {"player_number": random.randint(1,20), "guesses": [], "max_attempts": 7, "lower_bound": 1, "upper_bound": 20}

def number_guesser(state: Agent3State) -> dict[str, Any]:
    """Number Guesser Node which guesses a number"""
    guess = random.randint(state["lower_bound"], state["upper_bound"])
    return {"guesses": [guess]}

def evaluate_guess(state: Agent3State) -> dict[str, Any]:
    """Updates bounds based on guess"""
    guess = state["guesses"][-1]
    target = state["player_number"]

    if guess < target:
        print("Need Higher")
        return {"lower_bound": guess + 1}

    elif guess > target:
        print("Need Lower")
        return {"upper_bound": guess - 1}

    else:
        print("Correct Guess!")
        return {}

def should_continue_guessing(state: Agent3State) -> str:
    """Function to determine if the guessing should continue"""
    if len(state["guesses"]) >= state["max_attempts"]:
        return "exit"
    elif state["player_number"] == state["guesses"][-1]:
        return "exit"    
    return "continue"

graph3 = StateGraph(Agent3State)

graph3.add_node("setup_guesser", setup_guesser)
graph3.add_node("number_guesser", number_guesser)
graph3.add_node("evaluate_guess", evaluate_guess)

graph3.add_edge(START, "setup_guesser")
graph3.add_edge("setup_guesser", "number_guesser")
graph3.add_edge("number_guesser", "evaluate_guess")

graph3.add_conditional_edges(
    "evaluate_guess",
    should_continue_guessing,
    {
        "continue": "number_guesser",
        "exit": END,
    },
)

agent3 = graph3.compile()

png_bytes = agent3.get_graph().draw_mermaid_png()

with open("graph3.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph3.png")

agent3_shared_state = Agent3State(
    player_name="John",
)

response = agent3.invoke(agent3_shared_state)
print(response)


# Integrating LLMs into Graph

llm = ChatOllama(model="qwen2.5:latest")

class Agent4State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def processor(state: Agent4State) -> dict[str, Any]:
    return {"messages": [llm.invoke(state["messages"])]}

graph4 = StateGraph(Agent4State)

graph4.add_node("processor", processor)
graph4.add_edge(START, "processor")
graph4.add_edge("processor", END)

agent4 = graph4.compile()

png_bytes = agent4.get_graph().draw_mermaid_png()

with open("graph4.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph4.png")

user_message = input("Enter: ")

while user_message != "exit":
    agent4_shared_state = Agent4State(
        messages=[HumanMessage(content=user_message)]
    )
    response = agent4.invoke(agent4_shared_state)
    print(response)
    user_message = input("Enter: ")


# Integrating LLMs into Graph with Memory

class Agent5State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def processor(state: Agent5State) -> dict[str, Any]:
    """This node will response to your input"""
    return {"messages": [llm.invoke(state["messages"])]}

graph5 = StateGraph(Agent5State)
graph5.add_node("processor", processor)
graph5.add_edge(START, "processor")
graph5.add_edge("processor", END)

agent5 = graph5.compile()

png_bytes = agent5.get_graph().draw_mermaid_png()

with open("graph5.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph5.png")

conversation_history = []

user_message = input("Enter: ")

while user_message != "exit":
    conversation_history.append(HumanMessage(content=user_message))
    agent5_shared_state = Agent5State(
        messages=conversation_history
    )
    response = agent5.invoke(agent5_shared_state)
    print(response)
    user_message = input("Enter: ")
    conversation_history = response["messages"]

with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")

    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")


# Simple REACT Agent

class Agent6State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b:int) -> int:
    """This is an addition function that adds 2 numbers together"""
    return a + b

tools = [add]

llm_with_tools = llm.bind_tools(tools)

def model_call(state: Agent6State) -> dict[str, Any]:
    """This node will use llm to respond to your input"""
    system_prompt = SystemMessage(content="You are a helpful assistant. Please respond to the user's input.")
    response = llm_with_tools.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: Agent6State) -> str:
    """This function will determine if the loop should continue"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph6 = StateGraph(Agent6State)
graph6.add_node("our_agent", model_call)

tool_node = ToolNode(tools)
graph6.add_node("tools", tool_node)

graph6.add_edge(START, "our_agent")

graph6.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

graph6.add_edge("tools", "our_agent")

agent6 = graph6.compile()

png_bytes = agent6.get_graph().draw_mermaid_png()

with open("graph6.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph6.png")

input_messages = {"messages": [("user", "Add 3 + 4.")]}

response = agent6.invoke(input_messages)
print(response)


# Document Drafter Agent

document_content = ""

class Agent7State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_document(modified_content: str) -> str:
    """This function will update the document content based on user input"""
    global document_content
    document_content = modified_content
    return f"Document updated successfully! The current content is:\n{document_content}"

@tool
def save_document(document_name: str) -> str:
    """This function will save the document content to a file

    Args:
        document_name (str): The name of the document to save
    """

    global document_content

    if not document_name.endswith(".txt"):
        document_name += ".txt"
    
    try:
        with open(document_name, "w") as file:
            file.write(document_content)
            return f"Document has been saved successfully to {document_name}"
    except Exception as e:
        return f"Error saving document: {str(e)}"

tools = [update_document, save_document]

llm_with_tools = llm.bind_tools(tools)

def drafter_agent(state: Agent7State) -> dict[str, Any]:
    system_prompt = SystemMessage(content="""You are a helpful assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update the document, use the 'update_document' tool with the complete updated content.
    - If the user wants to save and finish, you need to user the 'save_document' tool with the document name.
    - Make sure to always show the current document state after modifications.

    The current document content is: {document_content}
    """)

    if not state["messages"]:
        print(state["messages"])
        user_input = "I'm ready to help you draft/update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_message = input("\n What would you like to do with this document?")
        user_message = HumanMessage(content=user_message)

    all_messages = [system_prompt] + state["messages"] + [user_message]
    response = llm_with_tools.invoke(all_messages)
    return {"messages": [user_message, response]}

def should_continue(state: Agent7State) -> str:
    messages = state["messages"]

    if not messages:
        return "continue"

    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and "saved successfully" in message.content.lower()):
            return "end"    
    return "continue"

graph7 = StateGraph(Agent7State)

graph7.add_node("drafter_agent", drafter_agent)

tool_node = ToolNode(tools)
graph7.add_node("tools", tool_node)

graph7.add_edge(START, "drafter_agent")
graph7.add_edge("drafter_agent", "tools")

graph7.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "drafter_agent",
        "end": END
    }
)

agent7 = graph7.compile()

png_bytes = agent7.get_graph().draw_mermaid_png()

with open("graph7.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph7.png")

def print_messages(messages):
    if not messages:
        return

    else:
        for message in messages[-3:]:
            if isinstance(message, ToolMessage):
                print(f"\n TOOL RESULT: {message.content}")


def run_drafter_agent():
    print("\n===== DRAFTER ====")

    state = Agent7State()
    
    for step in agent7.stream(state, stream_node="values"):
        if "messaages" in step:
            print_messages(step["messages"])

    print("\n==== THANK YOU! ====")


if __name__ == "__main__":
    run_drafter_agent()