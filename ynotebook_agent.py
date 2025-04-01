from typing_extensions import TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, FunctionMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import format_tool_to_openai_function
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode 
from dotenv import load_dotenv
from jupyter_ydoc.ynotebook import YNotebook
import os
import json


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

# Set up LLM
llm = ChatOpenAI(api_key=api_key, model="gpt-4-turbo", temperature=0)

### === Define State === ###
class State(TypedDict):
    messages: list
    ynotebook: YNotebook

memory = MemorySaver()

# define tools here
@tool("cut_cell_crdt", return_direct=True)
def cut_cell_crdt(id: int, ynotebook: YNotebook) -> str:
    """
    Removes a cell from a YNotebook at the given index and returns the cut cell's content.
    """
    try:
        if 0 <= id < ynotebook.cell_number:
            cut_cell = ynotebook.get_cell(id)
            ynotebook._ycells.pop(id)
            return f"✅ Cut cell {id}: {cut_cell['source']}"
        else:
            return f"❌ Invalid cell index: {id}"
    except Exception as e:
        return f"❌ Error cutting cell: {str(e)}"

@tool("add_cell_crdt", return_direct=True)
def add_cell_crdt(id: int, cell_type: str = "code", ynotebook: YNotebook = None) -> str:
    """
    Adds a new empty cell (code or markdown) at the specified index in a YNotebook.
    """
    try:
        if cell_type not in ("code", "markdown"):
            return f"❌ Invalid cell_type: {cell_type}"

        new_cell_dict = {
            "cell_type": cell_type,
            "metadata": {},
            "source": "",
            "execution_count": None,
            "outputs": [] if cell_type == "code" else None,
        }

        ycell = ynotebook.create_ycell(new_cell_dict)
        index = max(0, min(id, ynotebook.cell_number))
        ynotebook._ycells.insert(index, ycell)
        return f"✅ Added {cell_type} cell at index {index}."

    except Exception as e:
        return f"❌ Error adding cell: {str(e)}"

@tool("write_to_cell_crdt", return_direct=True)
def write_to_cell_crdt(id: int, content: str, ynotebook: YNotebook = None) -> str:
    """
    Overwrites the entire source of a cell at a given index in a YNotebook.
    """
    try:
        if 0 <= id < ynotebook.cell_number:
            cell = ynotebook.get_cell(id)
            cell["source"] = content
            ynotebook.set_cell(id, cell)
            return f"✅ Updated cell {id} with content:\n{content}"
        else:
            return f"❌ Invalid cell index: {id}"
    except Exception as e:
        return f"❌ Error writing to cell: {str(e)}"

@tool("read_cell_crdt", return_direct=True)
def read_cell_crdt(id: int, ynotebook: YNotebook = None) -> str:
    """
    Reads a specific cell from a YNotebook and returns its full content.
    """
    try:
        if 0 <= id < ynotebook.cell_number:
            cell = ynotebook.get_cell(id)
            return json.dumps(cell, indent=2)
        else:
            return f"❌ Invalid cell index: {id}"
    except Exception as e:
        return f"❌ Error reading cell: {str(e)}"



tools = [
    cut_cell_crdt,
    add_cell_crdt,
    write_to_cell_crdt,
    read_cell_crdt,
]
model = llm.bind_tools(tools)
tool_node = ToolNode(tools=tools)


def agent_node(state):
    """Executes notebook editing tasks and delegates decision-making to `should_continue`."""
    messages = state["messages"]
    ynotebook = state["ynotebook"]

    # 🔥 Inject the file_path so the LLM **always knows it**

    # 🚀 Let the LLM decide what needs to be done
    result = model.invoke(messages)


    return {"messages": messages + [result], "ynotebook": ynotebook}  # Append latest LLM response
        



def should_continue(state):
    """Determines if there are pending tool calls."""
    last_message = state["messages"][-1]

    # Check if 'tool_calls' exist in the last message (for bind_tools)
    if "tool_calls" in last_message.additional_kwargs and last_message.additional_kwargs["tool_calls"]:
        return "continue"  # Proceed to tool execution

    return "end"  # No more tool calls, end execution


def call_tool(state):
    """Executes tool calls using ToolNode correctly."""
    messages = state["messages"]
    ynotebook = state["ynotebook"]

    # 🔍 Ensure last message exists
    if not messages:
        print("❌ ERROR: No messages found in state.")
        return {"messages": messages, "ynotebook": ynotebook}

    last_message = messages[-1]
    print(last_message.additional_kwargs["tool_calls"][0]["function"])

    # 🔍 Ensure last message is an AIMessage with tool_calls
    if "tool_calls" not in last_message.additional_kwargs or not last_message.additional_kwargs["tool_calls"]:
        print("❌ ERROR: No tool calls found in last AIMessage.")
        return {"messages": messages, "ynotebook": ynotebook}


    # ✅ Fix: Pass only the messages list
    tool_results = tool_node.invoke({"messages": messages})  # ToolNode expects this format

    # 🔥 Fix: Ensure tool_results is a list of messages
    if isinstance(tool_results, dict):  
        tool_results = tool_results.get("messages", [])

    # ✅ Append results to messages and return updated state
    return {"messages": messages + tool_results, "ynotebook": ynotebook}





# 🔧 Construct the Graph
builder = StateGraph(State)

# nodes: 
builder.add_node("agent", agent_node)
builder.add_node("call_tool", call_tool)

# 1️⃣ Start by parsing the command
builder.add_edge(START, "agent")


# 3️⃣ Decide whether to continue or stop
builder.add_conditional_edges("agent", should_continue, {"continue": "call_tool", "end": END})  # ✅ Proper conditional routing

# 4️⃣ Execute tools when needed
builder.add_edge("call_tool", "agent")  # Return to editor after tool call

# 5️⃣ End condition (already handled in conditional)

graph = builder.compile(checkpointer=memory)

