from typing import Any
from mcp.server.fastmcp import FastMCP
from jupyter_ydoc.ynotebook import YNotebook
from registry import NotebookRegistry
import json
import asyncio

# Initialize the notebook registry
registry = NotebookRegistry()

# Create MCP server
mcp = FastMCP("ynotebook-tools")

@mcp.tool()
async def read_cell(path: str, index: int) -> str:
    """Return content of cell at index in the notebook at given path."""
    try:
        ynotebook = registry.get_or_load(path)
        cell = ynotebook.get_cell(index)
        return json.dumps(cell, indent=2)
    except Exception as e:
        return f"❌ Error reading cell {index} from {path}: {str(e)}"

@mcp.tool()
async def write_to_cell(path: str, index: int, content: str) -> str:
    """Overwrite the source of a cell in the notebook at given path."""
    try:
        ynotebook = registry.get_or_load(path)
        cell = ynotebook.get_cell(index)
        cell["source"] = content
        ynotebook.set_cell(index, cell)
        return f"✅ Updated cell {index} in {path}."
    except Exception as e:
        return f"❌ Error writing to cell {index} in {path}: {str(e)}"

@mcp.tool()
async def add_cell(path: str, index: int, cell_type: str = "code") -> str:
    """Insert a blank cell at the specified index in the notebook at given path."""
    try:
        ynotebook = registry.get_or_load(path)
        new_cell = {
            "cell_type": cell_type,
            "source": "",
            "metadata": {},
            "outputs": [] if cell_type == "code" else None,
            "execution_count": None,
        }
        ycell = ynotebook.create_ycell(new_cell)
        ynotebook._ycells.insert(index, ycell)
        return f"✅ Added {cell_type} cell at index {index} in {path}."
    except Exception as e:
        return f"❌ Error adding cell at index {index} in {path}: {str(e)}"

@mcp.tool()
async def cut_cell(path: str, index: int) -> str:
    """Remove the cell at the specified index and return its contents."""
    try:
        ynotebook = registry.get_or_load(path)
        cell = ynotebook.get_cell(index)
        ynotebook._ycells.pop(index)
        return f"✅ Cut cell {index} in {path}:\n{cell['source']}"
    except Exception as e:
        return f"❌ Error cutting cell {index} in {path}: {str(e)}"


@mcp.tool()
async def save_notebook(path: str) -> str:
    """Save the notebook at the given path back to disk."""
    try:
        registry.save(path)
        return f"✅ Saved notebook to {path}."
    except Exception as e:
        return f"❌ Error saving notebook {path}: {str(e)}"

# Autosave background task
async def autosave_loop(interval: int = 60):
    while True:
        await asyncio.sleep(interval)
        for path in registry.list_paths():
            try:
                registry.save(path)
                print(f"[autosave] Saved {path}")
            except Exception as e:
                print(f"[autosave error] Could not save {path}: {e}")

# Run the MCP server using stdio
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(autosave_loop(60))  # autosave every 60 seconds
    mcp.run(transport="stdio")
