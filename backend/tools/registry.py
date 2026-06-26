class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, func, description: str):
        self.tools[name] = {
            "func": func,
            "description": description
        }

    def execute_tool(self, name: str, kwargs: dict):
        if name in self.tools:
            return self.tools[name]["func"](**kwargs)
        return f"Tool {name} not found."

registry = ToolRegistry()

# Example tools
def filesystem_read(path: str):
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return str(e)

def terminal_run(command: str):
    # Dummy implementation for safety
    return f"Executed {command} in sandbox"

registry.register_tool("filesystem_read", filesystem_read, "Read file contents")
registry.register_tool("terminal_run", terminal_run, "Run a bash command")
