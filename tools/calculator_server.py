from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("CalculatorServer")


@mcp.tool()
async def calculate(expression: str) -> str:
    """
    Evaluate a math expression.
    :param expression: math expression, e.g. "2 + 3 * 4"
    :return: result string
    """
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculation error: {exc}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
