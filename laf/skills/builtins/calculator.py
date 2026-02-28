import ast
import operator
from typing import Dict, Any

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):
        return _ALLOWED_OPERATORS[type(node.op)](
            _eval(node.left), _eval(node.right)
        )
    elif isinstance(node, ast.UnaryOp):
        return _ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
    else:
        raise ValueError("Unsupported expression")


def calculator(expression: str = "") -> Dict[str, Any]:
    if not expression:
        raise ValueError("Missing expression")

    expression = (expression or "").strip()

    # Treat caret as exponent for user convenience in console mode.
    # (If you ever want XOR later, add an explicit "xor(a,b)" tool.)
    expression = expression.replace("^", "**")

    node = ast.parse(expression, mode="eval").body
    return {"expression": expression, "result": _eval(node)}

# MEMORY EXTRACTOR

def calculator_memory(args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    expression = result.get("expression")
    value = result.get("result")

    return {
        "calculations": [
            {
                "expression": expression,
                "result": value,
            }
        ]
    }