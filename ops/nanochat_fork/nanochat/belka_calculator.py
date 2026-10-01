"""Bounded AST arithmetic/string-count evaluator. No eval, exec or signals."""
from __future__ import annotations
import ast
import math
import operator

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
     ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod}


def calculate(expression):
    if not isinstance(expression,str) or len(expression)>2048: return None
    try:
        tree=ast.parse(expression.strip(),mode='eval')
        if sum(1 for _ in ast.walk(tree))>128: return None
        def visit(node,depth=0):
            if depth>24: raise ValueError('expression too deep')
            if isinstance(node,ast.Constant) and type(node.value) in (int,float,str):
                value=node.value
            elif isinstance(node,ast.UnaryOp) and type(node.op) in (ast.UAdd,ast.USub):
                value=visit(node.operand,depth+1)
                if type(value) not in (int,float):raise ValueError('numeric operand required')
                value=value if isinstance(node.op,ast.UAdd) else -value
            elif isinstance(node,ast.BinOp) and type(node.op) in OPS:
                left,right=visit(node.left,depth+1),visit(node.right,depth+1)
                if type(left) not in (int,float) or type(right) not in (int,float):raise ValueError('numeric operands required')
                value=OPS[type(node.op)](left,right)
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='count' and not node.keywords and 1<=len(node.args)<=3:
                text=visit(node.func.value,depth+1)
                arguments=[visit(arg,depth+1) for arg in node.args]
                if type(text) is not str or type(arguments[0]) is not str or any(type(a) is not int for a in arguments[1:]):
                    raise ValueError('literal string count only')
                value=text.count(*arguments)
            else:raise ValueError('unsupported syntax')
            if isinstance(value,str):
                if len(value)>2048:raise ValueError('string too long')
            elif type(value) is int:
                if value.bit_length()>256:raise ValueError('integer too large')
            elif not math.isfinite(value) or abs(value)>1e75:raise ValueError('numeric overflow')
            return value
        value=visit(tree.body)
        return value if type(value) in (int,float) else None
    except (ValueError,TypeError,SyntaxError,ArithmeticError,RecursionError):
        return None
