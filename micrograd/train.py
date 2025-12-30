from typing import Tuple
from graphviz import Digraph

class Value:

    def __init__(self, data: float, sources: Tuple['Value', ...]=(), name: str='', op: str=None):
        self.data = data
        self.name = name
        self.op = op
        self.sources = sources
        self._backward = lambda: None
        self.grad = 0
    
    def __repr__(self):
        return f'Value[{self.name}]={self.data}'

    
    def __add__(self, other: 'Value'):
        o = Value(self.data + other.data, sources=(self, other), name=f'{self.name} + {other.name}', op='+')

        def b():
            self.grad = o.grad * (1.0)
            other.grad = o.grad * (1.0)
        
        o._backward = b

        return o
    
    
    def __mul__(self, other: 'Value'):
        o = Value(self.data * other.data, sources=(self, other), name=f'({self.name}) * ({other.name})', op='*')

        def b():
            self.grad = o.grad * (other.data)
            other.grad = o.grad * (self.data)
        
        o._backward = b

        return o


    def backward(self):
        self.grad = 1

        nodes = []
        visited = set()

        def traverse(root: Value):
            visited.add(root)

            for child in root.sources:
                if child not in visited:
                    traverse(child)

            nodes.append(root)
        
        traverse(self)

        nodes = reversed(nodes)

        for node in nodes:
            node._backward()

def visualize(node: Value):
    dot = Digraph(format='png', graph_attr={'rankdir': 'LR'})

    nodes = []
    edges = []

    def traverse(root: Value):
        nodes.append(root)

        for s in root.sources:
            edges.append((s, root))
            traverse(s)
    
    traverse(node)

    for node in nodes:
        dot.node(name=node.name, label=f'{node.name} | data {node.data} | grad {node.grad}', shape='record')

        if node.op != None:
            dot.node(name=f'{node.name}_{node.op}', label=f'{node.op}')
            dot.edge(tail_name=f'{node.name}_{node.op}', head_name=node.name)
        
    for u, v in edges:
        dot.edge(tail_name=u.name, head_name=f'{v.name}_{v.op}')
    
    return dot


a = Value(data=2, name='a')
b = Value(data=-3, name='b')
c = Value(data=10, name='c')
e = a * b
d = e + c
f = Value(data=-2, name='f')
L = d * f

L.backward()
graph = visualize(L)
graph.render('computation_graph', view=True)



