from graphviz import Digraph
from collections import defaultdict

def find_common_prefix(strings):
    if not strings:
        return "", []
    if len(strings) == 1:
        return strings[0], []

    prefix = strings[0]
    for s in strings[1:]:
        i = 0
        while i < len(prefix) and i < len(s) and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
        
    if ' ' in prefix:
        # stop at the last space
        prefix = prefix.rsplit(' ', 1)[0]

    remaining = [s[len(prefix):].strip() for s in strings if s[len(prefix):].strip()]
    return prefix, remaining

def visualize_common_prefixes(strings):
    graph = Digraph()
    graph.attr(rankdir='LR')  # Set the direction to left-to-right
    
    def add_nodes_and_edges(strings, parent=None, level=0):
        if not strings:
            return

        prefix, remainder = find_common_prefix(strings)

        if prefix:
            node_label = prefix.strip()
            node_id = f"{level}-{node_label}"
            graph.node(node_id, node_label, shape='box', style='filled', fillcolor='lightgrey', fontsize='12')
            if parent and parent != node_label:  # Avoid self-looping edge
                graph.edge(parent, node_id)
                print(level, prefix, ':', parent, '=>', node_label)

        if remainder:
            child_groups = defaultdict(list)
            for s in remainder:
                child_groups[s.split()[0]].append(s)

            for group in child_groups.values():
                add_nodes_and_edges(group, node_id, level+1)

    # Add "[start]" prefix to each string
    prefixed_strings = ["[start] " + s for s in strings]
    add_nodes_and_edges(prefixed_strings, None)
    return graph

if __name__ == "__main__":
    strings = ["There was once", "There was a", "One sunny"]
    graph = visualize_common_prefixes(strings)
    graph.render('common_prefixes', format='png', cleanup=True)