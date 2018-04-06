"""Defines functions to make huffman coding of values (not really specific
to JPEG at all (certain specifics about JPEG are currently ignored such as
the rule about 1's and the rule about code length.)
"""
import queue


def get_huffman_codes(symbols):
    """Given a list of symbols returns a dict of mappings from
       symbols to codes.
    """
    codes = {}
    root = _create_huffman_tree(symbols)
    _create_codes(codes, root)
    return codes


class _HuffmanNode:
    def __init__(self, value, frequency):
        self.value = value
        self.frequency = frequency
        self.left = None
        self.right = None
        self.code = None

    def __lt__(self, other_node):
        return self.frequency < other_node.frequency

    def __gt__(self, other_node):
        return self.frequency > other_node.frequency

    def __le__(self, other_node):
        return self.frequency <= other_node.frequency

    def __ge__(self, other_node):
        return self.frequency >= other_node.frequency

    def __eq__(self, other_node):
        return self.frequency == other_node.frequency

    def __ne__(self, other_node):
        return self.frequency != other_node.frequency


def _create_huffman_nodes(symbols):
    nodes = []
    for symbol in symbols:
        if any(node.value == symbol for node in nodes):
            for node in nodes:
                if node.value == symbol:
                    node.frequency += 1
                    break
        else:
            nodes.append(_HuffmanNode(symbol, 1))
    return nodes


def _create_huffman_tree(symbols):
    priority_queue = queue.PriorityQueue()
    nodes = _create_huffman_nodes(symbols)
    for node in nodes:
        priority_queue.put(node)

    while priority_queue.qsize() > 1:
        smallest = priority_queue.get()
        next_smallest = priority_queue.get()
        internal_node = _HuffmanNode(None,
                                     smallest.frequency+next_smallest.frequency
                                     )
        internal_node.left = smallest
        internal_node.right = next_smallest
        priority_queue.put(internal_node)

    root = priority_queue.get()
    assert priority_queue.empty()
    return root


def _create_codes(output, node, current_code=''):
    if node.value:
        if current_code:
            node.code = current_code
            output[node.value] = node.code
        else:
            node.code = '0'
            output[node.value] = node.code
    else:
        _create_codes(output, node.left, current_code + '0')
        _create_codes(output, node.right, current_code + '1')
