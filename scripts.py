#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import re
from collections import OrderedDict
from IPython.display import display
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import jit, prange, njit, config
from numba import int64, types, typed
from numba.experimental import jitclass
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from numba.core.errors import NumbaWarning
import warnings
from numba.typed import Dict, List
from math import nan, sin, exp, log10



warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


def load_and_print_csvs_from_folders():
    cwd = os.getcwd()
    folders = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]
    data = {}
    for folder in folders:
        folder_path = os.path.join(cwd, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                data[file] = df

    return data


def pretty_print_dict(dictionary, indent=0):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            print('  ' * indent + str(key) + ':')
            pretty_print_dict(value, indent + 1)
        else:
            print('  ' * indent + str(key) + ': ' + str(value))


@njit
def int_to_base36(value):
    if value < 0:
        value = -value
    if value == 0:
        return "0"

    base36_digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    base36 = []
    while value:
        value, i = divmod(value, 36)
        base36.append(base36_digits[i])

    # Reversing the list
    base36 = base36[::-1]

    # Joining the list into a string
    base36_str = ""
    for char in base36:
        base36_str += char

    return base36_str


@njit()
def genotype_hash(genotype_param, precision=16):
    # Sort genotype keys to ensure a consistent order
    sorted_keys = sorted(genotype_param.keys())

    # Create a string representation of the genotype
    genotype_str = ""
    for key in sorted_keys:
        genotype_str += key + ":"
        # genotype_str += ",".join(str(val) for val in genotype_param[key])
        for i in range(len(genotype_param[key])):
            genotype_str += "," + str(genotype_param[key][i])
        genotype_str += ";"

    hash_object = hash(genotype_str)
    str_hash = int_to_base36(hash_object)
    length = min(precision, len(str_hash))

    # get the first 16 characters of the hash
    return str_hash[:length]


@njit
def deep_copy_genotype(genotype, empty_genotype):
    for key in genotype:
        values = genotype[key]
        copied_values = values.copy()
        empty_genotype[key] = copied_values

    return empty_genotype


def read_bnf_file(file_path):
    # read file as string
    with open(file_path, 'r') as f:
        bnf_lines = f.read()
    return bnf_lines


def parse_bnf(grammar_str, depth=1, variables=1):
    # productions = Dict.empty(key_type=types.unicode_type, value_type=types.ListType(types.ListType(types.unicode_type)))
    productions = {}
    grammar_str = re.sub(r'\s*\|\s*', ' |', grammar_str)
    lines = [line.strip() for line in grammar_str.split('\n') if line.strip()]
    PRODUCTION_SEPARATOR = '::='
    RULE_SEPARATOR = '|'
    RECURSIVE_PRODUCTIONS = ['<expr>']
    for line in lines:
        lhs, rhs = line.split(PRODUCTION_SEPARATOR)
        lhs = lhs.strip()
        rhs_productions = []
        for rule in rhs.split(RULE_SEPARATOR):
            rule_list = list(rule.strip().split())
            rhs_productions.append(rule_list)

        productions[lhs] = rhs_productions

    def list_new_non_recursive_expressions(recursive_rule):
        # create depth expressions
        expressions = [recursive_rule]
        for i in range(1, depth):
            expr_rule = recursive_rule.replace('>', f'{i}>')
            expressions.append(expr_rule)
        return expressions

    def filter_non_recursive_productions(recursive_rule):
        non_recursive_productions = set()
        for exp in productions[recursive_rule]:
            if recursive_rule not in exp:
                non_recursive_productions.add(tuple(exp))

        return non_recursive_productions

    def replace_expression_in_production(old_rule, old, new):
        new_rule = list(old_rule)
        for i, exp in enumerate(new_rule):
            if new_rule[i] == old:
                new_rule[i] = new
        return new_rule

    def fix_recursive_production(recursive_rule):
        non_recursive_productions = filter_non_recursive_productions(recursive_rule)
        if depth == 1:
            productions[recursive_rule] = non_recursive_productions
            return

        new_expressions = list_new_non_recursive_expressions(recursive_rule)

        all_production = set(tuple(p) for p in productions[recursive_rule])
        recursive_productions = all_production - non_recursive_productions

        for i in range(0, depth - 1):
            expr_rule = new_expressions[i]
            new_productions = []
            for rule in recursive_productions:
                new_rule = replace_expression_in_production(rule, recursive_rule, new_expressions[i + 1])
                new_productions.append(new_rule)
            productions[expr_rule] = [*non_recursive_productions, *new_productions]

        productions[new_expressions[-1]] = list(non_recursive_productions)

    for recursive_rule in RECURSIVE_PRODUCTIONS:
        fix_recursive_production(recursive_rule)

    variable_cases = ['x', 'y', 'z', '{', '|', '}', '~', '\x7f', '\x80', '\x81']

    for var_index in range(variables + 1):
        terminal_vars = variable_cases[:var_index]
        productions["<var>"] = terminal_vars

    return productions


demo_bnf = """
<start> ::= <expr> <op> <expr>
<expr> ::= <term> <op> <term> | '(' <term> <op> <term> ')'
<op> ::= '+' | '-' | '/' | '*'
<term> ::= 'x1' | '0.5'
"""


def get_terminals(productions):
    terminals = set()
    non_terminals = set(productions.keys())

    for rhs in productions.values():
        for rule in rhs:
            for token in rule:
                if token not in non_terminals:
                    terminals.add(token)

    return terminals


def create_typed_grammar(productions_dict):
    productions = Dict.empty(key_type=types.unicode_type, value_type=types.ListType(types.ListType(types.unicode_type)))

    for nt, rhs_list in productions_dict.items():
        # rhs_typed_list = List[types.ListType(types.unicode_type)]

        rhs_typed_list = typed.List.empty_list(types.ListType(types.unicode_type))
        for rhs in rhs_list:
            # rhs_typed = List[types.unicode_type]
            rhs_typed = typed.List.empty_list(types.unicode_type)
            for symbol in rhs:
                rhs_typed.append(symbol)

            rhs_typed_list.append(rhs_typed)

        productions[nt] = rhs_typed_list

    return productions


def create_grammar_from_bnf(bnf_file, depth=1, variables=2):
    bnf_lines = read_bnf_file(bnf_file)
    productions = parse_bnf(bnf_lines, depth, variables)
    productions = create_typed_grammar(productions)
    non_terminals = set(productions.keys())
    terminals = get_terminals(productions)
    return productions, non_terminals, terminals


def test_grammar_creation():
    productions, non_terminals, terminals = create_grammar_from_bnf("variable_depth.bnf", 5, 2)

    pretty_print_dict(productions)


def test_terminal_and_nonterminal():
    productions, _, _ = create_grammar_from_bnf("variable_depth.bnf", 5, 2)

    terminals = get_terminals(productions)

    print("terminals")
    print(terminals)
    non_terminals = set(productions.keys())
    print("non_terminals")
    print(non_terminals)


# In[4]:


def find_recursive_and_non_recursive_terminals(grammar):
    recursive_terminals = set()
    non_recursive_terminals = set()
    non_terminals = set(grammar.keys())
    terminals = get_terminals(grammar)

    def is_recursive(nt, visited):
        if nt in visited:
            return True
        visited.add(nt)
        for rule in grammar[nt]:
            for token in rule:
                if token in non_terminals and is_recursive(token, visited):
                    return True
        visited.remove(nt)
        return False

    for nt in non_terminals:
        if is_recursive(nt, set()):
            recursive_terminals.add(nt)
        else:
            non_recursive_terminals.add(nt)

    non_recursive_terminals |= terminals

    return recursive_terminals, non_recursive_terminals


def test_recursive_terminals_calculation():
    productions, _, _ = create_grammar_from_bnf("variable_depth.bnf", 5, 2)
    recursive_terminals, non_recursive_terminals = find_recursive_and_non_recursive_terminals(productions)

    display("Recursive terminals:", recursive_terminals)
    display("Non-recursive terminals:", non_recursive_terminals)


# In[5]:


def calculate_non_recursive_productions(symbol, grammar):
    non_recursive_indices = []
    recursive_terminals_in_grammar, _ = find_recursive_and_non_recursive_terminals(grammar)

    if symbol not in grammar:
        return non_recursive_indices

    for i, rule in enumerate(grammar[symbol]):
        if all(token in recursive_terminals_in_grammar for token in rule):
            non_recursive_indices.append(i)

    return non_recursive_indices


def calculate_recursive_productions(non_terminal, grammar, non_recursive_productions):
    all_productions = set(range(len(grammar[non_terminal])))
    non_recursive_set = set(non_recursive_productions[non_terminal])
    return list(all_productions - non_recursive_set)


def get_non_recursive_expansions(grammar):
    non_terminals = set(grammar.keys())
    non_recursive_expansions_set = OrderedDict()
    for nt in non_terminals:
        non_recursive_expansions_set[nt] = calculate_non_recursive_productions(nt, grammar)
    return non_recursive_expansions_set


def get_recursive_expansions(grammar, non_recursive_expansions_dict):
    recursive_expansions_set = OrderedDict()
    non_terminals = set(grammar.keys())

    for nt in non_terminals:
        recursive_expansions_set[nt] = calculate_recursive_productions(nt, grammar, non_recursive_expansions_dict)
    return recursive_expansions_set


def test_recursive_expansions_calculation():
    productions, _, _ = create_grammar_from_bnf("variable_depth.bnf", 5, 2)
    # create the non-recursive dictionary for each non-terminal
    non_recursive_expansions = get_non_recursive_expansions(productions)
    recursive_expansions = get_recursive_expansions(productions, non_recursive_expansions)

    print("Non recursive expansions per terminal:")
    pretty_print_dict(non_recursive_expansions)

    print("Recursive expansions per terminal:")
    pretty_print_dict(recursive_expansions)


# In[6]:


count_references_type = Dict.empty(key_type=types.unicode_type,
                                   value_type=types.DictType(types.unicode_type, types.int64))
is_referenced_by_type = Dict.empty(key_type=types.unicode_type, value_type=types.ListType(types.unicode_type))


def calculate_non_terminal_references(grammar, non_terminals_set):
    # count_references = {nt: {} for nt in non_terminals_set}
    # count_references = Dict.empty(key_type=types.unicode_type, value_type=types.DictType(types.unicode_type, types.int64))
    """
    count_references = dict()
    for nt in non_terminals_set:
        #count_references[nt] = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        count_references[nt] = dict()
    #is_referenced_by = Dict.empty(key_type=types.unicode_type, value_type=types.unicode_type[:])
    is_referenced_by = dict()
    for nt in non_terminals_set:
        #is_referenced_by[nt] = List.empty_list(types.int64)
        is_referenced_by[nt] = []
    #is_referenced_by = {nt: [] for nt in non_terminals_set}
    """
    count_references = count_references_type.copy()
    is_referenced_by = is_referenced_by_type.copy()

    for nt in non_terminals_set:
        count_references[nt] = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
        is_referenced_by[nt] = List.empty_list(types.unicode_type)

    for nt in non_terminals_set:
        for production in grammar[nt]:
            count = {option: 0 for option in non_terminals_set}
            for option in production:
                if option in non_terminals_set:
                    is_referenced_by[option].append(nt)
                    count[option] += 1
            for key in count:
                count_references[key][nt] = max(count_references[key].get(nt, 0), count[key])

    return count_references, is_referenced_by




# In[7]:


@jit(nopython=True)
def find_references(nt, start_symbol, is_referenced_by, count_references_by_prod):
    r = get_total_references_of_current_production(count_references_by_prod, nt)
    results = []

    if nt == start_symbol:
        return 1
    nt_str = str(nt)
    for ref in is_referenced_by[nt_str]:
        results.append(find_references(ref, start_symbol, is_referenced_by, count_references_by_prod))

    references = r * np.max(np.array(results))
    return references


@jit(nopython=True)
def get_total_references_of_current_production(count_references_by_prod, nt):
    nt_str = str(nt)
    return np.sum(np.array(list(count_references_by_prod[nt_str].values())))


def test_reference_counting(nt="<digit>"):
    print("Testing reference counting for non terminal: ", nt)
    productions, non_terminals, terminals = create_grammar_from_bnf("variable_depth.bnf", 5, 2)
    # Replace this with a non-terminal from your grammar
    count_refs, ref_by = calculate_non_terminal_references(productions, non_terminals)
    references = find_references(nt, '<start>', ref_by, count_refs)

    print("References: ", references)

    # Usage example
    count_refs, ref_by = calculate_non_terminal_references(productions, non_terminals)
    print("count_refs")

    pretty_print_dict(count_refs)
    print("Ref by: ")
    pretty_print_dict(ref_by)



# In[8]:


class Tree:

    def __init__(self, genome, productions):
        # get first rule for start symbol from the grammar

        self.productions = productions
        self.non_terminals = set(productions.keys())
        first_rule = next(iter(productions))
        first_production = productions[first_rule][0]
        self.root = Node(first_rule)

        # initialize OrderedDict with each non-terminal as a key and an empty list as the value, without list comprehension
        self.genome = genome

    def __repr__(self):
        # return f"Node({self.label}, {self.children})"
        # if it has children, call repr on each child
        return self.root.children[0].simple_repr()

    def __str__(self):
        return self.root.__str__()

    def _get_next_expansion(self):
        # find the first non-terminal that has not been expanded
        return self.root.find_first_unexpanded_non_terminal()

    def expand_next(self):
        node = self._get_next_expansion()
        if node is None:
            return False

        # get the vector of production indices for the current non-terminal
        production_indices = self.genome[node.label]

        if len(production_indices) == 0:
            raise ValueError(f"Genome for {node.label} is empty")

        # get the next production index
        production_index = production_indices.pop(0)

        # get the production for the desired non-terminal
        new_production = self.productions[node.label][production_index]

        # print("Expanding", node.label, "with", new_production)
        node.apply_rule(self.productions, new_production)
        return True


class Node:

    def __init__(self, label, first_production=None):
        # print("Creating node with label", label)
        self.label = label
        self.children = []
        self.is_terminal = False

        if first_production is not None and len(first_production) > 0:
            self.apply_rule(first_production)

    def simple_repr(self):
        if len(self.children) > 0:
            merged_string = " ".join([child.simple_repr() for child in self.children])
            return merged_string
        return self.label

    def find_first_unexpanded_non_terminal(self):
        if self.is_terminal:
            return None

        if len(self.children) == 0:
            return self

        for child in self.children:
            non_terminal = child.find_first_unexpanded_non_terminal()
            if non_terminal is not None:
                return non_terminal

        return None

    def apply_rule(self, current_tree_productions, production: list):

        if production not in current_tree_productions[self.label]:
            raise ValueError(f"Production {production} not found in grammar for {self.label}")

        children_list = []
        for symbol in production:
            child = Node(symbol)
            if child.label not in current_tree_productions:
                child.is_terminal = True
            children_list.append(child)

        self.children = children_list


# In[9]:


@njit
def seed(a):
    random.seed(a)


@njit
def rand():
    return random.random()


@njit
def random_int(a, b):
    return random.randint(a, b)


production_type = types.ListType(types.int64)
new_genotype_type = Dict.empty(
    key_type=types.unicode_type,
    value_type=production_type
)


@jit(nopython=False)
def create_full_tree(grammar, genotype, first_symbol, ref_by_dict, count_refs_dict, non_terminals_set):
    # print(non_terminals_set)
    # print(numba.typeof(non_terminals_set))
    # print(numba.version_info)
    # for every non-terminal, we create a vector of size equal to the upper bound of productions
    non_terminals_list = list(non_terminals_set)
    for symbol_index in prange(len(non_terminals_list)):
        symbol = str(non_terminals_list[symbol_index])

        upper_bound = find_references(symbol, first_symbol, ref_by_dict, count_refs_dict)

        productions_length = len(grammar[symbol])
        possible_productions = List.empty_list(types.int64)
        for i in range(upper_bound):
            codon = random_int(0, productions_length - 1)
            possible_productions.append(codon)

        genotype[symbol] = possible_productions


def create_individual_probabilistic(grammar, max_depth, genotype, symbol, non_terminals_set, depth):
    stack = [(symbol, depth)]
    non_recursive_expansions_dict = get_non_recursive_expansions(grammar)
    recursive_expansions = get_recursive_expansions(grammar, non_recursive_expansions_dict)
    is_terminal_cache = {s not in non_terminals_set for s in grammar}
    unique_depths = set()
    non_recursive_expansions_dict = get_non_recursive_expansions(grammar)

    while stack:
        symbol, depth = stack.pop()
        if depth not in unique_depths:
            # print(f"Reached new unique depth: {depth}")
            unique_depths.add(depth)
        production_rules = grammar[symbol]
        expansion_index = random.randint(0, len(production_rules) - 1)

        expansion = production_rules[expansion_index]

        # handle case where symbol is recursive, checking the dict
        is_expansion_rec = expansion in recursive_expansions[symbol]
        # if not is_symbol_rec:
        # print(f"Symbol {symbol} is non recursive")
        if is_expansion_rec:
            if depth >= max_depth:
                non_rec_exps = non_recursive_expansions_dict[symbol]
                if len(non_rec_exps) == 0:
                    print("Symbol", symbol, "has no non-recursive productions")
                    raise ValueError("No valid productions in this case")
                expansion_index = random.choice(non_rec_exps)
                expansion = grammar[symbol][expansion_index]

        if symbol in genotype:
            genotype[symbol].append(expansion_index)
        else:
            genotype[symbol] = [expansion_index]

        expansion_symbols = production_rules[expansion_index]

        for sym in expansion_symbols:
            if not is_terminal_cache.get(sym, True):
                stack.append((sym, depth + 1))
    print("Unique depths: ", unique_depths)


#
# def create_individual_recursive(grammar, max_depth, genotype, symbol, depth):
#     production_rules = grammar[symbol]
#     expansion_index = random.randint(0, len(production_rules) - 1)
#
#     expansion = production_rules[expansion_index]
#     # check if the symbol is a recursive terminal, ie, it can expand to itself
#     if is_recursive(symbol):
#         # check if the expansion is recursive
#         if expansion in recursive_expansions[symbol]:
#             if depth >= max_depth:
#                 # get non recursive productions of the symbol
#                 non_rec_exps = non_recursive_expansions[symbol]
#                 if len(non_rec_exps) == 0:
#                     print("Symbol", symbol, "has no non-recursive productions")
#                     return
#                 expansion_index = random.choice(non_rec_exps)
#                 expansion = grammar[symbol][expansion_index]
#     else:
#       print(f"Symbol {symbol} is non recursive!")
#
#     if symbol in genotype:
#         genotype[symbol].append(expansion_index)
#     else:
#         genotype[symbol] = [expansion_index]
#
#     expansion_symbols = production_rules[expansion_index]
#
#     for sym in expansion_symbols:
#         if not is_terminal(sym):
#             create_individual_probabilistic(grammar, max_depth, genotype, sym, depth + 1)

def create_genotype(grammar_file='variable_depth.bnf', max_depth=6, variables_count=2, option='full'):
    new_genotype = new_genotype_type.copy()

    desired_depth_grammar, non_terminals, terminals = create_grammar_from_bnf(grammar_file, max_depth, variables_count)
    first_symbol = next(iter(desired_depth_grammar.keys()))

    count_refs, ref_by = calculate_non_terminal_references(desired_depth_grammar, non_terminals)

    if option == 'full':
        create_full_tree(desired_depth_grammar, new_genotype, first_symbol, ref_by, count_refs, non_terminals)
    elif option == 'probabilistic':
        create_individual_probabilistic(desired_depth_grammar, max_depth, new_genotype, first_symbol, 0)
    return new_genotype, desired_depth_grammar


def test_create_genotype():
    create_genotype(max_depth=3)


# In[10]:


# NodeData class definition
node_data_spec = [
    ('id', types.int64),
    ('label', types.unicode_type),
    ('children', types.ListType(types.int64)),
    ('is_terminal', types.boolean),
]


@jitclass(node_data_spec)
class NodeData:
    def __init__(self, node_id, label):
        self.id = node_id
        self.label = label
        self.children = List.empty_list(types.int64)
        self.is_terminal = False


# Function to create a new NodeData instance and store it in the global hash table
@njit
def create_node(label, nodes_table):
    id = len(nodes_table)
    node = NodeData(id, label)
    nodes_table[id] = node
    return id


# Functions to work with NodeData instances
@njit
def apply_rule(node_id, node_table, grammar, production):
    node = node_table[node_id]

    found = False

    # print current productions
    # print("Current productions for", node.label, "are", current_tree_productions[node.label])

    for prods in grammar[node.label]:
        # print(set(production), set(prods))
        if set(production) == set(prods):
            found = True
            break
    if not found:
        raise ValueError(f"Production {production} not found in grammar for {node.label}")

    # if production not in current_tree_productions[node.label]:
    #    raise ValueError(f"Production {production} not found in grammar for {node.label}")

    children_list = List.empty_list(types.int64)
    for symbol in production:
        child_id = create_node(symbol, node_table)
        current_node = node_table[child_id]

        found = False
        for prod in grammar:
            # print(prod)
            if current_node.label == prod:
                found = True
                # print(f"Found {current_node.label} in {prod}, not terminal")
                break
        if not found:
            current_node.is_terminal = True

        children_list.append(child_id)

    node.children = children_list


@njit
def find_first_unexpanded_non_terminal(node_id: int, node_table, productions):
    node = node_table[node_id]
    if node.is_terminal:
        return -1

    if len(node.children) == 0:
        return node_id

    for child_id in node.children:
        non_terminal_idx = find_first_unexpanded_non_terminal(child_id, node_table, productions)
        if non_terminal_idx != -1:
            return non_terminal_idx

    return -1
    # Implement find_first_unexpanded_non_terminal functionality


@njit
def expand_next(node_id, node_table, genome, productions):
    expansion_node_id = find_first_unexpanded_non_terminal(node_id, node_table, productions)

    if expansion_node_id == -1:
        return False

    # get the vector of production indices for the current non-terminal
    node = node_table[expansion_node_id]
    production_indices = genome[node.label]

    # print(f"Node: {node.label}, children = {node.children}, genome={genome}")

    if len(production_indices) == 0:
        print(f"Node: {node.label}, children = {node.children}, genome={genome}")
        raise ValueError(f"Genome for {node.label} is empty")

    # get the next production index
    production_index = production_indices.pop(0)


    new_production = productions[node.label][production_index]

    apply_rule(expansion_node_id, node_table, productions, new_production)
    return True


@njit
def load_genotype(node_table, genome, productions, print_tree=False, print_aplications=False):
    counter = 0
    while expand_next(0, node_table, genome, productions):
        counter += 1

    if print_aplications:
        print("Applied rule ", counter, " times")
        print_nodes(node_table)
        print()
    if print_tree:
        simple_repr(0, node_table)


@njit
def simple_repr(node_id, node_table):
    node = node_table[node_id]

    if len(node.children) > 0:
        merged_string = ""
        for child_id in node.children:
            merged_string += simple_repr(child_id, node_table) + " "
        return merged_string.strip()
    return node.label


@njit
def print_nodes(nodes_table):
    for node_id in nodes_table.keys():
        node = nodes_table[node_id]
        print(f"{node_id}: {node.label}")


def generate_and_expand(node_table, depth=4, variables_count=2):
    genotype, grammar = create_genotype(max_depth=depth, variables_count=variables_count)
    copy_genome = {}

    for key in genotype:
        copy_genome[key] = genotype[key].copy()

    tree = Tree(copy_genome, grammar)
    while tree.expand_next():
        pass
    load_genotype(node_table, genotype, grammar)
    display(simple_repr(0, node_table))
    display(tree)
    return tree


# In[11]:




@njit
def remove_quotes(target: str) -> str:
    # replace quotes with nothing
    # return target.replace("'", "").replace('"', '')
    result = ""
    for char in target:
        if char != "'" and char != '"':
            result += char
    return result


@njit
def convertStrToInt(s: str) -> int:
    sign = 1
    if s.startswith('-'):
        sign = -1
        s = s[1:]
    elif s.startswith('+'):
        s = s[1:]

    integer_value = 0
    for c in s:
        if c < '0' or c > '9':
            return 0  # or you can return some error code to indicate the conversion failed

        digit = ord(c) - ord('0')
        integer_value = integer_value * 10 + digit

    return sign * integer_value


@njit
def evaluate_string(node_id: int, node_table):
    node = node_table[node_id]
    return node.label


@njit
def evaluate(node_id: int, node_table, variables_dict) -> float:
    node = node_table[node_id]
    label = node.label

    # print("Evaluating ", label)

    if label == "<start>":
        return evaluate(node.children[0], node_table, variables_dict)
    elif node.label.startswith('<expr'):
        if len(node.children) == 1:  # <number>
            return evaluate(node.children[0], node_table, variables_dict)
        else:
            operator_wrapper_node_id = node.children[1]
            operator_node_id = node_table[operator_wrapper_node_id].children[0]
            if len(node.children) == 4:  # ( <uop> <expr> )
                uop = evaluate_string(operator_node_id, node_table)
                uop = remove_quotes(uop)
                operand_node_id = node.children[2]
                operand = evaluate(operand_node_id, node_table, variables_dict)

                if uop == 'abs':
                    return abs(operand)
                elif uop == 'sin':
                    return sin(operand)
                elif uop == 'exp':
                    if operand > 50:
                        return float(1000000)
                    elif operand < -50:
                        return float(0.000001)
                    return exp(operand)
                elif uop == 'log':
                    if operand <= 0:
                        return 0
                    return log10(operand)

                else:
                    print("Error, unknown unary operator")
                    return nan
            elif len(node.children) == 5:  # ( <op> <expr> <expr> )

                op = evaluate_string(operator_node_id, node_table)
                op = remove_quotes(op)
                op1 = evaluate(node.children[2], node_table, variables_dict)
                op2 = evaluate(node.children[3], node_table, variables_dict)
                if op == "+":
                    return op1 + op2
                elif op == "-":
                    return op1 - op2
                elif op == "*":
                    return op1 * op2
                elif op == "/":
                    # safe division
                    if op2 == 0:
                        return 1000000
                    return op1 / op2
                else:
                    print("Error, unknown binary operator: ", op)
                    return nan

    elif label == "<number>":
        return evaluate(node.children[0], node_table, variables_dict)

    elif label == "<integer>":
        # Remove nodes where label is parenthesis from children
        digit_node_ids = List.empty_list(int64)
        for child_id in node.children:
            curr = node_table[child_id]
            if curr.label != '(' and curr.label != ')':
                # print("Adding ", curr.label, " to digit_node_ids, with id", child_id)
                digit_node_ids.append(child_id)

        # Evaluate each digit node and join them together
        merged_digits = ""
        for child_id in digit_node_ids:
            number_id = node_table[child_id].children[0]
            # evaluate returns a float, we need to cast to int
            evaluated_digit = int(evaluate(number_id, node_table, variables_dict))
            digit_value = str(evaluated_digit)
            # print("Digit value is ", digit_value)
            merged_digits += digit_value
        temp = convertStrToInt(merged_digits)
        return float(temp)

    elif label == "<var>":
        desired_variable = evaluate_string(node.children[0], node_table)
        desired_variable = remove_quotes(desired_variable)

        if desired_variable not in variables_dict:
            raise ValueError(f"Variable {desired_variable} not found in variables {variables_dict}")
        return float(variables_dict[desired_variable])

    elif label in ["<non-zero-digit>", "<digit>"]:
        return evaluate(node.children[0], node_table, variables_dict)
    elif label in "0123456789":
        # print("Digit found! ", label)
        temp = convertStrToInt(label)
        # print("Converted to ", temp)
        return float(temp)

    print(f"Unexpected node label: {label}")
    return nan


def test_print_tree_and_evaluate():
    # Global hash table to store NodeData instances
    nodes = Dict.empty(key_type=types.int64, value_type=NodeData.class_type.instance_type)

    variables = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    create_node('<start>', nodes)
    tree = generate_and_expand(nodes, depth=6, variables_count=2)

    # Add variables to the dictionary
    variables['x'] = 42.0
    variables['y'] = 10.0
    # print_nodes(nodes)
    print(simple_repr(0, nodes))
    evaluate(0, nodes, variables)


# In[12]:


def plot_tree(tree):
    G = nx.DiGraph()

    def add_edges(node):
        for child in node.children:
            G.add_edge(node.label, child.label)
            add_edges(child)

    add_edges(tree.root)

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=12, font_weight="bold",
            arrowsize=20)
    plt.show()


# In[13]:


def clean_string(target: str):
    # replace quotes with nothing
    return target.replace("'", "").replace('"', '')


def evaluate_slow(node: Node, variables):
    if node.label == "<start>":
        return evaluate_slow(node.children[0], variables)
    elif re.match(r'<expr\d*>', node.label):
        if len(node.children) == 1:  # <number>
            return evaluate_slow(node.children[0], variables)
        else:
            operator_node = node.children[1]
            if len(node.children) == 4:  # ( <uop> <expr> )
                # print("Operator node: ", operator_node)
                uop = evaluate_slow(operator_node, variables)
                uop = clean_string(uop)
                # print("Evaluated operator: [", uop, "]")
                operand_node = node.children[2]
                operand = evaluate_slow(operand_node, variables)
                if operand is None:
                    print("Operand was ", operand_node, "before evaluation, error")
                    # operand = evaluate(operand_node)
                if uop == 'abs':
                    # print("Doing abs of ", operand, " =",abs(operand))
                    return abs(operand)
                else:
                    raise ValueError(f"Unknown unary operator {uop}")
            elif len(node.children) == 5:  # ( <op> <expr> <expr> )
                op = evaluate_slow(operator_node, variables)
                op = clean_string(op)
                op1 = evaluate_slow(node.children[2], variables)
                op2 = evaluate_slow(node.children[3], variables)
                if op == "+":
                    return op1 + op2
                elif op == "-":
                    return op1 - op2
                elif op == "*":
                    return op1 * op2
                elif op == "/":
                    # safe division
                    if op2 == 0:
                        return 0
                    return op1 / op2

    elif node.label == "<number>":
        return evaluate_slow(node.children[0], variables)

    elif node.label == "<integer>":
        # Remove nodes where label is parenthesis from children
        parenthesis = ['(', ')']
        digit_nodes = [child for child in node.children if child.label not in parenthesis]

        # Evaluate each digit node and join them together
        merged_digits = [str(evaluate_slow(child, variables)) for child in digit_nodes]
        return int("".join(merged_digits))

    elif node.label == "<var>":
        desired_variable = node.children[0].label
        desired_variable = clean_string(desired_variable)
        if desired_variable not in variables:
            raise ValueError(f"Variable {desired_variable} not found in variables {variables}")
        return variables[desired_variable]
    elif node.label == "<op>":
        return node.children[0].label
    elif node.label == "<uop>":
        return node.children[0].label
    elif node.label in ["<non-zero-digit>", "<digit>"]:
        return evaluate_slow(node.children[0], variables)
    elif node.label in "0123456789":
        return int(node.label)
    elif node.label == "(" or node.label == ")":
        return ""
    else:
        raise ValueError(f"Unexpected node label: {node.label}")


# In[14]:


import concurrent.futures


def create_single_genotype(args):
    max_depth, variables_count = args
    try:
        genotype, _ = create_genotype(max_depth=max_depth, variables_count=variables_count)
        return genotype
    except Exception as e:
        print("Error creating genotype: ", e)
        raise e


def create_n_genotypes(n: int, max_depth: int, variables_count=3) -> List[new_genotype_type]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        genotypes = list(executor.map(create_single_genotype, [(max_depth, variables_count) for _ in range(n)]))
    _, grammar = create_genotype(max_depth=max_depth, variables_count=variables_count)
    # genotypes = [create_single_genotype((max_depth, variables_count)) for _ in range(n)]
    # print("First created genotype: ", genotypes[0])
    return genotypes, grammar


def test_n_creation():
    # Example usage
    n = 100
    depth = 3
    variables = {'x': 42, 'y': 10}

    genotypes, grammar = create_n_genotypes(n, depth, len(variables.keys()))
    assert len(genotypes) == n, "Should create n genotypes"

    all_hashes = [genotype_hash(genotype) for genotype in genotypes]
    print("Genotypes: ", len(genotypes))
    unique_hashes = set(all_hashes)
    assert len(unique_hashes) == len(genotypes), "Genotypes must be, generally, unique"

    print("Unique genotypes: ", len(set(all_hashes)))
    print("First genotype: ", genotypes[0])


# In[15]:


GENOTYPE_TYPE = Dict.empty(
    key_type=types.unicode_type,
    value_type=production_type
)


def assert_equality_of_hashes(hash1, hash2):
    if hash1 != hash2:
        print("Hashes are not equal")
        print("Hash1: ", hash1)
        print("Hash2: ", hash2)
        raise ValueError("Hashes are not equal")


def mutate_genotypes_1(genotypes, grammar, mutation_rate_operators: float):
    for genotype_instance in genotypes:
        mutate_genotype_inplace(genotype_instance, grammar, mutation_rate_operators)


def mutate_genotypes_2(genotypes, grammar, mutation_rate_items: float, mutation_rate_operators: float):
    for genotype in genotypes:
        # Pick random genotypes to mutate, using the mutation_rate_items.
        if rand() < mutation_rate_items:
            mutate_genotype_inplace(genotype, grammar, mutation_rate_operators)


def mutate_genotypes_3(genotypes, grammar, mutation_rate_items: float, mutation_rate_operators: float):
    for genotype in genotypes:
        # Pick random keys to mutate, using the mutation_rate_items
        subgenotype_to_mutate = new_genotype_type.copy()
        for key in genotype:
            if random.random() < mutation_rate_items:
                subgenotype_to_mutate[key] = genotype[key]
            # [key for key in genotype if random.random() < mutation_rate_items]

        mutate_genotype_inplace(subgenotype_to_mutate, grammar, mutation_rate_operators)

        # Update the genotype with the mutated sub-genotype
        for key in subgenotype_to_mutate:
            genotype[key] = subgenotype_to_mutate[key]


def mutate_genotype_old(mutable_genotype, grammar, mutation_rate_operators: float):
    genotype_keys = list(mutable_genotype.keys())
    for i in range(len(genotype_keys)):
        print("Current genotype: ", mutable_genotype)
        key = genotype_keys[i]
        gene = mutable_genotype[key]
        print("Current key: ", key)
        print("Current gene: ", gene)

        productions = grammar[key]
        productions_length = len(productions)
        for j in range(len(gene)):
            # Decide whether to mutate this gene based on the mutation_rate_operators
            if random.random() < mutation_rate_operators:
                # Choose a random production index different from the current gene index
                print(f"Mutating gene {gene} at index {j}")
                new_production_index = random_int(0, productions_length - 1)
                while new_production_index == gene[j]:
                    new_production_index = random_int(0, productions_length - 1)
                gene[j] = new_production_index

        mutable_genotype[key] = gene


#@njit
def mutate_genotype_inplace(mutable_genotype: Dict, grammar: Dict, mutation_rate_operators: float):
    genotype_keys = List(mutable_genotype.keys())
    count = 0
    for i in range(len(genotype_keys)):
        key = genotype_keys[i]
        gene = mutable_genotype[key]

        productions = grammar[key]
        productions_length = len(productions)
        for j in range(len(gene)):
            # Decide whether to mutate this gene based on the mutation_rate_operators
            # the check for productions_length > 1 is to avoid mutating a gene that has only one production, leading to broken grammar
            if np.random.random() < mutation_rate_operators and productions_length > 1:
                # Choose a random production index different from the current gene index
                new_production_index = random_int(0, productions_length-1)
                attempts = 0
                max_attempts = 10
                while new_production_index == gene[j] and attempts < max_attempts:
                    new_production_index = random_int(0, productions_length-1)
                    attempts += 1
                if attempts < max_attempts:
                    gene[j] = new_production_index
                    count += 1
                #else:
                #    print(
                #        f"At key {key}: Could not mutate gene {gene} at index {j} after {max_attempts} attempts, only have {productions_length} productions: {productions}")

        mutable_genotype[key] = gene
    return count


def test_mutation():
    genotypes, grammar = create_n_genotypes(1, 2, 2)
    print("Genotypes before mutation: ")
    print(genotypes[0])
    print("Hash: ", genotype_hash(genotypes[0]))

    mutate_genotypes_3(genotypes, grammar, 1, 1)

    print("\n\nGenotypes after mutation: ")
    print(genotypes[0])
    print("Hash: ", genotype_hash(genotypes[0]))


# In[16]:


@njit
def create_mask(length: int, crossover_probability: float) -> List[int]:
    mask = List.empty_list(types.int64)
    for _ in range(length):
        mask.append(1 if rand() < crossover_probability else 0)
    # print(mask)
    return mask


@njit
def crossover_numba(a: Dict, b: Dict, crossover_probability: float, genotype_type=new_genotype_type) -> List[Dict]:
    mask = create_mask(len(a), crossover_probability)

    child1 = a.copy()
    child2 = a.copy()

    idx = 0
    for key in a.keys():
        if mask[idx] == 1:
            child1[key] = a[key]
            child2[key] = b[key]
        else:
            child1[key] = b[key]
            child2[key] = a[key]
        idx += 1

    result = list([child1, child2])
    #print("Finished crossover")
    return result


def print_genotypes(genotypes):
    for genotype in genotypes:
        print(genotype)
    print("_______")


def test_crossover():
    n = 2
    depth = 2

    genotypes_test, grammar = create_n_genotypes(n, depth, 2)
    print("Genotypes before crossover: ")
    print_genotypes(genotypes_test[0:2])
    hash_a_before = genotype_hash(genotypes_test[0])
    hash_b_before = genotype_hash(genotypes_test[1])

    new_genotypes = crossover_numba(genotypes_test[0], genotypes_test[1], 0.5, new_genotype_type)

    print("Genotypes after crossover: ")
    print_genotypes(new_genotypes)

    hash_a_after = genotype_hash(genotypes_test[0])
    hash_b_after = genotype_hash(genotypes_test[1])

    assert_equality_of_hashes(hash_a_before, hash_a_after)
    assert_equality_of_hashes(hash_b_before, hash_b_after)


# In[17]:


@njit
def create_full_tree_from_genome(genotype, grammar, node_value_type, print_tree=False) -> Dict:
    new_nodes = Dict.empty(key_type=types.int64, value_type=node_value_type)
    create_node('<start>', new_nodes)

    load_genotype(new_nodes, genotype, grammar)

    if print_tree:
        print(simple_repr(0, new_nodes))
    return new_nodes


@njit
def create_and_eval_tree(genotype, grammar, variables, node_value_type):
    new_nodes = create_full_tree_from_genome(genotype, grammar, node_value_type, print_tree=True)
    print(evaluate(0, new_nodes, variables))


# In[18]:


@njit
def create_variable_dict(variable_names, variable_values):
    variables = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for i in range(len(variable_names)):
        variables[variable_names[i]] = variable_values[i]
    return variables


NODE_TYPE = NodeData.class_type.instance_type


def test_variable_dict():
    variable_2 = create_variable_dict(['x', 'y'], [42, 10])
    print(variable_2)


def test_evaluate(variables=None, max_depth=5):
    if variables is None:
        variables = create_variable_dict(['x', 'y'], [42, 10])
    iterations = 20
    variable_count = len(variables)
    genotypes, grammar = create_n_genotypes(iterations, max_depth, variable_count)

    for genotype in genotypes:
        create_and_eval_tree(genotype, grammar, variables, NODE_TYPE)
        print("_______")


# In[56]:


@njit
def create_variable_names(variable_count):
    variable_names = List.empty_list(types.unicode_type)
    # start at x, y,z...
    for i in range(variable_count):
        variable_names.append(chr(ord('x') + i))
    return variable_names


@njit
def calculate_fitness(variables_values, y_values, genotype, grammar, node_value_type, EMPTY_GENOTYPE, print_debug=False, print_tree=False):
    empty_genotype = EMPTY_GENOTYPE.copy()

    empty_genotype_hash = genotype_hash(EMPTY_GENOTYPE)
    new_genotype = deep_copy_genotype(genotype, empty_genotype)

    if print_debug:
        param_hash = genotype_hash(genotype)
        new_genotype_hash = genotype_hash(new_genotype)
        report = "Before tree creation: " + str(param_hash) + " and copy " + str(new_genotype_hash) + " and empty " + str(empty_genotype_hash)
        print(report)

    new_nodes = create_full_tree_from_genome(new_genotype, grammar, node_value_type)

    if print_debug:
        param_hash = genotype_hash(genotype)
        new_genotype_hash = genotype_hash(new_genotype)
        empty_genotype_hash = genotype_hash(EMPTY_GENOTYPE)

        report = "After tree creation: " + str(param_hash) + " and copy " + str(new_genotype_hash) + " and empty " + str(empty_genotype_hash)
        print(report)
    if print_tree:
        print(simple_repr(0, new_nodes))
        print("_______")

    # create variables
    variable_count = len(variables_values[0])
    variable_names = create_variable_names(variable_count)

    evaluation_cases = len(y_values)

    sum = 0
    results = List.empty_list(types.float64)
    for i in range(evaluation_cases):
        current_variable_values = []
        #print(f"Len evaluation cases: {evaluation_cases}, len variables_values: {len(variables_values)} and len variables_values[i]: {len(variables_values[i])}")
        for j in range(variable_count):
            if i >= len(variables_values):
                print("Error: i >= len(variables_values) " + str(i) + " >= " + str(len(variables_values)))
            if j >= len(variables_values[0]):
                print("Error: j >= len(variables_values[i]) " + str(j) + " >= " + str(len(variables_values[0])))
            current_variable_values.append(variables_values[i][j])


        variable_dict = create_variable_dict(variable_names, current_variable_values)
        y_predicted = evaluate(0, new_nodes, variable_dict)
        if i >= len(y_values):
            print("Error: i >= len(y_values) " + str(i) + " >= " + str(len(y_values)))
        y_correct = y_values[i]
        squared_diff = (y_predicted - y_correct) ** 2
        results.append(y_predicted)
        sum += squared_diff

    if print_debug:
        param_hash = genotype_hash(genotype)
        new_genotype_hash = genotype_hash(new_genotype)
        empty_genotype_hash = genotype_hash(EMPTY_GENOTYPE)

        report = "After fitness: " + str(param_hash) + " and copy " + str(new_genotype_hash) + " and empty " + str(empty_genotype_hash)
        print(report)
    rmse = np.sqrt(sum / evaluation_cases)
    return rmse, results


def _test_single_fitness(genotype_test, test_data, test_y, grammar, NODE_TYPE, GENOTYPE_TYPE):
    # Call the calculate_fitness function
    rmse, y_predicted_values = calculate_fitness(test_data, test_y, genotype_test, grammar, NODE_TYPE, GENOTYPE_TYPE)

    # Print the values
    for y_pred, y_corr in zip(y_predicted_values, test_y):
        print(f"y_predicted: {y_pred}, y_correct: {y_corr}")

    print(f"RMSE: {rmse}")


def test_fitness():
    test_data = np.array([[-1.23592861, -1.36410559],
                          [-0.60259712, -0.60758157],
                          [2.80419539, 2.66919459],
                          [-0.22628393, -2.97797806],
                          [2.0402239, -0.59282888]])

    test_y = np.array([6.51571868, 1.14283484, 40.67709954, 7.42636336, 9.6026114])

    genotype_test, grammar = create_genotype(max_depth=5, variables_count=2)
    _test_single_fitness(genotype_test, test_data, test_y, grammar, NODE_TYPE, GENOTYPE_TYPE)
    print("Should return equal values and hashes...")
    _test_single_fitness(genotype_test, test_data, test_y, grammar, NODE_TYPE, GENOTYPE_TYPE)


# In[20]:


@njit(parallel=True)
def calculate_all_fitnesses(genotypes: List[dict],
                            variables_values: np.ndarray,
                            y_values: np.ndarray,
                            grammar: dict,
                            node_value_type: types.Type,
                            empty_genotype: dict) -> np.ndarray:
    num_genotypes = len(genotypes)
    fitness_list = np.empty(num_genotypes, dtype=np.float64)

    for i in prange(num_genotypes):
        genotype = genotypes[i]
        rmse, _ = calculate_fitness(variables_values, y_values, genotype, grammar, node_value_type, empty_genotype)
        fitness_list[i] = rmse

    return fitness_list


def test_calculate_all_fitnesses():
    # Example usage
    genotypes, grammar = create_n_genotypes(80, 5, 2)
    test_data = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.float64)
    test_y = [1, 2]

    hash_pre_fitness = [genotype_hash(genotype) for genotype in genotypes]
    fitness_list = calculate_all_fitnesses(genotypes, test_data, test_y, grammar, NODE_TYPE, GENOTYPE_TYPE)
    hash_post_fitness = [genotype_hash(genotype) for genotype in genotypes]
    diff = 0
    print("Fitness list: ", fitness_list)

    for i in range(len(genotypes)):
        if hash_pre_fitness[i] != hash_post_fitness[i]:
            diff += 1
            print("hash pre: ", hash_pre_fitness[i], " hash post: ", hash_post_fitness[i])

    assert diff == 0, "Genotypes changed after fitness calculation"


# In[21]:


@njit(parallel=True)
def selection_tournament(fitness_list, tournament_size, maximize=False, print_groups=False):
    shuffled_indices = np.arange(len(fitness_list))
    np.random.shuffle(shuffled_indices)

    num_groups = len(fitness_list) // tournament_size
    extra_group = len(fitness_list) % tournament_size != 0

    best_elements = np.empty(num_groups, dtype=np.int64)
    group_strings = np.empty(num_groups, dtype=types.unicode_type)

    for group_idx in prange(num_groups):
        start_idx = group_idx * tournament_size
        end_idx = start_idx + tournament_size
        # account for the last partial group being smaller and unaccounted, so we merge it with the last full group
        if group_idx == num_groups - 1 and extra_group:
            end_idx = len(fitness_list)
        group_string = "["
        best_in_group = shuffled_indices[start_idx]
        group_string += str(fitness_list[best_in_group]) + " "
        for idx in range(start_idx + 1, end_idx):
            correct_index = shuffled_indices[idx]
            group_string += str(fitness_list[correct_index]) + " "
            if maximize:
                if fitness_list[correct_index] > fitness_list[best_in_group]:
                    best_in_group = shuffled_indices[idx]
            else:
                if fitness_list[correct_index] < fitness_list[best_in_group]:
                    best_in_group = shuffled_indices[idx]
        group_string += "]"
        group_strings[group_idx] = group_string
        best_elements[group_idx] = best_in_group

    if print_groups:
        for group_string in group_strings:
            print(group_string)
    return best_elements


@njit(parallel=True)
def best_n_items(fitness_list, n, maximize=False):
    if maximize:
        sorted_indices = np.argsort(-fitness_list)
    else:
        sorted_indices = np.argsort(fitness_list)
    return sorted_indices[:n]


def test_selection_tournament():
    fitness_list = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91])
    tournament_size = 6
    best_elements = selection_tournament(fitness_list, tournament_size, maximize=False, print_groups=True)

    # Get the value of best elements
    best_elements_selection = [fitness_list[idx] for idx in best_elements]
    print("Best elements: ", best_elements_selection)
    best_itens = best_n_items(fitness_list, 3, maximize=False)

    best_elements_global = [fitness_list[idx] for idx in best_itens]
    print("Global best elements: ", best_elements_global)


# In[22]:
def print_resumed_genotype(genotype: Dict):
    # calculate an unique hash for the genotype
    hash_genotype = genotype_hash(genotype)

    print(f"{hash_genotype}: ______")
    keys = list(genotype.keys())
    alphabetical_keys = sorted(keys)
    for key in alphabetical_keys:
        value = genotype[key]
        real_len = min(len(value), 4)
        print(f"{key}: {value[:real_len]}", end=" ")


if __name__ == '__main__':


    print("\n\nTESTING GRAMMAR RELATED __________")
    test_grammar_creation()
    test_terminal_and_nonterminal()

    print("\n\nTESTING REFERENCE COUNTING RELATED __________")

    test_reference_counting()
    test_reference_counting('<expr>')

    print("\n\nTESTING GENOTYPE CREATION RELATED __________")

    test_create_genotype()

    test_print_tree_and_evaluate()

    test_n_creation()
    print("\n\nTESTING CROSSOVER AND MUTATION __________")

    test_mutation()
    test_crossover()
    print("\n\nTESTING EVALUATE RELATED __________")

    test_variable_dict()
    test_evaluate()
    print("\n\nTESTING FITNESS RELATED __________")

    test_fitness()
    test_calculate_all_fitnesses()

    print("\n\nTESTING SELECTION RELATED __________")

    test_selection_tournament()
