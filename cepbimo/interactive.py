"""Module to simplify code output in Jupyter notebooks."""
from utils import is_notebook

if is_notebook():
    from IPython.core.interactiveshell import InteractiveShell

    InteractiveShell.ast_node_interactivity = 'all'
