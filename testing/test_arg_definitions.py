import ast
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _dict_literal_keys(node):
    """Extracts keys from an ast.Dict node (static literals only)."""
    keys = []
    for k in node.keys:
        if isinstance(k, ast.Constant):
            keys.append(k.value)
        else:
            raise AssertionError(
                "argument_factory must only contain string literal keys"
            )
    return keys


def test_argument_factory_has_unique_keys():
    """
    Guards against silent key collisions in the argparse factory: duplicate
    dict keys collapse at parse time and silently drop CLI arguments.
    """
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "adell_mri",
        "entrypoints",
        "assemble_args.py",
    )
    tree = ast.parse(open(path).read())
    found = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "argument_factory"
        ):
            found = node.value
            break
    assert found is not None, "argument_factory not found"
    assert isinstance(found, ast.Dict), "argument_factory must be a dict"
    keys = _dict_literal_keys(found)
    duplicates = {k for k in keys if keys.count(k) > 1}
    assert not duplicates, f"Duplicate argument definitions: {duplicates}"
