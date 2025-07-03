from fdtdx.core.jax.pytrees import TreeClass


def test_string_parsing():
    fn = TreeClass._parse_operations

    valid_cases = [
        "a->b->c->[0]->['name']",
        "obj->attr->[123]->['simple key']",
        "x->[0]->[1]->y->['test']",
        "single",
        "a->b->[-5]->['no special chars']",
        "data->['hello world']->result",
    ]

    print("=== Valid cases ===")
    for test in valid_cases:
        try:
            result = fn(test)
            print(f"\nInput: {test}")
            print("Parsed operations:")
            for op, op_type in result:
                print(f"  {repr(op)} ({op_type})")
        except ValueError as e:
            print(f"\nInput: {test}")
            print(f"Unexpected error: {e}")

    # Test error cases
    print("\n\n=== Error cases ===")
    error_cases = [
        ("a->", "ends with arrow"),
        ("->b", "starts with arrow"),
        ("a->[", "unclosed bracket"),
        ("a->[invalid]", "invalid bracket content"),
        ("a->123invalid", "invalid attribute name"),
        ("a->b->[]", "empty brackets"),
        ("", "empty string"),
        ("a->['string with [brackets] inside']", "brackets inside string"),
        ("a->['can't use quotes']", "quotes inside string"),
        ("a->['escaped \\'quote\\'']", "escaped quotes not allowed"),
        ("a->[']", "incomplete string"),
        ("a->['", "incomplete string"),
        ("a->['multiple' 'strings']", "invalid format"),
    ]

    for test, description in error_cases:
        try:
            result = fn(test)
            print(f"\nInput: {test} ({description})")
            print(f"ERROR: Unexpected success: {result}")
        except ValueError as e:
            print(f"\nInput: {test} ({description})")
            print(f"Expected error: {e}")
