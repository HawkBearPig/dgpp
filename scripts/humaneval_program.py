"""Assemble HumanEval's supplied context, model answer and tests."""


def humaneval_program(item, code):
    """Keep supplied helpers/imports even when the answer is a full function.

    Full modules execute in the prompt's namespace as separate code units,
    so a response's future imports remain legal. Body-only completions still
    append to the prompt's unfinished function, as in the original harness.
    """
    try:
        compile(code, "<answer>", "exec")
    except SyntaxError:
        answer = item["prompt"] + code
    else:
        answer = f"exec({item['prompt']!r})\nexec({code!r})"
    return answer + "\n\n" + item["test"] + "\n\n" + f"check({item['entry_point']})\n"
