import re

def verify_luffy_math_format(model_output: str) -> bool:
    """
    Verify if the model output is in the correct format.
    """

    # make sure there are no more than 3 white spaces at the beginning or end of the string
    start_white_space_count = 0
    end_white_space_count = 0

    # check for any code execution blocks
    # code_block_count = model_output.count("```")
    # if code_block_count != 0:
    #     return False

    # check for exactly one boxed answer
    boxed_answer_count = model_output.count("boxed")
    if boxed_answer_count != 1:
        return False

    for i, c in enumerate(model_output):
        if c == " ":
            start_white_space_count += 1
        else:
            break

    for i, c in enumerate(model_output[::-1]):
        if c == " ":
            end_white_space_count += 1
        else:
            break
    
    if start_white_space_count > 3 or end_white_space_count > 3:
        return False

    # check that <think> is the first in the string
    # if not model_output.strip().startswith("<think>"):
    #     return False
    
    tags = ["</think>"]

    # check exactly one of each tag is present
    for tag in tags:
        if model_output.count(tag) != 1:
            return False

    # # check that the order of the tags is correct
    # for i in range(1, len(tags)):
    #     if model_output.find(tags[i]) < model_output.find(tags[i-1]):
    #         return False

    return True

def verify_math_format(model_output: str) -> bool:
    """
    Verify if the model output is in the correct format.
    """

    # make sure there are no more than 3 white spaces at the beginning or end of the string
    start_white_space_count = 0
    end_white_space_count = 0

    # check for exactly one boxed answer
    boxed_answer_count = model_output.count("boxed")
    if boxed_answer_count != 1:
        return False

    for i, c in enumerate(model_output):
        if c == " ":
            start_white_space_count += 1
        else:
            break

    for i, c in enumerate(model_output[::-1]):
        if c == " ":
            end_white_space_count += 1
        else:
            break
    
    if start_white_space_count > 3 or end_white_space_count > 3:
        return False

    # check that <think> is the first in the string
    if not model_output.strip().startswith("<think>"):
        return False
        
    # check that <answer> is the last in the string
    if not model_output.strip().endswith("</answer>"):
        return False
    
    tags = ["<think>", "</think>", "<answer>", "</answer>"]

    # check exactly one of each tag is present
    for tag in tags:
        if model_output.count(tag) != 1:
            return False

    think_end = model_output.find("</think>")
    answer_start = model_output.find("<answer>")

    # Extract the string between </think> and <answer>
    between_tags = model_output[think_end + len("</think>"):answer_start].strip()

    # Check that it contains only whitespace/newlines and that it's not longer than 5 characters
    if len(between_tags) > 5 or not re.match(r'^[ \n]*$', between_tags):
        return False

    # check that the order of the tags is correct
    for i in range(1, len(tags)):
        if model_output.find(tags[i]) < model_output.find(tags[i-1]):
            return False

    return True

def verify_code_format(model_output: str) -> bool:
    """
    Verify if the model output is in the correct format.
    """

    # make sure there are no more than 3 white spaces at the beginning or end of the string
    start_white_space_count = 0
    end_white_space_count = 0

    # check for ```python ... ```
    python_begin_count = model_output.count("```python")
    # check that there is exactly one ``` after ```python
    python_end_count = model_output.count("```", model_output.find("```python") + len("```python"))
    if python_begin_count != 1 or python_end_count != 1:
        return False

    for i, c in enumerate(model_output):
        if c == " ":
            start_white_space_count += 1
        else:
            break

    for i, c in enumerate(model_output[::-1]):
        if c == " ":
            end_white_space_count += 1
        else:
            break
    
    if start_white_space_count > 3 or end_white_space_count > 3:
        return False

    # check that <think> is the first in the string
    if not model_output.strip().startswith("<think>"):
        return False
        
    # check that <answer> is the last in the string
    if not model_output.strip().endswith("</answer>"):
        return False
    
    tags = ["<think>", "</think>", "<answer>", "</answer>"]

    # check exactly one of each tag is present
    for tag in tags:
        if model_output.count(tag) != 1:
            return False

    think_end = model_output.find("</think>")
    answer_start = model_output.find("<answer>")

    # Extract the string between </think> and <answer>
    between_tags = model_output[think_end + len("</think>"):answer_start].strip()

    # Check that it contains only whitespace/newlines and that it's not longer than 5 characters
    if len(between_tags) > 5 or not re.match(r'^[ \n]*$', between_tags):
        return False

    # check that the order of the tags is correct
    for i in range(1, len(tags)):
        if model_output.find(tags[i]) < model_output.find(tags[i-1]):
            return False

    return True

def verify_llama_simple_format(model_output: str) -> bool:
    boxed_answer_count = model_output.count("\\boxed{")
    if boxed_answer_count != 1:
        return False
    
    last_line = model_output.strip().split('\n')[-1]
    if last_line.count("\\boxed{") != 1:
        return False
    
    count_trail_whitespace = 0
    for i in range(len(model_output)-1, -1, -1):
        if model_output[i] == " " or model_output[i] == "\n":
            count_trail_whitespace += 1
        else:
            break

    if count_trail_whitespace > 3:
        return False

    return True

def verify_qwen_simple_format(model_output: str) -> bool:
    boxed_answer_count = model_output.count("\\boxed{")
    if boxed_answer_count != 1:
        return False
    
    count_trail_whitespace = 0
    for i in range(len(model_output)-1, -1, -1):
        if model_output[i] == " " or model_output[i] == "\n":
            count_trail_whitespace += 1
        else:
            break

    if count_trail_whitespace > 3:
        return False

    return True