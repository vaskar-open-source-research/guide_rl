qwen_math_chat_template="<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>user\nProblem: {prompt}<|im_end|>\n<|im_start|>assistant\n"

llama_math_chat_template = """<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: {prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"""


if __name__ == "__main__":
    print(qwen_math_chat_template.format(prompt="Find an integer congruent to $1 \pmod{7}$ and to $2 \pmod{11}$."))
    print(llama_math_chat_template.format(prompt="Find an integer congruent to $1 \pmod{7}$ and to $2 \pmod{11}$."))