"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union

from graders.verl_grader.reward_score.deepscaler.globals import THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END, OAI_RM_MODEL
from graders.verl_grader.reward_score.deepscaler.rewards import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from graders.verl_grader.reward_score.deepscaler.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd
from graders.verl_grader.reward_score.deepscaler.system_prompts import ORM_PROMPT
from graders.verl_grader.reward_score.deepscaler.utils import call_oai_rm_llm

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""

class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # Extract solution.
        # if THOUGHT_DELIMITER_START in model_response and THOUGHT_DELIMITER_END in model_response:
        #     model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        # else:
        #     return RewardOutput(reward=self.config.format_error_reward, is_correct=False)
        model_solution = model_response
        
        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)
        
        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]
            
        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)
        
        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        # If latex heuristics fail and ORM is enabled, use LLM as ORM to evaluate correctness
        if self.config.use_math_orm:
            for ground_truth in processed_ground_truths:
                try:
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                    )

                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                except Exception as e:
                    print ("Error calling Gemini ORM, trying OAI RM")
                    orm_response = call_oai_rm_llm(
                        system_prompt=ORM_PROMPT,
                        prompt=ORM_USER_TEMPLATE.format(problem=problem, answer_1=model_answer, answer_2=ground_truth),
                        temperature=0.0,
                        model_id=OAI_RM_MODEL,
                    )
                    
                    if "[[YES]]" in orm_response:
                        return RewardOutput(reward=self.config.correct_reward, is_correct=True)
                    continue
                
        return RewardOutput(reward=self.config.incorrect_reward, is_correct=False)

def deepscaler_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": ground_truth}))
    return reward_response.is_correct

if __name__ == "__main__":
    print("Testing math reward function")
    reward = RewardMathFn(RewardConfig)
    print("Running reward function")
    input = RewardInput(problem="hello", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$", "a"]})
    output = reward(input)
    print(output)
