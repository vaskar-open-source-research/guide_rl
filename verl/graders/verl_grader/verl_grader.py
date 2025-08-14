import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List, Union

from tqdm import tqdm

from graders.verl_grader.reward_score import _default_compute_score
from utils.common import flatten_list_strict, group_list_elements, repeat_list_elements
from utils.parallel import map_parallel_timeout_pebble
from utils.profiler import profiler_decorator


class VERLGrader:
    """
    math and coding grader from verl/prime
    """

    def __init__(self):
        pass

    def _grade(
        self,
        prediction: Union[str, float],
        ground_truth: Any,
        data_source: str,
        return_meta_data: bool = False,
        strict: bool = False,
    ) -> Union[dict, float]:
        result = {
            "result": 0.0,  # default score
            "meta_data": {},  # metadata about the result
        }
        try:
            result["result"] = _default_compute_score(
                data_source, prediction, ground_truth, strict
            )
        except Exception as e:
            result["meta_data"]["exception"] = "Exception occurred: " + str(e)
            print(e)
        return result if return_meta_data else result["result"]

    @profiler_decorator
    def grade(
        self,
        prediction: Union[str, float],
        ground_truth: Any,
        data_source: str,
        return_meta_data: bool = False,
    ) -> Union[dict, float]:
        result = self._grade(
            prediction=prediction,
            ground_truth=ground_truth,
            data_source=data_source,
            return_meta_data=return_meta_data,
        )

        return result

    @profiler_decorator
    def grade_batch(
        self,
        prediction: List[Union[str, float]],
        ground_truth: List[Any],
        data_source: List[str],
        return_meta_data: bool = False,
        num_workers: int = 16,
        timeouts: Union[List[float], float] = None,
        strict: bool = False
    ) -> List[Union[dict, float]]:
        """
        grade a batch of samples in using multiprocessing.
        example:
        prediction: ["pred1", "pred2", "pred3", ...]
        ground_truth: ["ans1", "ans2", "ans3", ...]

        returns: [res1, res2, res3, ...]
        """
        assert len(prediction) == len(
            ground_truth
        ), "prediction and ground_truth must have the same length!"
        assert len(prediction) == len(
            data_source
        ), "prediction and data_source must have the same length!"

        if isinstance(timeouts, list):
            assert len(prediction) == len(
                timeouts
            ), "prediction and timeouts must have the same length!"
        elif isinstance(timeouts, float):
            timeouts = [timeouts] * len(prediction)
        else:
            raise ValueError(f"Invalid timeouts: {timeouts}")

        # repeat args for each sample
        _return_meta_data = [return_meta_data] * len(prediction)

        math_or_stem_indices = [i for i, d in enumerate(data_source) if d == "math" or d == "stem" or d == "deepscaler"]
        code_indices = [i for i, d in enumerate(data_source) if d == "code"]

        math_or_stem_prediction = [prediction[i] for i in math_or_stem_indices]
        math_or_stem_ground_truth = [ground_truth[i] for i in math_or_stem_indices]
        math_or_stem_data_source = [data_source[i] for i in math_or_stem_indices]
        math_or_stem_timeouts = [timeouts[i] for i in math_or_stem_indices]

        math_or_stem_results = map_parallel_timeout_pebble(
            func=self._grade,
            args_list=zip(
                math_or_stem_prediction,
                math_or_stem_ground_truth,
                math_or_stem_data_source,
                _return_meta_data,
            ),
            num_workers=num_workers,
            timeouts=math_or_stem_timeouts,
            default_return_value=0.0,
        )

        code_prediction = [prediction[i] for i in code_indices]
        code_ground_truth = [ground_truth[i] for i in code_indices]
        code_data_source = [data_source[i] for i in code_indices]

        code_results = []
        if code_indices:  # Only run if there are code examples to evaluate
            code_results = self._grade_code_parallel(
                code_prediction,
                code_ground_truth,
                code_data_source,
                [return_meta_data] * len(code_prediction),
                num_workers=num_workers,
                strict=strict,
            )

        results = [0.0] * len(prediction)
        for i, d in enumerate(math_or_stem_indices):
            results[d] = math_or_stem_results[i]
        for i, d in enumerate(code_indices):
            results[d] = code_results[i]

        return results

    @profiler_decorator
    def grade_multiple_samples_batch(
        self,
        prediction: List[List[Union[str, float]]],
        ground_truth: List[Any],
        data_source: str,
        return_meta_data: bool = False,
        num_workers: int = 16,
        max_batch_size: int = 32768,
    ) -> List[List[Union[dict, float]]]:
        """
        similar to grade_batch but for multiple samples per prompt.
        example:
        prediction: [["pred11", "pred12", ... ], ["pred21", "pred22", ... ], ["pred31", "pred32", ... ]]
        correct_answer: ["ans1", "ans2", "ans3", ...]

        returns: [[res11, res12, ...], [res21, res22, ...], [res31, res32, ...]]
        """
        n = len(prediction[0])  # num samples per prompt
        prediction = flatten_list_strict(prediction, n)
        ground_truth = repeat_list_elements(ground_truth, n)

        results = self.grade_batch(
            prediction=prediction,
            ground_truth=ground_truth,
            data_source=data_source,
            return_meta_data=return_meta_data,
            num_workers=num_workers,
            max_batch_size=max_batch_size,
        )

        return group_list_elements(results, n)

    def _grade_with_progress(self, idx, pred, gt, ds, meta):
        try:
            result = self._grade(pred, gt, ds, meta)
        except Exception as e:
            result = (
                0.0
                if not meta
                else {"result": 0.0, "meta_data": {"exception": f"Exception: {str(e)}"}}
            )

            result = (
                0.0
                if not meta
                else {"result": 0.0, "meta_data": {"exception": f"Exception: {str(e)}"}}
            )

        return idx, result

    def _grade_code_parallel(
        self,
        predictions: List[Union[str, float]],
        ground_truths: List[Any],
        data_sources: List[str],
        return_meta_data_list: List[bool],
        num_workers: int = 16,
        strict: bool = False,
    ) -> List[Union[dict, float]]:
        """
        Direct thread pool implementation for code grading that runs multiple evaluations in parallel
        with a maximum number of concurrent workers
        """
        # Create a progress bar
        pbar = tqdm(total=len(predictions), desc="Grading code", position=0, leave=True)
        results = [None] * len(predictions)

        try:
            # Create a thread pool and submit all grading tasks
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks without progress updates
                futures_to_index = {}
                for idx, (pred, gt, ds, meta) in enumerate(
                    zip(predictions, ground_truths, data_sources, return_meta_data_list)
                ):
                    future = executor.submit(self._grade, pred, gt, ds, meta, strict)
                    futures_to_index[future] = idx

                # Collect results as they complete
                for future in concurrent.futures.as_completed(futures_to_index):
                    idx = futures_to_index[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as e:
                        # Handle any exceptions that might occur
                        if return_meta_data_list[idx]:
                            results[idx] = {
                                "result": 0.0,
                                "meta_data": {"exception": f"Exception: {str(e)}"},
                            }
                            results[idx] = {
                                "result": 0.0,
                                "meta_data": {"exception": f"Exception: {str(e)}"},
                            }
                        else:
                            results[idx] = 0.0
                    # Update progress in main thread
                    pbar.update(1)

            return results
        finally:
            # Close the progress bar
            pbar.close()


if __name__ == "__main__":
    # from utils.common import setup_logging
    import time

    from utils.profiler import profiler

    # setup_logging()

    grader = VERLGrader()

    num_workers = 4
    num_repeat = 2

    num_samples = 10

    # data = [
    #     {"prediction": "123", "ground_truth": "123"},
    #     {"prediction": r"\text{Evelyn}", "ground_truth": "Evelyn"},
    #     {"prediction": r"\frac{28}{6}", "ground_truth": r"\frac{14}{3}"},
    #     {"prediction": r"$14.0/3$", "ground_truth": r"\frac{14}{3}"},
    #     {"prediction": r"$20\sqrt{3 * 1}$ ", "ground_truth": r"20\sqrt{3}"},
    #     {"prediction": r"3, 5, 7", "ground_truth": r"$\{3, 5, 7\}$"},
    # ] * num_repeat

    # prediction = [d["prediction"] for d in data]
    # ground_truth = [d["ground_truth"] for d in data]

    # results_iter = []
    # for i in range(len(prediction)):
    #     results_iter.append(
    #         grader.grade(
    #             prediction=prediction[i],
    #             ground_truth=ground_truth[i],
    #             data_source="numina_aops_forum",
    #         )
    #     )

    # results_batch = grader.grade_batch(
    #     prediction=prediction,
    #     ground_truth=ground_truth,
    #     data_source=["math"],
    #     num_workers=num_workers,
    # )

    # samples = [[d["prediction"]] * num_samples for d in data]
    # results_all_samples = grader.grade_multiple_samples_batch(
    #     prediction=samples,
    #     ground_truth=ground_truth,
    #     data_source="numina_aops_forum",
    #     num_workers=num_workers,
    # )

    # profiler.print_stats()

    # for i, d in enumerate(data):
    #     print("-" * 60)
    #     print(f"prediction: {d['prediction']}, ground_truth: {d['ground_truth']}")
    #     print(results_iter[i])
    #     print(results_batch[i])
    #     print(results_all_samples[i])
    # for i, d in enumerate(data):
    #     print("-" * 60)
    #     print(f"prediction: {d['prediction']}, ground_truth: {d['ground_truth']}")
    #     print(results_iter[i])
    #     print(results_batch[i])
    #     print(results_all_samples[i])

    # # test code grader
    # data = [
    #     {
    #         "prediction": "<think> \nTo solve this problem, we need to calculate the permutation difference between the strings `s` and `t`. The permutation difference is defined as the sum of the absolute differences between the indices of each character in `s` and `t`.\n\n1. First, we need to map each character in `s` to its index.\n2. Then, we need to map each character in `t` to its index.\n3. Finally, we calculate the permutation difference by summing the absolute differences between the indices of each character in `s` and `t`.\n\nLet's implement this step-by-step.\n</think>\n\n<answer>\nTo solve the problem, we can follow these steps:\n\n1. Create a dictionary to map each character in `s` to its index.\n2. Create another dictionary to map each character in `t` to its index.\n3. Iterate through the characters in `s` and calculate the absolute difference between the indices of each character in `s` and `t`.\n4. Sum these absolute differences to get the permutation difference.\n\nHere is the Python code to implement this:\n\n```python\ndef permutation_difference(s, t):\n    # Create a dictionary to map each character in s to its index\n    s_indices = {char: idx for idx, char in enumerate(s)}\n    \n    # Create a dictionary to map each character in t to its index\n    t_indices = {char: idx for idx, char in enumerate(t)}\n    \n    # Calculate the permutation difference\n    permutation_diff = sum(abs(s_indices[char] - t_indices[char]) for char in s)\n    \n    return permutation_diff\n\n# Example usage\ns = input()\nt = input()\nprint(permutation_difference(s, t))```\n\nThis code first creates two dictionaries to map each character in `s` and `t` to their respective indices. Then, it calculates the permutation difference by summing the absolute differences between the indices of each character in `s` and `t`. The final result is returned as the permutation difference.\n</answer>",
    #         "ground_truth": '{\n  "inputs": [\n    "\\"abc\\"\\n\\"bac\\"",\n    "\\"abcde\\"\\n\\"edbac\\"",\n    "\\"rwohu\\"\\n\\"rwuoh\\"",\n    "\\"cdrsjkbo\\"\\n\\"dcrsjkbo\\"",\n    "\\"rcntxajvbpukdowieghz\\"\\n\\"zhgeiwodkupbvjaxtncr\\"",\n    "\\"kml\\"\\n\\"lmk\\"",\n    "\\"trckompbhgefysw\\"\\n\\"wsyfeghbpmokcrt\\"",\n    "\\"abcdefghijklmnopqrstuvwx\\"\\n\\"xwvutsrqponmlkjihgfedcba\\"",\n    "\\"yirmpv\\"\\n\\"vpmriy\\"",\n    "\\"nlhpdyvzjburgiaoxtqcsmwekf\\"\\n\\"tegkhubsimnqjpdawofylvrczx\\"",\n    "\\"pkdgjmwfvqozsyiurabtxlench\\"\\n\\"fivlaekjdxcqghoswpzunrtybm\\"",\n    "\\"oupvimskbhlajtcezfnyqgrxwd\\"\\n\\"cgzmhpeonjklwrtqidbfuxavys\\"",\n    "\\"ncvweofspxgztdi\\"\\n\\"cifsozvtwdxpnge\\"",\n    "\\"oenlsyigjwuv\\"\\n\\"oiegnjlwsuyv\\"",\n    "\\"twzyhropbgkxlqnmivjdefsuac\\"\\n\\"mrgfnpxtyukwjqioablszevhdc\\"",\n    "\\"cdrsjkbo\\"\\n\\"odsrkjbc\\"",\n    "\\"abcdefghijklmnopqrstuvwxyz\\"\\n\\"zyxwvutsrqponmlkjihgfedcba\\"",\n    "\\"jlmrhnpgikdatxyczwbsoeqvfu\\"\\n\\"kvlyeshcnjxbatqpdmifwzrgou\\"",\n    "\\"qtjvmuhzaeyxrfbd\\"\\n\\"dbfrxyeazhumvjtq\\"",\n    "\\"iydezrxlqvn\\"\\n\\"xldzvryniqe\\"",\n    "\\"hsavuydxbtpflqnmzkorgeiwjc\\"\\n\\"iawqgjuschdplfozebkvmyntrx\\"",\n    "\\"gadlqobvtyj\\"\\n\\"jytvboqldag\\"",\n    "\\"onkgrjvabqdis\\"\\n\\"srdinvkjgoaqb\\"",\n    "\\"abcdefgh\\"\\n\\"bcdefgha\\"",\n    "\\"dtcyezqiaolfs\\"\\n\\"sfloaiqzeyctd\\"",\n    "\\"sdnrbuxomwhgcklipeztajvfqy\\"\\n\\"ncztwxiqhypuksfdvblaejrgom\\"",\n    "\\"akhpirxcysoeqdufbznjltvgmw\\"\\n\\"ycwqvipmzfsbhjexnlukdtorga\\""\n  ],\n  "outputs": [\n    "26",\n    "12",\n    "4",\n    "2",\n    "200",\n    "4",\n    "112",\n    "288",\n    "18",\n    "256",\n    "228",\n    "226",\n    "76",\n    "30",\n    "188",\n    "18",\n    "338",\n    "208",\n    "128",\n    "42",\n    "232",\n    "60",\n    "64",\n    "14",\n    "84",\n    "236",\n    "224"\n  ]\n}',
    #     },
    # ]

    # prediction = [d["prediction"] for d in data]
    # ground_truth = [d["ground_truth"] for d in data]

    # results_batch = grader.grade_batch(
    #     prediction=prediction,
    #     ground_truth=ground_truth,
    #     data_source=["code"],
    #     num_workers=num_workers,
    #     timeouts=300.0,
    # )

    # data = [
    #     {
    #         "prediction": "while True:\n    pass",
    #         "ground_truth": {"inputs": ["", ""], "outputs": ["", ""]},
    #     }
    # ]

    # prediction = [d["prediction"] for d in data]
    # ground_truth = [d["ground_truth"] for d in data]

    # t1 = time.time()
    # results_batch = grader.grade_batch(
    #     prediction=prediction,
    #     ground_truth=ground_truth,
    #     data_source=["code"],
    #     num_workers=num_workers,
    #     timeouts=300.0,
    # )
    # t2 = time.time()
    # print(f"Time taken with for infinite loop with 2 test cases: {t2 - t1} seconds")

    # print(results_batch)

    # data = [
    #     {
    #         "prediction": "while True:\n    pass",
    #         "ground_truth": {"inputs": ["", "", ""], "outputs": ["", "", ""]},
    #     }
    # ]

    # prediction = [d["prediction"] for d in data]
    # ground_truth = [d["ground_truth"] for d in data]

    # t1 = time.time()
    # results_batch = grader.grade_batch(
    #     prediction=prediction,
    #     ground_truth=ground_truth,
    #     data_source=["code"],
    #     num_workers=num_workers,
    #     timeouts=300.0,
    # )
    # t2 = time.time()
    # print(f"Time taken with for infinite loop with 3 test cases: {t2 - t1} seconds")

    # print(results_batch)

    # data = [
    #     {
    #         "prediction": "while True:\n    pass",
    #         "ground_truth": {"inputs": ["", ""], "outputs": ["", ""]},
    #     }
    # ]

    # multiplier = 10000
    # multidata = [data] * multiplier
    # multiground_truth = [ground_truth] * multiplier
    # multidata_source = ["code"] * multiplier

    # t1 = time.time()
    # results_batch = grader.grade_batch(
    #     prediction=multidata,
    #     ground_truth=multiground_truth,
    #     data_source=multidata_source,
    #     num_workers=20,
    #     timeouts=300.0,
    # )
    # t2 = time.time()
    # print(f"Time taken with 64 workers: {t2 - t1} seconds")

    # t1 = time.time()
    # results_batch = grader.grade_batch(
    #     prediction=multidata,
    #     ground_truth=multiground_truth,
    #     data_source=multidata_source,
    #     num_workers=16,
    #     timeouts=300.0,
    # )
    # t2 = time.time()
    # print(f"Time taken with 16 workers: {t2 - t1} seconds")

    # t1 = time.time()
    # results_batch = grader.grade_batch(
    #     prediction=multidata,
    #     ground_truth=multiground_truth,
    #     data_source=multidata_source,
    #     num_workers=1,
    #     timeouts=300.0,
    # )
    # t2 = time.time()
    # print(f"Time taken with 1 worker: {t2 - t1} seconds")

    # data = [
    #     {
    #         "prediction": "while True:\n    pass",
    #         "ground_truth": {"inputs": ["", ""], "outputs": ["", ""]},
    #     }
    # ]

    # multiplier = 8
    # multidata = data * multiplier

    # prediction = [d["prediction"] for d in multidata]
    # ground_truth = [d["ground_truth"] for d in multidata]

    # t1 = time.time()
    # results_batch = grader.grade_batch(
    #     prediction=prediction,
    #     ground_truth=ground_truth,
    #     data_source=["code"] * multiplier,
    #     num_workers=8,
    #     timeouts=300.0,
    # )
    # t2 = time.time()
    # print(f"Time taken with 2 workers: {t2 - t1} seconds")

    # t1 = time.time()
    # results_batch = grader.grade_batch(
    #     prediction=prediction,
    #     ground_truth=ground_truth,
    #     data_source=["code"] * multiplier,
    #     num_workers=1,
    #     timeouts=300.0,
    # )
    # t2 = time.time()
    # print(f"Time taken with 1 workers: {t2 - t1} seconds")

    # # test code grader
    # data = [
    #     {
    #         "prediction": "<think> \nTo solve this problem, we need to calculate the permutation difference between the strings `s` and `t`. The permutation difference is defined as the sum of the absolute differences between the indices of each character in `s` and `t`.\n\n1. First, we need to map each character in `s` to its index.\n2. Then, we need to map each character in `t` to its index.\n3. Finally, we calculate the permutation difference by summing the absolute differences between the indices of each character in `s` and `t`.\n\nLet's implement this step-by-step.\n</think>\n\n<answer>\nTo solve the problem, we can follow these steps:\n\n1. Create a dictionary to map each character in `s` to its index.\n2. Create another dictionary to map each character in `t` to its index.\n3. Iterate through the characters in `s` and calculate the absolute difference between the indices of each character in `s` and `t`.\n4. Sum these absolute differences to get the permutation difference.\n\nHere is the Python code to implement this:\n\n```python\ndef permutation_difference(s, t):\n    # Create a dictionary to map each character in s to its index\n    s_indices = {char: idx for idx, char in enumerate(s)}\n    \n    # Create a dictionary to map each character in t to its index\n    t_indices = {char: idx for idx, char in enumerate(t)}\n    \n    # Calculate the permutation difference\n    permutation_diff = sum(abs(s_indices[char] - t_indices[char]) for char in s)\n    \n    return permutation_diff\n\n# Example usage\ns = input()\nt = input()\nprint(permutation_difference(s, t))\nassert permutation_difference('abc', 'bac') == 2```\n\nThis code first creates two dictionaries to map each character in `s` and `t` to their respective indices. Then, it calculates the permutation difference by summing the absolute differences between the indices of each character in `s` and `t`. The final result is returned as the permutation difference.\n</answer>",
    #         "ground_truth": '{\n  "inputs": [" \n "], "outputs": [""]}',
    #     },
    # ]

    # # preprocess data to remove input lines
    # lines = data[0]["prediction"].split("\n")
    # lines = [line for line in lines if 'input()' not in line]
    # print(lines)
    # data[0]["prediction"] = "\n".join(lines)

    # prediction = [d["prediction"] for d in data]
    # ground_truth = [d["ground_truth"] for d in data]

    # results_batch = grader.grade_batch(
    #     prediction=prediction,
    #     ground_truth=ground_truth,
    #     data_source=["code"],
    #     num_workers=num_workers,
    #     timeouts=300.0,
    # )

    # print(results_batch)


    # test math grader
    # model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$", "a"]})
    data = [
        {
            "prediction": "<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.",
            "ground_truth": "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$",
        },
    ]

    # preprocess data to remove input lines
    lines = data[0]["prediction"].split("\n")
    lines = [line for line in lines if 'input()' not in line]
    print(lines)
    data[0]["prediction"] = "\n".join(lines)

    prediction = [d["prediction"] for d in data]
    ground_truth = [d["ground_truth"] for d in data]

    results_batch = grader.grade_batch(
        prediction=prediction,
        ground_truth=ground_truth,
        data_source=["math"],
        num_workers=num_workers,
        timeouts=300.0,
    )

    print(results_batch)

    data = [
        {
            "prediction": "while True:\n    pass",
            "ground_truth": {"inputs": ["", ""], "outputs": ["", ""]},
        }
    ]

    multiplier = 10000
    multidata = [data] * multiplier
    multiground_truth = [ground_truth] * multiplier
    multidata_source = ["code"] * multiplier

    t1 = time.time()
    results_batch = grader.grade_batch(
        prediction=multidata,
        ground_truth=multiground_truth,
        data_source=multidata_source,
        num_workers=20,
        timeouts=300.0,
    )
    t2 = time.time()
    print(f"Time taken with 64 workers: {t2 - t1} seconds")

    t1 = time.time()
    results_batch = grader.grade_batch(
        prediction=multidata,
        ground_truth=multiground_truth,
        data_source=multidata_source,
        num_workers=16,
        timeouts=300.0,
    )
    t2 = time.time()
    print(f"Time taken with 16 workers: {t2 - t1} seconds")

    t1 = time.time()
    results_batch = grader.grade_batch(
        prediction=multidata,
        ground_truth=multiground_truth,
        data_source=multidata_source,
        num_workers=1,
        timeouts=300.0,
    )
    t2 = time.time()
    print(f"Time taken with 1 worker: {t2 - t1} seconds")

    data = [
        {
            "prediction": "while True:\n    pass",
            "ground_truth": {"inputs": ["", ""], "outputs": ["", ""]},
        }
    ]

    multiplier = 8
    multidata = data * multiplier

    prediction = [d["prediction"] for d in multidata]
    ground_truth = [d["ground_truth"] for d in multidata]

    t1 = time.time()
    results_batch = grader.grade_batch(
        prediction=prediction,
        ground_truth=ground_truth,
        data_source=["code"] * multiplier,
        num_workers=8,
        timeouts=300.0,
    )
    t2 = time.time()
    print(f"Time taken with 2 workers: {t2 - t1} seconds")

    t1 = time.time()
    results_batch = grader.grade_batch(
        prediction=prediction,
        ground_truth=ground_truth,
        data_source=["code"] * multiplier,
        num_workers=1,
        timeouts=300.0,
    )
    t2 = time.time()
    print(f"Time taken with 1 workers: {t2 - t1} seconds")

    # test code grader
    data = [
        {
            "prediction": "<think> \nTo solve this problem, we need to calculate the permutation difference between the strings `s` and `t`. The permutation difference is defined as the sum of the absolute differences between the indices of each character in `s` and `t`.\n\n1. First, we need to map each character in `s` to its index.\n2. Then, we need to map each character in `t` to its index.\n3. Finally, we calculate the permutation difference by summing the absolute differences between the indices of each character in `s` and `t`.\n\nLet's implement this step-by-step.\n</think>\n\n<answer>\nTo solve the problem, we can follow these steps:\n\n1. Create a dictionary to map each character in `s` to its index.\n2. Create another dictionary to map each character in `t` to its index.\n3. Iterate through the characters in `s` and calculate the absolute difference between the indices of each character in `s` and `t`.\n4. Sum these absolute differences to get the permutation difference.\n\nHere is the Python code to implement this:\n\n```python\ndef permutation_difference(s, t):\n    # Create a dictionary to map each character in s to its index\n    s_indices = {char: idx for idx, char in enumerate(s)}\n    \n    # Create a dictionary to map each character in t to its index\n    t_indices = {char: idx for idx, char in enumerate(t)}\n    \n    # Calculate the permutation difference\n    permutation_diff = sum(abs(s_indices[char] - t_indices[char]) for char in s)\n    \n    return permutation_diff\n\n# Example usage\ns = input()\nt = input()\nprint(permutation_difference(s, t))\nassert permutation_difference('abc', 'bac') == 2```\n\nThis code first creates two dictionaries to map each character in `s` and `t` to their respective indices. Then, it calculates the permutation difference by summing the absolute differences between the indices of each character in `s` and `t`. The final result is returned as the permutation difference.\n</answer>",
            "ground_truth": '{\n  "inputs": [" \n "], "outputs": [""]}',
        },
    ]

    # preprocess data to remove input lines
    lines = data[0]["prediction"].split("\n")
    lines = [line for line in lines if "input()" not in line]
    print(lines)
    data[0]["prediction"] = "\n".join(lines)

    prediction = [d["prediction"] for d in data]
    ground_truth = [d["ground_truth"] for d in data]

    results_batch = grader.grade_batch(
        prediction=prediction,
        ground_truth=ground_truth,
        data_source=["code"],
        num_workers=num_workers,
        timeouts=300.0,
    )

    print(results_batch)
