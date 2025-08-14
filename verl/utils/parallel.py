import asyncio
import json
import logging
import multiprocessing
import threading
from concurrent.futures import TimeoutError
from functools import wraps
from queue import Queue as ThreadQueue
from typing import Any, Callable, Optional

import pebble
from tqdm import tqdm
import resource



def timeout_decorator(timeout: Optional[float] = None, default_return_value: Any = None):
    """
    Decorator that runs a function in a separate thread with a timeout.
    If an exception occurs, it will log a warning and return the default_return_value if provided, else raise the exception.
    Usage:
        @timeout_decorator(timeout=5, default_return_value=0)
        def my_function(arg1, arg2):
            ...
    """

    def _wrapper(func: Callable):
        def _worker(queue: ThreadQueue, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                queue.put(result)
            except Exception as e:
                queue.put(e)  # if the function raises an exception, put it in the queue

        @wraps(func)
        def _wrapped(*args, **kwargs) -> Any:
            if timeout is None:
                return func(*args, **kwargs)

            func_name = func.__qualname__ if hasattr(func, "__qualname__") else func.__name__

            result_queue = ThreadQueue()
            thread = threading.Thread(target=_worker, args=(result_queue, *args), kwargs=kwargs)
            thread.daemon = True

            thread.start()
            thread.join(timeout=timeout)

            e = None  # default exception to raise

            if thread.is_alive():
                # Note: We can't forcefully terminate threads in Python
                # The thread will continue running but we'll move on
                e = TimeoutError(f"function {func_name} timed out after {timeout}s")

            if e is None and result_queue.empty():
                e = Exception(f"function {func_name} failed without returning a result")

            result = result_queue.get()
            if isinstance(result, Exception):  # if the result is an exception, raise it
                e = result

            # if exception, log a warning and return default value if provided, else raise the exception
            if e is not None:
                if default_return_value is not None:
                    logging.warning(
                        f"returning default return value ({default_return_value}) for function {func_name} because it timed out with exception {e} after {timeout}s"
                    )
                    return default_return_value
                raise e

            # if no exception, return the result
            return result

        return _wrapped

    return _wrapper


async def _process_batch_helper(func, batch):
    """Helper function to process a batch of items concurrently"""
    tasks = []
    for args in batch:
        if isinstance(args, tuple):
            task = asyncio.create_task(func(*args))
        else:
            task = asyncio.create_task(func(args))
        tasks.append(task)
    return await asyncio.gather(*tasks)


async def map_parallel_asyncio_helper(func, args_list, max_batch_size=16384):
    if isinstance(args_list, zip):  # if zip, convert to list of tuples
        args_list = list(args_list)

    results = []
    for i in range(0, len(args_list), max_batch_size):  # limit batch size
        batch = args_list[i : i + max_batch_size]
        batch_results = await _process_batch_helper(func, batch)
        results.extend(batch_results)
    return results


def map_parallel_asyncio(func, args_list, max_batch_size=16384):
    """
    Map a function over a list of args using asyncio with batching
    example:
        result[0] = func(args_list[0])
        result[1] = func(args_list[1])        ...
    """

    async def wrapped_func(*args):
        return await asyncio.to_thread(func, *args)

    return asyncio.run(map_parallel_asyncio_helper(wrapped_func, args_list, max_batch_size))


def map_parallel_mp(func, args_list, num_workers=16, max_batch_size=16384, mp_mode="starmap"):
    """
    Map a function over a list of args using multiprocessing with batching
    args_list for starmap: list of args to pass to func [(arg_1, arg_2, ...), (arg_1, arg_2, ...), ...]
    args_list for map: list of args to pass to func [arg_1, arg_2, ...]
    Usage:
        map_parallel_mp(func,
                        args_list=zip(arg1_list, arg2_list, arg3_list),
                        num_workers=8,
                        max_batch_size=4096,
                        mp_mode="starmap")
    example:
        result[0] = func(args_list[0])
        result[1] = func(args_list[1])
        ...
    """
    func_name = func.__qualname__ if hasattr(func, "__qualname__") else func.__name__
    if isinstance(args_list, zip):  # if zip, convert to list of tuples
        args_list = list(args_list)
    results = []
    with multiprocessing.Pool(num_workers) as pool:
        with tqdm(total=len(args_list), desc=f"map_parallel_mp {func_name}") as pbar:
            for i in range(0, len(args_list), max_batch_size):  # run mp map in batches
                batch = args_list[i : i + max_batch_size]
                if mp_mode == "starmap":
                    results.extend(pool.starmap(func, batch))
                elif mp_mode == "map":
                    results.extend(pool.map(func, batch))
                else:
                    raise ValueError(f"map_parallel_mp {func_name}: invalid mp_mode {mp_mode}")
                pbar.update(len(batch))
    return results

# 64MB
MAX_MEM = 67108864


def map_parallel_timeout_pebble(
    func,
    args_list,
    timeouts,
    num_workers=16,
    default_return_value=None,
):
    """
    Map a function over a list of arguments using pebble ProcessPool with batching and optional per-task timeouts.

    Parameters:
    - func: The function to execute in parallel.
    - args_list: Arguments to pass to the function.
    - num_workers: Number of worker processes (default: 16).
    - timeouts: Timeout in seconds per task (default: None, meaning no timeout).
    - default_return_value: Value to return if a task times out or raises an exception (default: None).

    Returns:
    - List of results in the order of args_list.
    """
    # Get function name for logging and progress bar
    func_name = func.__qualname__ if hasattr(func, "__qualname__") else func.__name__

    # Convert zip object to list if necessary (common in starmap scenarios)
    if isinstance(args_list, zip):
        args_list = list(args_list)

    results = []
    with pebble.ProcessPool(max_workers=num_workers) as pool:
        with tqdm(total=len(args_list), desc=f"map_parallel_timeout_pebble {func_name}") as pbar:
            tasks = [
                pool.schedule(func, tuple(args), timeout=timeout)
                for args, timeout in zip(args_list, timeouts)
            ]

            for i, task in enumerate(tasks):
                try:
                    result = task.result()  # Blocks until task completes or times out
                    results.append(result)
                except TimeoutError:
                    task.cancel()
                    logging.warning(f"Function {func_name} timed out after {timeouts[i]}s")
                    results.append(default_return_value)
                except MemoryError:
                    task.cancel()
                    logging.error(f"Memory limit exceeded for example {args_list[i][1]}")
                    results.append(default_return_value)
                except Exception as e:
                    task.cancel()
                    logging.error(f"Function {func_name} raised an exception: {e}")
                    results.append(default_return_value)

                pbar.update(1)

    return results


def map_parallel_timeout_pebble_code_execution(
    func,
    args_list,
    timeouts,
    num_workers=16,
    default_return_value=None,
):
    """
    Map a function over a list of arguments using pebble ProcessPool with batching and optional per-task timeouts.

    Parameters:
    - func: The function to execute in parallel.
    - args_list: Arguments to pass to the function.
        - For mp_mode="starmap": List of tuples [(arg1, arg2, ...), (arg1, arg2, ...), ...]
        - For mp_mode="map": List of single arguments [arg1, arg2, ...]
    - num_workers: Number of worker processes (default: 16).
    - timeout: Timeout in seconds per task (default: None, meaning no timeout).
    - default_return_value: Value to return if a task times out or raises an exception (default: None).

    Returns:
    - List of results in the order of args_list.
    """
    # Get function name for logging and progress bar
    func_name = func.__qualname__ if hasattr(func, "__qualname__") else func.__name__

    # Convert zip object to list if necessary (common in starmap scenarios)
    if isinstance(args_list, zip):
        args_list = list(args_list)

    def initializer(limit):
        """Set maximum amount of memory each worker process can allocate."""
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))

    results = []
    with pebble.ProcessPool(max_workers=num_workers) as pool:
        with tqdm(
            total=len(args_list),
            desc=f"map_parallel_timeout_pebble {func_name} with continuous=False",
        ) as pbar:
            tasks = [
                pool.schedule(func, args, timeout=timeout)
                for args, timeout in zip(args_list, timeouts)
            ]

            for i, task in enumerate(tasks):
                try:
                    result = task.result()  # Blocks until task completes or times out
                    results.append(result)
                    task.cancel()
                except TimeoutError:
                    task.cancel()
                    logging.warning(f"Function {func_name} timed out after {timeouts[i]}s")
                    results.append(default_return_value)
                except MemoryError:
                    task.cancel()
                    logging.error(f"Memory limit exceeded for example {args_list[i][1]}")
                    results.append(default_return_value)
                except Exception as e:
                    task.cancel()
                    logging.error(f"Function {func_name} raised an exception: {e}")
                    results.append(default_return_value)
                
                task.cancel()
                pbar.update(1)

    with pebble.ProcessPool(max_workers=num_workers) as pool:
        with tqdm(
            total=len(args_list),
            desc=f"map_parallel_timeout_pebble_code_execution {func_name} with continuous=True",
        ) as pbar:
            tasks = []
            for i, result in enumerate(results):
                if not result:
                    # run each ground truth test case individually
                    sub_results = []
                    # limit to the first 10 test cases for now otherwise it's too slow since some examples have 1000+ test cases
                    try:
                        test_cases = args_list[i][1]
                        if not isinstance(test_cases, dict):
                            test_cases = json.loads(test_cases.strip())
                        inputs = test_cases["inputs"][:min(len(test_cases["inputs"]), 10)]
                        outputs = test_cases["outputs"][:min(len(test_cases["outputs"]), 10)]
                        for input, output in zip(inputs, outputs):
                            args = (args_list[i][0], {"inputs": [input], "outputs": [output]}, args_list[i][2], args_list[i][3])
                            sub_results.append(pool.schedule(func, args, timeout=timeouts[i]))
                        tasks.append(sub_results)
                    except MemoryError:
                        task.cancel()
                        logging.error(f"Memory limit exceeded for example {args_list[i][1]}")
                        tasks.append(None)
                    except Exception as e:
                        task.cancel()   
                        print(f"Error processing test cases for example {args_list[i][1]}: {e}")
                        tasks.append(None)
                else:
                    tasks.append(None)

            for i, task in enumerate(tasks):
                if task is not None:
                    try:
                        total_correct = 0
                        total_test_cases = 0
                        for sub_result in task:
                            result = sub_result.result()
                            total_correct += result
                            total_test_cases += 1
                            sub_result.cancel()
                        results[i] = total_correct / total_test_cases
                    except TimeoutError:
                        for sub_result in task:
                            sub_result.cancel()
                        logging.warning(f"Function {func_name} timed out after {timeouts[i]}s")
                        results[i] = default_return_value
                    except MemoryError:
                        for sub_result in task:
                            sub_result.cancel()
                        logging.error(f"Memory limit exceeded for example {args_list[i][1]}")
                        results[i] = default_return_value
                    except Exception as e:
                        for sub_result in task:
                            sub_result.cancel()
                        logging.error(f"Function {func_name} raised an exception: {e}")
                        results[i] = default_return_value

                pbar.update(1)

    return results
