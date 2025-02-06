import re
from typing import List
from pydantic import BaseModel
from schema import *

def parse_solutions_string(solutions_str: str) -> Solutions:
    """
    Converts a string of format:
      solutions=[Solution(problem_id='000', solution=[2, 0, 3])]
    into a Solutions object:
      Solutions(solutions=[Solution(problem_id='000', solution=[2, 0, 3])])
    """
    # Remove "solutions=" if present
    if solutions_str.startswith("solutions="):
        solutions_str = solutions_str[len("solutions="):].strip()

    pattern = re.compile(
        r"Solution\s*\(\s*problem_id\s*=\s*'([^']+)'\s*,\s*solution\s*=\s*\[\s*(.*?)\s*\]\s*\)"
    )

    found_solutions = []
    for match in pattern.finditer(solutions_str):
        problem_id = match.group(1)
        sol_list_str = match.group(2)

        if sol_list_str.strip():
            str_numbers = [x.strip() for x in sol_list_str.split(",")]
            int_numbers = list(map(int, str_numbers))
        else:
            int_numbers = []

        # Changed from id to problem_id to match schema
        found_solutions.append(Solution(problem_id=problem_id, solution=int_numbers))

    if not found_solutions:
        print(solutions_str)
        raise ValueError("No solutions found in input string")

    return Solutions(solutions=found_solutions)
    """
    Converts a string of format:
      solutions=[Solution(problem_id='000', solution=[2, 0, 3])]
    into a Solutions object:
      Solutions(solutions=[Solution(id='000', solution=[2, 0, 3])])
    """
    # Remove "solutions=" if present
    if solutions_str.startswith("solutions="):
        solutions_str = solutions_str[len("solutions="):].strip()

    # Updated regex to match problem_id instead of id
    pattern = re.compile(
        r"Solution\s*\(\s*problem_id\s*=\s*'([^']+)'\s*,\s*solution\s*=\s*\[\s*(.*?)\s*\]\s*\)"
    )

    found_solutions = []
    for match in pattern.finditer(solutions_str):
        sol_id = match.group(1)
        sol_list_str = match.group(2)

        # Split the solution array by commas, ignoring spaces
        if sol_list_str.strip():
            str_numbers = [x.strip() for x in sol_list_str.split(",")]
            int_numbers = list(map(int, str_numbers))
        else:
            int_numbers = []

        # Note: Solution expects 'id', not 'problem_id'
        found_solutions.append(Solution(id=sol_id, solution=int_numbers))

    if not found_solutions:
        raise ValueError("No solutions found in input string")

    return Solutions(solutions=found_solutions)
    """
    Converts a string of format:
      solutions=[Solution(id='000', solution=[2, 0, 3]), Solution(id='001', solution=[0, 1, 3])]
    into a Solutions object:
      Solutions(solutions=[Solution(id='000', solution=[2, 0, 3]), ...])
    """
    # 1) Remove "solutions=" if present
    if solutions_str.startswith("solutions="):
        solutions_str = solutions_str[len("solutions="):].strip()

    # This regex captures something like:
    # Solution(id='000', solution=[2, 0, 3])
    # in two groups:
    #   group(1) = "000"
    #   group(2) = "2, 0, 3"
    pattern = re.compile(
        r"Solution\s*\(\s*id\s*=\s*'([^']+)'\s*,\s*solution\s*=\s*\[\s*(.*?)\s*\]\s*\)"
    )

    found_solutions = []
    for match in pattern.finditer(solutions_str):
        sol_id = match.group(1)
        sol_list_str = match.group(2)

        # Split the solution array by commas, ignoring spaces
        if sol_list_str.strip():
            # e.g. "2, 0, 3" => ["2", "0", "3"]
            str_numbers = [x.strip() for x in sol_list_str.split(",")]
            int_numbers = list(map(int, str_numbers))
        else:
            int_numbers = []

        found_solutions.append(Solution(id=sol_id, solution=int_numbers))

    return Solutions(solutions=found_solutions)