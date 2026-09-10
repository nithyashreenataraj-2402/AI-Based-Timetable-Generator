

"""Graph-coloring and backtracking based examination scheduler.

This file contains the main timetable generation algorithm.

Main idea:
1. Validate input data.
2. Build a conflict graph from courses.
3. Sort difficult courses first.
4. Use recursive backtracking to assign every course a slot and room.
5. Keep the best valid timetable found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from algorithm.constraints import (
    can_assign,
    evaluate_schedule,
    validate_input_data,
)
from algorithm.graph_builder import build_conflict_graph


@dataclass
class ScheduleResult:
    """Stores the final output returned by generate_timetable()."""

    # True if a valid timetable was generated, otherwise False.
    success: bool

    # Message shown in the UI, such as success or failure reason.
    message: str

    # Final generated timetable as a DataFrame.
    schedule: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Metrics such as conflicts, utilization, days used, etc.
    metrics: dict[str, float | int] = field(default_factory=dict)

    # Conflict graph used during scheduling.
    graph: Any | None = None

    # Number of recursive backtracking states visited.
    nodes_visited: int = 0

    # True if max_nodes or max_solutions limit was reached.
    search_limit_hit: bool = False


def _prepare_slot_options(slots_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert slots DataFrame into a list of easy-to-use slot dictionaries."""

    slot_options: list[dict[str, Any]] = []

    # reset_index(drop=True) gives clean index values starting from 0.
    for index, row in slots_df.reset_index(drop=True).iterrows():
        date = str(row["date"])
        time_slot = str(row["time_slot"])

        # slot_key uniquely identifies a date-time slot.
        # Example: "2026-07-01 | 09:30-12:30".
        slot_options.append(
            {
                "slot_index": int(index),
                "date": date,
                "time_slot": time_slot,
                "slot_key": f"{date} | {time_slot}",
            }
        )
    return slot_options


def _build_schedule_dataframe(
    assignments: dict[str, dict[str, Any]],
    courses_by_id: dict[str, dict[str, Any]],
    rooms_by_id: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Convert the internal assignments dictionary into a timetable DataFrame."""

    rows: list[dict[str, Any]] = []

    # Each assignment contains the selected slot and room for one course.
    for course_id, assignment in assignments.items():
        course = courses_by_id[course_id]
        room = rooms_by_id[assignment["room_id"]]
        room_capacity = int(room["capacity"])
        strength = int(course["strength"])

        # Add one row to the final timetable.
        rows.append(
            {
                "course_id": course_id,
                "course_name": course["course_name"],
                "date": assignment["date"],
                "time_slot": assignment["time_slot"],
                "room_id": assignment["room_id"],
                "student_groups": course["student_groups"],
                "strength": strength,
                "room_capacity": room_capacity,
                "room_utilization_percent": round(strength / room_capacity * 100, 2),
                "slot_key": assignment["slot_key"],
                "slot_index": assignment["slot_index"],
            }
        )

    schedule = pd.DataFrame(rows)

    # If there are no assignments, return an empty DataFrame.
    if schedule.empty:
        return schedule

    # Sort timetable by slot, room, and course for neat display.
    return schedule.sort_values(["slot_index", "room_id", "course_id"]).reset_index(drop=True)


def _optimization_key(metrics: dict[str, float | int]) -> tuple[float, ...]:
    """Create a comparison key to decide which valid timetable is better."""

    # Lower tuple value is better.
    # 1. Fewer conflicts is best.
    # 2. Fewer days used is better.
    # 3. Fewer slots used is better.
    # 4. Less room waste is better.
    # 5. Higher utilization is better, so it is made negative.
    return (
        float(metrics["conflicts_found"]),
        float(metrics["days_used"]),
        float(metrics["slots_used"]),
        float(metrics["room_waste"]),
        -float(metrics["room_utilization_percent"]),
    )


def generate_timetable(
    courses_df: pd.DataFrame,
    rooms_df: pd.DataFrame,
    slots_df: pd.DataFrame,
    *,
    max_solutions: int = 100,
    max_nodes: int = 100_000,
) -> ScheduleResult:
    """
    Generate the best valid timetable found by recursive backtracking.

    Date-time pairs act as graph colors. Room selection is checked as an
    additional CSP constraint at each recursive step.
    """

    # Step 1: Validate courses, rooms, and slots.
    # If input is invalid, stop immediately and return the error.
    validation = validate_input_data(courses_df, rooms_df, slots_df)
    if not validation.is_valid:
        return ScheduleResult(
            success=False,
            message="Invalid input: " + " ".join(validation.errors),
        )

    # Get cleaned DataFrames from validation result.
    courses = validation.courses
    rooms = validation.rooms
    slots = validation.slots

    # These assert statements tell Python that after successful validation,
    # courses, rooms, and slots will not be None.
    assert courses is not None and rooms is not None and slots is not None

    # Step 2: Build conflict graph.
    # Each course is a node, and an edge means two courses share a student group.
    graph = build_conflict_graph(courses)

    # Create lookup dictionaries for quick access using course_id and room_id.
    # Example: courses_by_id["CS201"] gives details of CS201.
    courses_by_id = courses.set_index("course_id").to_dict("index")
    rooms_by_id = rooms.set_index("room_id").to_dict("index")

    # Convert slots DataFrame into a list of slot dictionaries.
    slot_options = _prepare_slot_options(slots)

    # Step 3: Early impossible-case check.
    # If a course has more students than the largest room capacity,
    # no timetable can ever be valid.
    largest_room_capacity = int(rooms["capacity"].max())
    impossible_courses = courses.loc[
        courses["strength"] > largest_room_capacity,
        "course_id",
    ].tolist()

    if impossible_courses:
        return ScheduleResult(
            success=False,
            message=(
                "No valid timetable found: these courses exceed every room capacity: "
                + ", ".join(impossible_courses)
            ),
            graph=graph,
        )

    # Step 4: Sort courses by difficulty.
    # Courses with more conflicts are scheduled first.
    # If degree is same, courses with larger strength are scheduled first.
    course_order = sorted(
        graph.nodes,
        key=lambda course_id: (
            -graph.degree(course_id),
            -int(courses_by_id[course_id]["strength"]),
            course_id,
        ),
    )

    # Sort rooms by capacity in ascending order.
    # This tries the smallest suitable room first, improving utilization.
    room_order = sorted(
        rooms_by_id,
        key=lambda room_id: (int(rooms_by_id[room_id]["capacity"]), room_id),
    )

    # assignments stores the current partial timetable during backtracking.
    assignments: dict[str, dict[str, Any]] = {}

    # best_schedule stores the best valid complete timetable found so far.
    best_schedule = pd.DataFrame()
    best_metrics: dict[str, float | int] = {}
    best_key: tuple[float, ...] | None = None

    # solutions_found counts how many complete schedules have been generated.
    solutions_found = 0

    # nodes_visited counts recursive states explored by backtracking.
    nodes_visited = 0

    # This becomes True if search stops due to max_solutions or max_nodes.
    search_limit_hit = False

    def ordered_candidates(course_id: str) -> list[tuple[dict[str, Any], str]]:
        """Return all possible slot-room candidates for one course."""

        strength = int(courses_by_id[course_id]["strength"])
        candidates: list[tuple[dict[str, Any], str]] = []

        # Try every slot and every room.
        for slot in slot_options:
            for room_id in room_order:
                room_capacity = int(rooms_by_id[room_id]["capacity"])

                # Only keep rooms that can fit the course strength.
                if room_capacity >= strength:
                    candidates.append((slot, room_id))

        # Sort candidates in a useful order:
        # 1. Earlier slots first.
        # 2. Smaller room waste first.
        # 3. Room ID order for consistency.
        candidates.sort(
            key=lambda item: (
                item[0]["slot_index"],
                int(rooms_by_id[item[1]]["capacity"]) - strength,
                item[1],
            )
        )
        return candidates

    def current_days_used() -> int:
        """Count how many unique exam dates are currently used."""

        return len({assignment["date"] for assignment in assignments.values()})

    def backtrack(index: int) -> None:
        """Recursive backtracking function."""

        nonlocal best_schedule, best_metrics, best_key, solutions_found
        nonlocal nodes_visited, search_limit_hit

        # Stop if maximum allowed recursive states are reached.
        if nodes_visited >= max_nodes:
            search_limit_hit = True
            return

        # Count this recursive call as one visited search state.
        nodes_visited += 1

        # Pruning condition.
        if best_key is not None and current_days_used() > best_key[1]:
            return

        # Base case: all courses have been assigned.
        if index == len(course_order):
            schedule = _build_schedule_dataframe(assignments, courses_by_id, rooms_by_id)
            metrics = evaluate_schedule(schedule, graph)
            key = _optimization_key(metrics)

            if int(metrics["conflicts_found"]) == 0 and (
                best_key is None or key < best_key
            ):
                best_schedule = schedule
                best_metrics = metrics
                best_key = key

            solutions_found += 1
            return

        if solutions_found >= max_solutions:
            search_limit_hit = True
            return

        course_id = course_order[index]

        for slot, room_id in ordered_candidates(course_id):
            if can_assign(
                course_id,
                slot["slot_key"],
                room_id,
                assignments,
                graph,
                courses_by_id,
                rooms_by_id,
            ):
                assignments[course_id] = {
                    "slot_index": slot["slot_index"],
                    "date": slot["date"],
                    "time_slot": slot["time_slot"],
                    "slot_key": slot["slot_key"],
                    "room_id": room_id,
                }

                # Recursive step: move to the next course.
                backtrack(index + 1)

                # Backtracking undo step.
                del assignments[course_id]

                if nodes_visited >= max_nodes or solutions_found >= max_solutions:
                    search_limit_hit = True
                    return

    # Start recursive search from index 0, meaning the first course.
    backtrack(0)

    if best_schedule.empty:
        return ScheduleResult(
            success=False,
            message=(
                "No valid timetable found. Add more date-time slots, add rooms, "
                "or increase room capacity."
            ),
            graph=graph,
            nodes_visited=nodes_visited,
            search_limit_hit=search_limit_hit,
        )

    message = "Valid timetable generated with zero hard conflicts."
    if search_limit_hit:
        message += " Search limit was reached, so this is the best valid timetable found so far."

    return ScheduleResult(
        success=True,
        message=message,
        schedule=best_schedule.drop(columns=["slot_index"]),
        metrics=best_metrics,
        graph=graph,
        nodes_visited=nodes_visited,
        search_limit_hit=search_limit_hit,
    )

