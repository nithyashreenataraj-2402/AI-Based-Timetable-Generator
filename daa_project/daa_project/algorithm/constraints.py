"""Validation, constraint checks, and schedule metrics.

This file is responsible for three main tasks:
1. Cleaning and validating input CSV data.
2. Checking whether a course can be assigned to a slot and room.
3. Calculating final timetable metrics such as conflicts and utilization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


# These are the required columns expected in each uploaded CSV file.
# If any of these columns are missing, the program will show an input error.
COURSE_COLUMNS = ["course_id", "course_name", "student_groups", "strength"]
ROOM_COLUMNS = ["room_id", "capacity"]
SLOT_COLUMNS = ["date", "time_slot"]


@dataclass
class ValidationResult:
    """Stores the result of validating courses, rooms, and slots data."""

    is_valid: bool
    errors: list[str]
    courses: pd.DataFrame | None = None
    rooms: pd.DataFrame | None = None
    slots: pd.DataFrame | None = None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with predictable lower_snake_case column names."""
    cleaned = df.copy()

    # Convert column names like "Course ID" or "Course-ID" into "course_id".
    # This makes the program tolerant of small differences in uploaded CSV files.
    cleaned.columns = [
        str(column).strip().lower().replace(" ", "_").replace("-", "_")
        for column in cleaned.columns
    ]
    return cleaned


def split_student_groups(value: Any) -> list[str]:
    """Split a group field such as 'AIML-A; CSE-A' into canonical group names."""

    # If the value is empty or missing, return an empty list.
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    # If the groups are already provided as a list/tuple/set, use them directly.
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        # Accept different separators such as ; , and |.
        raw_text = str(value).replace("|", ";").replace(",", ";")
        raw_items = raw_text.split(";")

    groups: list[str] = []
    seen: set[str] = set()

    for item in raw_items:
        # Clean spaces and convert group names to uppercase for consistency.
        group = str(item).strip().upper()

        # Avoid duplicate groups like "AIML-A; AIML-A".
        if group and group not in seen:
            groups.append(group)
            seen.add(group)

    return groups


def normalize_courses_dataframe(courses_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the courses table."""

    df = normalize_columns(courses_df)

    # Check whether all required course columns are present.
    missing = set(COURSE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Courses CSV is missing columns: {', '.join(sorted(missing))}")

    df = df.copy()

    # Clean course ID and make it uppercase.
    df["course_id"] = df["course_id"].astype(str).str.strip().str.upper()

    # Remove extra spaces from course names.
    df["course_name"] = df["course_name"].astype(str).str.strip()

    # Normalize student groups into a standard format like "AIML-A; CSE-A".
    df["student_groups"] = df["student_groups"].apply(
        lambda value: "; ".join(split_student_groups(value))
    )

    # Convert strength to a number. Invalid values become NaN and are handled below.
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")

    errors: list[str] = []

    # Validate basic course rules.
    if df.empty:
        errors.append("Add at least one course.")

    if df["course_id"].eq("").any():
        errors.append("Every course must have a course_id.")

    if df["course_id"].duplicated().any():
        duplicates = sorted(df.loc[df["course_id"].duplicated(), "course_id"].unique())
        errors.append(f"Duplicate course_id values found: {', '.join(duplicates)}")

    if df["student_groups"].eq("").any():
        errors.append("Every course must have at least one student group.")

    if df["strength"].isna().any() or (df["strength"] <= 0).any():
        errors.append("Course strength must be a positive number for every course.")

    # If any validation error was found, stop and show all errors together.
    if errors:
        raise ValueError(" ".join(errors))

    # If course_name is empty, use course_id as the course name.
    df["course_name"] = df.apply(
        lambda row: row["course_name"] if row["course_name"] else row["course_id"],
        axis=1,
    )

    # Store strength as an integer and return only the required columns.
    df["strength"] = df["strength"].astype(int)
    return df[COURSE_COLUMNS].sort_values("course_id").reset_index(drop=True)


def normalize_rooms_dataframe(rooms_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the rooms table."""

    df = normalize_columns(rooms_df)

    # Check whether all required room columns are present.
    missing = set(ROOM_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Rooms CSV is missing columns: {', '.join(sorted(missing))}")

    df = df.copy()

    # Clean room ID and make it uppercase.
    df["room_id"] = df["room_id"].astype(str).str.strip().str.upper()

    # Convert room capacity to a number.
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce")

    errors: list[str] = []

    if df.empty:
        errors.append("Add at least one room.")

    if df["room_id"].eq("").any():
        errors.append("Every room must have a room_id.")

    if df["room_id"].duplicated().any():
        duplicates = sorted(df.loc[df["room_id"].duplicated(), "room_id"].unique())
        errors.append(f"Duplicate room_id values found: {', '.join(duplicates)}")

    if df["capacity"].isna().any() or (df["capacity"] <= 0).any():
        errors.append("Room capacity must be a positive number for every room.")

    if errors:
        raise ValueError(" ".join(errors))

    df["capacity"] = df["capacity"].astype(int)
    return df[ROOM_COLUMNS].sort_values("room_id").reset_index(drop=True)


def normalize_slots_dataframe(slots_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the exam slots table."""

    df = normalize_columns(slots_df)

    missing = set(SLOT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Slots CSV is missing columns: {', '.join(sorted(missing))}")

    df = df.copy()

    df["date"] = df["date"].astype(str).str.strip()
    df["time_slot"] = df["time_slot"].astype(str).str.strip()

    errors: list[str] = []

    if df.empty:
        errors.append("Add at least one exam date and time slot.")

    if df["date"].eq("").any():
        errors.append("Every slot must have a date.")

    if df["time_slot"].eq("").any():
        errors.append("Every slot must have a time_slot.")

    if errors:
        raise ValueError(" ".join(errors))

    # Remove duplicate date-time slot combinations.
    df = df.drop_duplicates(subset=["date", "time_slot"])
    return df[SLOT_COLUMNS].reset_index(drop=True)


def validate_input_data(
    courses_df: pd.DataFrame,
    rooms_df: pd.DataFrame,
    slots_df: pd.DataFrame,
) -> ValidationResult:
    """Validate courses, rooms, and slots together."""

    errors: list[str] = []
    courses = rooms = slots = None

    try:
        courses = normalize_courses_dataframe(courses_df)
    except ValueError as exc:
        errors.append(str(exc))

    try:
        rooms = normalize_rooms_dataframe(rooms_df)
    except ValueError as exc:
        errors.append(str(exc))

    try:
        slots = normalize_slots_dataframe(slots_df)
    except ValueError as exc:
        errors.append(str(exc))

    return ValidationResult(
        is_valid=not errors,
        errors=errors,
        courses=courses,
        rooms=rooms,
        slots=slots,
    )


def can_assign(
    course_id: str,
    slot_key: str,
    room_id: str,
    assignments: dict[str, dict[str, Any]],
    graph: Any,
    courses_by_id: dict[str, dict[str, Any]],
    rooms_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Check all hard CSP constraints for one candidate assignment."""

    course = courses_by_id[course_id]
    room = rooms_by_id[room_id]

    # Constraint 1: Room capacity must be enough for the course strength.
    if int(room["capacity"]) < int(course["strength"]):
        return False

    # Check the new assignment against all already assigned courses.
    for assigned_course_id, assigned in assignments.items():
        # Only courses in the same slot can clash.
        if assigned["slot_key"] != slot_key:
            continue

        # Constraint 2: Student clash check.
        if graph.has_edge(course_id, assigned_course_id):
            return False

        # Constraint 3: Room clash check.
        if assigned["room_id"] == room_id:
            return False

    return True


def evaluate_schedule(
    schedule_df: pd.DataFrame,
    graph: Any | None = None,
) -> dict[str, float | int]:
    """Calculate conflicts, utilization, and a demo-friendly fitness score."""

    if schedule_df is None or schedule_df.empty:
        return {
            "total_courses": 0,
            "total_rooms": 0,
            "slots_used": 0,
            "days_used": 0,
            "student_clashes": 0,
            "room_conflicts": 0,
            "capacity_issues": 0,
            "conflicts_found": 0,
            "room_utilization_percent": 0.0,
            "room_waste": 0,
            "fitness_score": 0.0,
        }

    df = schedule_df.copy()

    if "slot_key" not in df.columns:
        df["slot_key"] = df["date"].astype(str) + " | " + df["time_slot"].astype(str)

    # Count student clashes using the conflict graph.
    student_clashes = 0
    if graph is not None:
        rows_by_course = df.set_index("course_id").to_dict("index")
        for course_a, course_b in graph.edges():
            if course_a in rows_by_course and course_b in rows_by_course:
                if rows_by_course[course_a]["slot_key"] == rows_by_course[course_b]["slot_key"]:
                    student_clashes += 1

    # Count room conflicts.
    room_conflicts = 0
    for _, group in df.groupby(["slot_key", "room_id"]):
        if len(group) > 1:
            room_conflicts += len(group) - 1

    # Count capacity issues.
    capacity_issues = int(
        (df["strength"].astype(int) > df["room_capacity"].astype(int)).sum()
    )

    conflicts_found = student_clashes + room_conflicts + capacity_issues

    # Room utilization = total students / total assigned room capacity * 100.
    total_capacity = int(df["room_capacity"].sum())
    total_strength = int(df["strength"].sum())
    utilization = (total_strength / total_capacity * 100) if total_capacity else 0.0

    # Room waste means unused seats.
    room_waste = int((df["room_capacity"] - df["strength"]).clip(lower=0).sum())

    days_used = int(df["date"].nunique())
    slots_used = int(df["slot_key"].nunique())

    # Fitness score starts from 100 and subtracts penalties.
    conflict_penalty = conflicts_found * 25
    day_penalty = max(0, days_used - 1) * 2
    waste_penalty = max(0.0, 100.0 - utilization) * 0.15

    fitness = max(0.0, 100.0 - conflict_penalty - day_penalty - waste_penalty)

    return {
        "total_courses": int(df["course_id"].nunique()),
        "total_rooms": int(df["room_id"].nunique()),
        "slots_used": slots_used,
        "days_used": days_used,
        "student_clashes": int(student_clashes),
        "room_conflicts": int(room_conflicts),
        "capacity_issues": capacity_issues,
        "conflicts_found": int(conflicts_found),
        "room_utilization_percent": round(utilization, 2),
        "room_waste": room_waste,
        "fitness_score": round(fitness, 2),
    }