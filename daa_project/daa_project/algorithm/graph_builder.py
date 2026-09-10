"""Conflict graph construction for examination courses.

This file builds the graph used in the timetable generator.

In this graph:
1. Each course is represented as a node.
2. An edge is added between two courses if they share at least one student group.
3. An edge means those two courses cannot be scheduled in the same date-time slot.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx
import pandas as pd

# These helper functions are imported from constraints.py.
# normalize_courses_dataframe() cleans and validates the courses table.
# split_student_groups() converts a string like "AIML-A; CSE-A" into a list.
from algorithm.constraints import normalize_courses_dataframe, split_student_groups


def build_conflict_graph(courses_df: pd.DataFrame) -> nx.Graph:
    """Build an undirected graph where edges mean courses share a student group."""

    # First clean and validate the course data.
    # This ensures course_id, course_name, student_groups, and strength are correct.
    courses = normalize_courses_dataframe(courses_df)

    # Create an empty undirected graph using NetworkX.
    # Undirected graph is used because conflict relation is mutual:
    # if Course A conflicts with Course B, then Course B also conflicts with Course A.
    graph = nx.Graph()

    # This dictionary maps each student group to the list of courses taken by that group.
    # Example:
    # {
    #     "AIML-A": ["CS201", "MA201", "AI201"],
    #     "CSE-A": ["CS201", "DB201"]
    # }
    group_to_courses: dict[str, list[str]] = defaultdict(list)

    # Add every course as a node in the graph.
    for _, row in courses.iterrows():
        # Convert student_groups string into a list of group names.
        # Example: "AIML-A; CSE-A" becomes ["AIML-A", "CSE-A"].
        groups = split_student_groups(row["student_groups"])

        # Add the course as a graph node.
        # Extra information is stored as node attributes.
        graph.add_node(
            row["course_id"],
            course_name=row["course_name"],
            student_groups=groups,
            strength=int(row["strength"]),
        )

        # For each student group of this course, record that this course belongs to it.
        # This will later help us find courses that share the same student group.
        for group in groups:
            group_to_courses[group].append(row["course_id"])

    # Now create edges between courses that share a student group.
    for group, course_ids in group_to_courses.items():
        # combinations(sorted(course_ids), 2) creates every pair of courses
        # that belong to the same student group.
        #
        # Example:
        # course_ids = ["CS201", "MA201", "AI201"]
        # pairs generated:
        # ("AI201", "CS201"), ("AI201", "MA201"), ("CS201", "MA201")
        for course_a, course_b in combinations(sorted(course_ids), 2):
            # If an edge already exists, it means these two courses already
            # conflict because of another student group.
            # So we only add the new group name to the existing conflict list.
            if graph.has_edge(course_a, course_b):
                graph[course_a][course_b]["conflict_groups"].append(group)
            else:
                # If no edge exists, create a new edge.
                # conflict_groups stores which student group caused the conflict.
                graph.add_edge(course_a, course_b, conflict_groups=[group])

    # Remove duplicate conflict group names and sort them.
    # This keeps the edge data clean and easy to display in the UI.
    for course_a, course_b in graph.edges():
        graph[course_a][course_b]["conflict_groups"] = sorted(
            set(graph[course_a][course_b]["conflict_groups"])
        )

    # Return the final conflict graph.
    return graph


def graph_summary(graph: nx.Graph) -> dict[str, int | float]:
    """Return small graph metrics that are useful for the UI and presentation."""

    # Total number of courses in the graph.
    node_count = graph.number_of_nodes()

    # Total number of conflicts between courses.
    edge_count = graph.number_of_edges()

    # Graph density tells how connected the graph is.
    # Density is between 0 and 1.
    # 0 means no conflicts, 1 means every course conflicts with every other course.
    density = nx.density(graph) if node_count > 1 else 0.0

    # Degree of a course means how many other courses it conflicts with.
    # max_degree gives the highest conflict count among all courses.
    max_degree = max(dict(graph.degree()).values(), default=0)

    # Return summary values used in the Streamlit UI.
    return {
        "nodes": node_count,
        "edges": edge_count,
        "density": round(density, 3),
        "max_degree": max_degree,
    }