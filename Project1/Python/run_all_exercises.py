#!/usr/bin/env python3
"""
Run all Project 1 exercises in a single command.

Usage (from repo root):
  & ".\\compmctrl\\Scripts\\python.exe" Project1\\Python\\run_all_exercises.py
"""

from exercise_all import exercise_all


def main():
    # exercise_all() expects a list of flags indicating which exercises to run
    exercise_all(['1_1', '1_2', '2_1', '2_2', '2_3', '3_1', '3_2', '3_3'])


if __name__ == "__main__":
    main()

