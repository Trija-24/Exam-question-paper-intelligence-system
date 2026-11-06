"""Interactive input helpers that keep asking until the answer is valid."""
import os


def ask_pdf_path(prompt):
    while True:
        raw=input(prompt).strip().strip('"').strip("'")
        path=os.path.expanduser(raw)
        if raw and os.path.isfile(path) and path.lower().endswith(".pdf"):
            return path
        print("Invalid file path. Please enter a valid PDF path.")


def ask_positive_int(prompt):
    while True:
        raw=input(prompt).strip()
        if raw.isdigit() and int(raw)>0:
            return int(raw)
        print("Please enter a whole number greater than 0.")


def ask_year(prompt):
    while True:
        raw=input(prompt).strip()
        if raw.isdigit() and 1900<=int(raw)<=2100:
            return int(raw)
        print("Please enter a valid 4-digit year (for example 2023).")
