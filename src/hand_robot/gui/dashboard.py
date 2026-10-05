from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def launch():
    """Launch a lightweight desktop dashboard."""
    root = tk.Tk()
    root.title("20-DOF Hand Robot — Piano Tasks")
    root.geometry("720x420")

    title = ttk.Label(
        root,
        text="20-DOF Branched Hand Robot",
        font=("TkDefaultFont", 18, "bold"),
    )
    title.pack(pady=20)

    info = ttk.Label(
        root,
        text=(
            "Tasks: greeting | single key | chord | octave jump | "
            "dynamic key | musical piece"
        ),
        wraplength=650,
    )
    info.pack(pady=10)

    status = tk.StringVar(value="Ready — simulation only")
    ttk.Label(root, textvariable=status).pack(pady=10)

    ttk.Button(
        root,
        text="Close",
        command=root.destroy,
    ).pack(pady=20)

    root.mainloop()
