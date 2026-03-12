import tkinter as tk

# from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from tkcalendar import Calendar  # pip install tkcalendar


class SimpleGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Data Processor")
        self.window.geometry("500x400")
        self.window.resizable(False, False)

        # Variables to store selections
        self.selected_date = None
        self.selected_file = None

        self._create_widgets()

    def _create_widgets(self):
        """Create all GUI elements."""
        # Main frame with padding
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # ===== DATE PICKER =====
        ttk.Label(main_frame, text="Select Date:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5)
        )

        # Date display and button
        date_frame = ttk.Frame(main_frame)
        date_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        self.date_var = tk.StringVar(value="No date selected")
        ttk.Label(
            date_frame, textvariable=self.date_var, relief=tk.SUNKEN, width=30
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(date_frame, text="Choose Date", command=self._open_calendar).pack(
            side=tk.LEFT
        )

        # ===== FILE PICKER =====
        ttk.Label(main_frame, text="Select File:", font=("Arial", 10, "bold")).grid(
            row=2, column=0, sticky=tk.W, pady=(0, 5)
        )

        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(
            file_frame, textvariable=self.file_var, relief=tk.SUNKEN, width=30
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(file_frame, text="Browse...", command=self._browse_file).pack(
            side=tk.LEFT
        )

        # ===== ACTION BUTTONS =====
        ttk.Label(main_frame, text="Actions:", font=("Arial", 10, "bold")).grid(
            row=4, column=0, sticky=tk.W, pady=(0, 10)
        )

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, sticky=(tk.W, tk.E))

        ttk.Button(
            button_frame, text="Process Data", command=self._process_data, width=15
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame, text="Analyze", command=self._analyze_data, width=15
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(
            button_frame, text="Export", command=self._export_data, width=15
        ).pack(side=tk.LEFT)

        # ===== STATUS BAR =====
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def _open_calendar(self):
        """Open calendar dialog for date selection."""
        cal_window = tk.Toplevel(self.window)
        cal_window.title("Select Date")
        cal_window.geometry("300x300")
        cal_window.resizable(False, False)

        # Create calendar widget
        cal = Calendar(cal_window, selectmode="day", date_pattern="yyyy-mm-dd")
        cal.pack(pady=20, padx=20, fill="both", expand=True)

        def select_date():
            self.selected_date = cal.get_date()
            self.date_var.set(self.selected_date)
            cal_window.destroy()

        ttk.Button(cal_window, text="Select", command=select_date).pack(pady=10)

    def _browse_file(self):
        """Open file dialog for file selection."""
        filename = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[
                ("All files", "*.*"),
                ("Text files", "*.txt"),
                ("CSV files", "*.csv"),
                ("NetCDF files", "*.nc"),
                ("Binary files", "*.bin"),
            ],
        )

        if filename:
            self.selected_file = Path(filename)
            # Show only filename if path is too long
            display_name = self.selected_file.name
            if len(str(self.selected_file)) > 40:
                display_name = "..." + str(self.selected_file)[-37:]
            else:
                display_name = str(self.selected_file)
            self.file_var.set(display_name)

    def _validate_selections(self):
        """Check if date and file are selected."""
        if not self.selected_date:
            messagebox.showwarning("Missing Selection", "Please select a date.")
            return False

        if not self.selected_file:
            messagebox.showwarning("Missing Selection", "Please select a file.")
            return False

        return True

    def _process_data(self):
        """Process button action."""
        if not self._validate_selections():
            return

        self.status_var.set("Processing data...")
        self.window.update()

        try:
            # Call your actual processing method here
            _result = self.process_method(self.selected_date, self.selected_file)

            messagebox.showinfo(
                "Success",
                f"Data processed successfully!\n\n"
                f"Date: {self.selected_date}\n"
                f"File: {self.selected_file.name}",
            )
            self.status_var.set("Processing complete")

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
            self.status_var.set("Processing failed")

    def _analyze_data(self):
        """Analyze button action."""
        if not self._validate_selections():
            return

        self.status_var.set("Analyzing data...")
        self.window.update()

        try:
            result = self.analyze_method(self.selected_date, self.selected_file)
            messagebox.showinfo(
                "Analysis Complete", f"Analysis finished!\n\nResults: {result}"
            )
            self.status_var.set("Analysis complete")
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed:\n{str(e)}")
            self.status_var.set("Analysis failed")

    def _export_data(self):
        """Export button action."""
        if not self._validate_selections():
            return

        # Ask where to save
        save_path = filedialog.asksaveasfilename(
            title="Save export as",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")],
        )

        if save_path:
            self.status_var.set("Exporting...")
            self.window.update()

            try:
                self.export_method(self.selected_date, self.selected_file, save_path)
                messagebox.showinfo("Success", f"Exported to:\n{save_path}")
                self.status_var.set("Export complete")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
                self.status_var.set("Export failed")

    # ===== YOUR ACTUAL METHODS =====
    def process_method(self, date, file_path):
        """Replace with your actual processing logic."""
        print(f"Processing: {file_path} for date {date}")
        # Simulate work
        import time

        time.sleep(1)
        return "Success"

    def analyze_method(self, date, file_path):
        """Replace with your actual analysis logic."""
        print(f"Analyzing: {file_path} for date {date}")
        return "Analysis results here"

    def export_method(self, date, file_path, output_path):
        """Replace with your actual export logic."""
        with open(output_path, "w") as f:
            f.write(f"Export from {file_path}\n")
            f.write(f"Date: {date}\n")
        print(f"Exported to: {output_path}")

    def run(self):
        """Start the GUI event loop."""
        self.window.mainloop()


# ===== MAIN ENTRY POINT =====
if __name__ == "__main__":
    app = SimpleGUI()
    app.run()
