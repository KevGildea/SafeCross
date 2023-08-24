import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import math
import matplotlib.pyplot as plt
import re 
import pandas as pd 

# Model parameters
B_ALPHA = -5.317
B_BETA = .405
C_ALPHA = 8.294
C_BETA = -.043

class SafeCrossApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('SafeCross tool')
        self.geometry("750x450")
        self.create_widgets()
        self.create_menu()

    def create_widgets(self):
        # Total count of cyclists
        tk.Label(self, text='Total count of cyclists').pack()
        self.ent1 = tk.Entry(self, width=6)
        self.ent1.pack()

        # Gap width
        tk.Label(self, text='Gap width (mm) (only required for model c)').pack()
        self.ent2 = tk.Entry(self, width=6)
        self.ent2.pack()

        # Sample of crossing angles
        tk.Label(self, text='Sample of crossing angles (degrees) (space separated)').pack()
        self.ent3 = tk.Entry(self, width=100)
        self.ent3.pack()

        # Model c button and output
        tk.Button(self, text='Model c', command=self.NUC_c).pack()
        self.output1 = tk.Text(self, height=1, width=60)
        self.output1.pack()

        # Model b button and output
        tk.Button(self, text='Model b', command=self.NUC_b).pack()
        self.output2 = tk.Text(self, height=1, width=60)
        self.output2.pack()

    def create_menu(self):
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Load angles (.xlsx)", command=self.load_angles_from_xlsx)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        # Add Help menu
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about)  # Add this line
        menubar.add_cascade(label="Help", menu=helpmenu)

    def NUC_c(self):
        valid, message = self.validate_input()
        if not valid:
            self.output1.insert(tk.END, message)
            return

        # Model c logic
        angles = [float(item) for item in re.split(r'\D+', self.ent3.get()) if item]
        x = [float(self.ent2.get()) / math.sin(angle * (math.pi / 180)) for angle in angles]
        ps = [1 - math.exp(C_ALPHA + C_BETA * xi) / (1 + math.exp(C_ALPHA + C_BETA * xi)) for xi in x]
        N_UC = int(self.ent1.get()) * np.average(ps)
        self.output1.insert(tk.END, str(round(N_UC)))
        self.plot_model_c(angles)

    def NUC_b(self):
        valid, message = self.validate_input()
        if not valid:
            self.output2.insert(tk.END, message)
            return

        # Model b logic
        angles = [float(item) for item in re.split(r'\D+', self.ent3.get()) if item]
        ps = [1 - math.exp(B_ALPHA + B_BETA * angle) / (1 + math.exp(B_ALPHA + B_BETA * angle)) for angle in angles]
        N_UC = int(self.ent1.get()) * np.average(ps)
        self.output2.insert(tk.END, str(round(N_UC)))
        self.plot_model_b(angles)

    def load_angles_from_xlsx(self):
        filepath = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if not filepath:
            return
        df = pd.read_excel(filepath)
        if 'Angle' in df.columns:
            angles = [round(angle) for angle in df['Angle'].tolist()]  # Round each angle
            self.ent3.delete(0, tk.END)
            self.ent3.insert(0, ' '.join(map(str, angles)))
        else:
            messagebox.showerror("Error", "The xlsx file does not have a column named 'Angle'.")



    def plot_model_c(self, angles):
        # Plotting logic for Model c
        fig, ax = plt.subplots()
        ax.set_title('Model (c): Gap=' + str(self.ent2.get()) + 'mm', size=12)
        ax.set_ylabel('Probability of crossing success', size=10)
        ax.set_xlabel('Effective width (mm)', size=10)
        
        ang = np.linspace(1, 90, 1000)
        EWs = [float(self.ent2.get()) / math.sin(a * (np.pi / 180)) for a in ang]
        probs = [math.exp(C_ALPHA + C_BETA * EW) / (1 + math.exp(C_ALPHA + C_BETA * EW)) for EW in EWs]
        ax.plot(EWs, probs, linewidth=3, color='black')
        
        # Plot dashed lines for crossing angles
        for angle in angles:
            x = float(self.ent2.get()) / math.sin(angle * (np.pi / 180))
            y = math.exp(C_ALPHA + C_BETA * x) / (1 + math.exp(C_ALPHA + C_BETA * x))
            ax.plot([x, x], [0, y], '--', color='gray')
            ax.plot([0, x], [y, y], '--', color='gray')

        plt.savefig('model_c_plot.png', dpi=300)
        plt.show()

    def plot_model_b(self, angles):
        # Plotting logic for Model b
        fig, ax = plt.subplots()
        ax.set_title('Model (b)', size=12)
        ax.set_ylabel('Probability of crossing success', size=10)
        ax.set_xlabel('Crossing angle (°)', size=10)
        
        ang = np.linspace(1, 90, 1000)
        probs = [math.exp(B_ALPHA + B_BETA * a) / (1 + math.exp(B_ALPHA + B_BETA * a)) for a in ang]
        ax.plot(ang, probs, linewidth=3, color='black')
        
        # Plot dashed lines for crossing angles
        for angle in angles:
            y = math.exp(B_ALPHA + B_BETA * angle) / (1 + math.exp(B_ALPHA + B_BETA * angle))
            ax.plot([angle, angle], [0, y], '--', color='gray')
            ax.plot([0, angle], [y, y], '--', color='gray')

        plt.savefig('model_b_plot.png', dpi=300)
        plt.show()

    def validate_input(self):
        angles = [float(item) for item in re.split(r'\D+', self.ent3.get()) if item]
        if any(angle > 90 for angle in angles):
            return False, 'Error: sample contains value(s) over 90 degrees'
        if any(angle == 0 for angle in angles):
            return False, 'Error: sample contains value(s) of zero degrees'
        return True, ''

    def show_about(self):
        about_window = tk.Toplevel(self)
        about_window.title("About")

        info = (
            "Developed by Kevin Gildea, Ph.D.\n"
            "Faculty of Engineering, LTH\n"
            "Lund University\n"
            "Email: kevin.gildea@tft.lth.se"
        )

        label = tk.Label(about_window, text=info, font=("Arial", 8))
        label.pack(pady=15)

if __name__ == "__main__":
    app = SafeCrossApp()
    app.mainloop()