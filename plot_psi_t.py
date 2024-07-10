import os

try:
    os.system("awk 'FNR==1 {print}' psi*.dat > psi_t.dat")
    print("Data successfully combined.")
except OSError:
    print("Error executing the AWK command.")

try:
    os.system("gnuplot gnu_plot_script")
    print("Data successfully plotted. Output to psi_t.png")
except OSError:
    print("Error executing the gnuplot command.")