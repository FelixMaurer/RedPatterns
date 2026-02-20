# --------------------------------------------------------------------------
# Usage: gnuplot -c plot_sim.gp <sim_directory>
# Example: gnuplot -c plot_sim.gp sim_20260219_133113
# --------------------------------------------------------------------------

if (ARGC < 1) {
    print "Error: Please provide the simulation directory as an argument."
    print "Usage: gnuplot -c plot_sim.gp sim_20260219_133113"
    exit
}

sim_dir = ARG1
sim_name = system("basename " . sim_dir)

# Gather files in natural version sort (-v) to maintain correct time order
files_flat = system("ls -v " . sim_dir . "/psi*.dat | tr '\n' ' '")
if (strlen(files_flat) == 0) {
    print "Error: No psi*.dat files found in " . sim_dir
    exit
}

# Stacking the single-line files vertically creates a Time (rows) x Space (cols) matrix
cmd = "< cat " . files_flat

# Quietly scan the matrix to find dimensions and limits
stats cmd matrix nooutput name "M"

# Because of cat: rows are time, cols are space
num_cols = M_size_x 
num_rows = M_size_y 

psi_max = M_max * 100.0
psi_min = (M_min * 100.0 < 0.01) ? 0.01 : (M_min * 100.0)

# Failsafe: Prevent Z-axis crash if data is perfectly flat (e.g., all zeros)
if (psi_max <= psi_min) {
    psi_max = psi_min + 1.0
}

# --- Custom Colormap Math ---
# Force variables into the complex plane (+ {0.0, 0.0}) to avoid NaNs on negative roots
u(x, b2, b3) = (x - b2)/b3 + {0.0, 0.0}
cmod(x, b0, b1, b2, b3, b4) = real( b0 + b1 * u(x, b2, b3) / (1.0 + u(x, b2, b3)**b4)**(1.0/b4) )

log10_val(v)  = log10(v / 2.22)
clip(x)       = (x < 0.0) ? 0.0 : ((x > 1.0) ? 1.0 : x)

R_func(v) = clip( cmod(log10_val(v),  145.7586, -130.4334,  0.4656, 0.7014, 2.4949) / 255.0 )
G_func(v) = clip( cmod(log10_val(v),  134.7227, -124.1164, -0.4106, 0.3524, 2.2571) / 255.0 )
B_func(v) = clip( cmod(log10_val(v),  130.9048, -101.2598, -0.4582, 0.4840, 5.9188) / 255.0 )

get_v(gray) = psi_min + gray * (psi_max - psi_min)
set palette functions R_func(get_v(gray)), G_func(get_v(gray)), B_func(get_v(gray))

# --- Formatting & Output ---
set terminal svg size 514,205 font "sans-serif,11" enhanced linewidth 1.2
set output sim_name . ".svg"

set pm3d map interpolate 1,1
unset key

set xrange [0:20]
set yrange [0:6]
set xtics 0, 5, 20
set ytics 0, 3, 6

set xlabel "{/cmmi10 t} {/sans-serif [min]}"
set ylabel "{/cmmi10 x} {/sans-serif [cm]}"

set cbrange [0:psi_max]
set zrange [0:psi_max]
cb_tick_max = floor(psi_max / 10.0) * 10.0
set cbtics 0, (cb_tick_max / 2.0), cb_tick_max

set lmargin at screen 0.081
set bmargin at screen 0.203
set rmargin at screen 0.717
set tmargin at screen 0.890

set colorbox user origin 0.732, 0.203 size 0.037, 0.687

# --- Render Plot ---
# $2 is the row index (Time), $1 is the column index (Space)
splot cmd matrix using ($2/(num_rows-1)*20.0):($1/(num_cols-1)*6.0):($3*100) notitle