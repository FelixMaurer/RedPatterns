import subprocess
import re
import sys
import os

def main():
    # --- Configuration ---
    executable = "./sim_serial"  
    output_file = "benchmark_serial_results.tsv"

    # --- Environment Setup ---
    # The serial code handles pinning internally via pinToCore(0).
    # We clear OMP variables to ensure no interference.
    env = os.environ.copy()
    keys_to_remove = ["OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_DYNAMIC"]
    for key in keys_to_remove:
        if key in env:
            del env[key]

    # --- Fixed Parameters ---
    isf = "0.00001"
    psi = "0.02"
    it_val = 0.001      # Time step
    no = "200000"       # High output interval to avoid I/O bottlenecks
    d_coeff = "1e-9"

    # Grid sizes: Powers of 2 from 2^7 (128) to 2^13 (8192)
    grid_sizes = [2**i for i in range(7, 14)] 

    results = []

    print(f"=======================================================")
    print(f" Benchmarking Serial CPU: {executable}")
    print(f" Mode: Single-Threaded (Pinned to Core 0)")
    print(f" Optimization: FTZ, Loop Hoisting, Outlier Filtering")
    print(f"=======================================================")
    print(f"{'N':<10} {'Steps':<10} {'Time(ms)':<15} {'StdDev(ms)':<15}")
    print("-" * 60)

    for n in grid_sizes:
        # Calculate Total Time (T) based on your formula
        t_val_float = 15360.0 / float(n) / 12.0
        
        # Calculate number of steps (integers only for display)
        n_steps = int(t_val_float / it_val)

        # Signature: ./sim_serial <N> <ISF> <PSI> <IT> <T> <NO> <D>
        args = [
            executable,
            str(n),
            isf,
            psi,
            str(it_val),
            str(t_val_float),
            no,
            d_coeff
        ]

        try:
            # Run the simulation
            result = subprocess.run(
                args, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env
            )
            
            output = result.stdout
            
            # --- Robust Regex Parsing ---
            # 1. Priority: "Clean Mean" (Outliers filtered)
            clean_mean = re.search(r"Clean Mean\s*:\s+([0-9.]+)\s+ms", output)
            clean_std  = re.search(r"Clean StdDev\s*:\s+([0-9.]+)\s+ms", output)
            
            # 2. Fallback: "Avg Step" (Raw mean if no filtering happened)
            raw_mean = re.search(r"Avg Step\s*:\s+([0-9.]+)\s+ms", output)
            raw_std  = re.search(r"Std Dev\s*:\s+([0-9.]+)\s+ms", output)
            
            val_mean = "N/A"
            val_std = "N/A"

            if clean_mean and clean_std:
                val_mean = clean_mean.group(1)
                val_std = clean_std.group(1)
            elif raw_mean and raw_std:
                val_mean = raw_mean.group(1)
                val_std = raw_std.group(1)
            
            if val_mean != "N/A":
                results.append((n, val_mean, val_std))
                print(f"{n:<10} {n_steps:<10} {val_mean:<15} {val_std:<15}")
            else:
                print(f"{n:<10} {n_steps:<10} {'PARSE ERROR':<15} {'-':<15}")
                # Debug: print last few lines of output
                print(f"--- Output Tail ---\n{output[-200:]}\n-------------------")

        except subprocess.CalledProcessError as e:
            print(f"Error executing N={n}: Return code {e.returncode}")
        except FileNotFoundError:
            print(f"Error: Executable '{executable}' not found.")
            sys.exit(1)

    # --- Save Results to TSV ---
    try:
        with open(output_file, "w") as f:
            f.write("N\tTime_ms\tStdDev_ms\n")
            for row in results:
                f.write(f"{row[0]}\t{row[1]}\t{row[2]}\n")
        print(f"\nResults saved to {output_file}")
    except IOError as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    main()