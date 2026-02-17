import subprocess
import re
import sys
import os

def main():
    # --- Configuration ---
    # Points to the serial version created in the previous step
    julia_script = "serial_sim.jl"   
    output_file = "benchmark_results_julia_serial.tsv"

    # Fixed Parameters
    isf = "0.00001"
    psi = "0.02"
    it_val = "0.001" 
    no = "200000"     # High output interval to avoid I/O bottlenecks
    d_coeff = "1e-9"

    # Grid sizes: Powers of 2 from 128 to 8192 (2^7 to 2^13)
    grid_sizes = [2**i for i in range(7, 14)] 
    results = []

    # --- Setup Environment ---
    # For the serial script, JULIA_NUM_THREADS doesn't strictly matter,
    # but setting it to 1 is good practice to ensure no implicit threading occurs.
    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = "1"
    
    print(f"Starting Serial Benchmark (Target: {julia_script})")
    print(f"{'N':<10} {'Steps':<10} {'Time(ms)':<15} {'StdDev(ms)':<15}")
    print("-" * 55)

    if not os.path.exists(julia_script):
        print(f"Error: Script '{julia_script}' not found.")
        sys.exit(1)

    for n in grid_sizes:
        # --- Workload Scaling ---
        # Removed the '/ 12' factor to ensure Total Steps > 100.
        # The simulation skips the first/last 50 steps for warmup cleanup.
        # If steps < 100, no data is collected.
        t_val_float = 15360.0 / float(n) / 12
        t_val = str(t_val_float)
        n_steps = int(t_val_float / float(it_val))

        cmd = ["julia", julia_script, str(n), isf, psi, it_val, t_val, no, d_coeff]

        try:
            # Run Julia
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            output = result.stdout
            
            # --- REGEX PARSING ---
            # Looks for: "-> avg step time: 0.4500 ms (+/- 0.0000 ms)"
            match = re.search(r"avg step time:\s+([0-9.]+)\s+ms\s+\(\+/-\s+([0-9.]+)\s+ms\)", output)
            
            if match:
                mean_ms = match.group(1)
                std_ms = match.group(2)
                results.append((n, mean_ms, std_ms))
                print(f"{n:<10} {n_steps:<10} {mean_ms:<15} {std_ms:<15}")
            else:
                print(f"{n:<10} {n_steps:<10} {'PARSE ERROR':<15} {'-':<15}")
                # Debug: Check if 'warning: not enough steps' appeared
                if "not enough steps" in output:
                    print(f"   [!] Warning: Simulation too short (Steps={n_steps})")

        except subprocess.CalledProcessError as e:
            print(f"Error executing N={n}: Return code {e.returncode}")
        except KeyboardInterrupt:
            print("\nBenchmark interrupted.")
            sys.exit(1)

    # --- Write to File ---
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