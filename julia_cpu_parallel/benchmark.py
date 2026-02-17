import subprocess
import re
import sys
import os

def main():
    # --- Configuration ---
    # Assumes you saved the previous code as cpu_sim.jl
    julia_script = "cpu_sim.jl"   
    output_file = "benchmark_results_julia_cpu.tsv"

    # Fixed Parameters
    isf = "0.00001"
    psi = "0.02"
    it_val = "0.001" 
    no = "200000"     # High output interval to avoid I/O bottlenecks during bench
    d_coeff = "1e-9"

    # Grid sizes: Powers of 2 from 128 to 8192 (2^7 to 2^13)
    # You can adjust this range based on your CPU RAM/Time constraints
    grid_sizes = [2**i for i in range(7, 14)] 
    results = []

    # --- Setup Environment (Threading) ---
    # Crucial for CPU Benchmarks: Ensure Julia knows how many threads to use.
    env = os.environ.copy()
    if "JULIA_NUM_THREADS" not in env:
        # 'auto' lets Julia decide based on available CPU cores
        env["JULIA_NUM_THREADS"] = "auto"
    
    threads_used = env["JULIA_NUM_THREADS"]

    print(f"Starting Benchmark with JULIA_NUM_THREADS={threads_used}")
    print(f"{'N':<10} {'Steps':<10} {'Time(ms)':<15} {'StdDev(ms)':<15}")
    print("-" * 55)

    if not os.path.exists(julia_script):
        print(f"Error: Script '{julia_script}' not found.")
        sys.exit(1)

    for n in grid_sizes:
        # --- Workload Scaling ---
        # Scaling T inversely with N to keep total simulation wall-time manageable
        # while ensuring enough steps run to get a stable average.
        t_val_float = 15360.0 / float(n) / 12
        t_val = str(t_val_float)
        n_steps = int(t_val_float / float(it_val))

        cmd = ["julia", julia_script, str(n), isf, psi, it_val, t_val, no, d_coeff]

        try:
            # Run Julia using the modified environment
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            output = result.stdout
            
            # --- REGEX PARSING ---
            # Looks for: "-> avg step time: 0.4500 ms (+/- 0.0000 ms)"
            # This matches the @printf output in the Julia script
            match = re.search(r"avg step time:\s+([0-9.]+)\s+ms\s+\(\+/-\s+([0-9.]+)\s+ms\)", output)
            
            if match:
                mean_ms = match.group(1)
                std_ms = match.group(2)
                results.append((n, mean_ms, std_ms))
                print(f"{n:<10} {n_steps:<10} {mean_ms:<15} {std_ms:<15}")
            else:
                print(f"{n:<10} {n_steps:<10} {'PARSE ERROR':<15} {'-':<15}")
                # Debug: Uncomment below if you suspect format changes
                # print(output[-300:]) 

        except subprocess.CalledProcessError as e:
            print(f"Error executing N={n}: Return code {e.returncode}")
            # print(e.stderr) # Uncomment to see Julia error trace
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