import subprocess
import re
import sys
import os

def main():
    # --- Configuration ---
    executable = "./sim_cpu"  
    output_file = "benchmark_cpu_results.tsv"

    # --- OpenMP Environment Setup ---
    # These settings are crucial for minimizing standard deviation (jitter)
    env = os.environ.copy()
    env["OMP_DYNAMIC"] = "FALSE"     # Disable dynamic thread adjustment to keep throughput constant
    env["OMP_PROC_BIND"] = "TRUE"    # Bind threads to physical cores to prevent cache thrashing
    env["OMP_NUM_THREADS"] = "8"     # Explicitly set thread count to match your hardware target

    # Fixed Parameters
    isf = "0.00001"
    psi = "0.02"
    it_val = 0.001
    no = "200000"     # High output interval to disable intermediate Disk IO overhead
    d_coeff = "1e-9"

    # Grid sizes: Powers of 2 from 128 to 8192
    grid_sizes = [2**i for i in range(13, 14)] 

    results = []

    print(f"=======================================================")
    print(f" Benchmarking CPU: {executable}")
    print(f" OMP_NUM_THREADS: {env['OMP_NUM_THREADS']} | OMP_PROC_BIND: {env['OMP_PROC_BIND']}")
    print(f"=======================================================")
    print(f"{'N':<10} {'Steps':<10} {'Avg(ms)':<15} {'StdDev(ms)':<15}")
    print("-" * 60)

    for n in grid_sizes:
        # T is inversely proportional to N to keep total workload manageable
        t_val = 15360.0 / float(n) 
        n_steps = int(t_val / it_val)

        # Signature: ./sim_cpu <N> <ISF> <PSI> <IT> <T> <NO> <D>
        args = [
            executable,
            str(n),
            isf,
            psi,
            str(it_val),
            str(t_val),
            no,
            d_coeff
        ]

        try:
            # Run simulation with the explicit OpenMP environment
            result = subprocess.run(
                args, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env
            )
            
            output = result.stdout
            
            # --- Regex Parsing for the PERFORMANCE Block ---
            # Corrected to use group(1) to avoid IndexError
            avg_match = re.search(r"Avg Step:\s+([0-9.]+)\s+ms", output)
            std_match = re.search(r"Std Dev\s*:\s+([0-9.]+)\s+ms", output)
            
            if avg_match and std_match:
                mean_ms = avg_match.group(1)
                std_ms = std_match.group(1) 
                
                results.append((n, mean_ms, std_ms))
                print(f"{n:<10} {n_steps:<10} {mean_ms:<15} {std_ms:<15}")
            else:
                print(f"{n:<10} {n_steps:<10} {'PARSE ERROR':<15} {'-':<15}")
                # Print the end of output to see what the C++ code actually produced
                print(f"--- Debug Snippet ---\n{output[-250:]}\n---------------------") 

        except subprocess.CalledProcessError as e:
            print(f"Error executing N={n}: Return code {e.returncode}")
            print(f"Stderr: {e.stderr}")
        except FileNotFoundError:
            print(f"Error: Executable '{executable}' not found.")
            sys.exit(1)

    # --- Save Results ---
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