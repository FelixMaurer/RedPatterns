import subprocess
import re
import sys

def main():
    # --- Configuration ---
    executable = "./sim_linear"  # Path to your compiled CUDA binary
    output_file = "benchmark_results.tsv"

    # Fixed Parameters
    isf = "0.00001"
    psi = "0.02"
    it_val = 0.001
    no = "200000"     # High output interval to disable intermediate IO
    d_coeff = "1e-9"

    # Grid sizes: Powers of 2 from 128 to 8192
    # 2^7=128 ... 2^13=8192
    grid_sizes = [2**i for i in range(7, 14)] 

    results = []

    print(f"{'N':<10} {'Steps':<10} {'Time(ms)':<15} {'StdDev(ms)':<15}")
    print("-" * 55)

    for n in grid_sizes:
        # Calculate Total Time (T)
        # T should be 120.0 for N=128 and inversely proportional to N
        # Constant = 120.0 * 128 = 15360.0
        t_val = 15360.0 / float(n)
        
        # Calculate expected number of steps (for display/checking)
        n_steps = int(t_val / it_val)

        # Construct Command Arguments
        # Signature: ./sim_linear <N> <ISF> <PSI> <IT> <T> <NO> <D>
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
            # Run simulation and capture output
            result = subprocess.run(
                args, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            output = result.stdout
            
            # Parse Output using Regex
            # Looking for: "-> avg step time: 0.4500 ms (+/- 0.0020 ms)"
            match = re.search(r"avg step time:\s+([0-9.]+)\s+ms\s+\(\+/-\s+([0-9.]+)\s+ms\)", output)
            
            if match:
                mean_ms = match.group(1)
                std_ms = match.group(2)
                
                results.append((n, mean_ms, std_ms))
                print(f"{n:<10} {n_steps:<10} {mean_ms:<15} {std_ms:<15}")
            else:
                print(f"{n:<10} {n_steps:<10} {'PARSE ERROR':<15} {'-':<15}")
                print(f"Debug Output for N={n}:\n{output[-200:]}") # Print last 200 chars

        except subprocess.CalledProcessError as e:
            print(f"Error executing N={n}: Return code {e.returncode}")
            print(e.stderr)
        except FileNotFoundError:
            print(f"Error: Executable '{executable}' not found.")
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