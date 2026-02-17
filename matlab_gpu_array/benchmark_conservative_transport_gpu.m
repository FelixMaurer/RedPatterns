function benchmark_conservative_transport_gpu()
% BENCHMARK_CONSERVATIVE_TRANSPORT_GPU
% Runs a benchmark suite on run_conservative_transport_gpu.m 
% Matches the logic of the Python/C++ benchmark scripts.

    % --- Configuration ---
    output_file = 'benchmark_matlab_gpu_results.tsv';
    
    % Check for kernel file (required by the simulation)
    if ~exist('kernelInput.dat', 'file')
        % Create a dummy kernel if missing so benchmark doesn't crash
        fprintf('Generating dummy kernelInput.dat for benchmark...\n');
        dlmwrite('kernelInput.dat', rand(31, 1), 'delimiter', '\t');
    end

    % Fixed Parameters
    isf_val = 0.00001;
    psi_val = 0.02;
    it_val  = 0.001;
    no_val  = 200000; % High interval to disable intermediate Disk IO
    d_coeff = 1e-9;
    
    % Grid sizes: Powers of 2 from 128 (2^7) to 8192 (2^13)
    grid_powers = 12:13; 
    grid_sizes = 2.^grid_powers; 
    
    results = [];
    
    fprintf('\n=======================================================\n');
    fprintf(' GPU BENCHMARK: run_conservative_transport_gpu.m\n');
    fprintf('=======================================================\n');
    fprintf('%-10s %-10s %-15s %-15s\n', 'N', 'Steps', 'Avg(ms)', 'StdDev(ms)');
    fprintf('%s\n', repmat('-', 1, 60));
    
    for k = 1:length(grid_sizes)
        n = grid_sizes(k);
        
        % Scale Simulation Time (T) inversely with N to keep total work roughly constant
        % Formula matches original benchmark logic: T = 15360 / N / 12
        t_val = 15360.0 / double(n) / 12;
        
        % Ensure we have at least enough steps for statistics (> 100)
        % The GPU script ignores the first and last 50 steps for timing.
        min_steps = 150;
        calculated_steps = t_val / it_val;
        
        if calculated_steps < min_steps
            t_val = min_steps * it_val;
        end
        
        n_steps = floor(t_val / it_val);
        
        % --- Execution ---
        try
            % Construct command string
            % Signature: (N, [ISF], [PSI], [IT], [T], [NO], [D_coeff])
            cmd = sprintf('run_conservative_transport_gpu(%d, %.10f, %.10f, %.10f, %.10f, %d, %.10f);', ...
                          n, isf_val, psi_val, it_val, t_val, no_val, d_coeff);
            
            % Capture stdout to parse results
            sim_output = evalc(cmd);
            
            % --- Regex Parsing ---
            % Matches: "-> avg step time: 0.4532 ms (+/- 0.0012 ms)"
            pattern = 'avg step time:\s+([0-9.]+)\s+ms\s+\(\+/-\s+([0-9.]+)\s+ms\)';
            tokens = regexp(sim_output, pattern, 'tokens');
            
            if ~isempty(tokens)
                mean_ms = str2double(tokens{1}{1});
                std_ms  = str2double(tokens{1}{2});
                
                % Store results
                results = [results; n, mean_ms, std_ms]; %#ok<AGROW>
                
                % Print row to console
                fprintf('%-10d %-10d %-15.4f %-15.4f\n', n, n_steps, mean_ms, std_ms);
            else
                % Parsing failed
                fprintf('%-10d %-10d %-15s %-15s\n', n, n_steps, 'PARSE ERROR', '-');
                
                % Debugging: Print a snippet of the output
                disp('--- Output Snippet (Last 500 chars) ---');
                disp(sim_output(max(1, end-500):end));
                disp('---------------------------------------');
            end
            
        catch ME
            fprintf('Error executing N=%d: %s\n', n, ME.message);
        end
    end
    
    % --- Save Results to TSV ---
    try
        fid = fopen(output_file, 'w');
        if fid == -1
            error('Cannot open file %s for writing.', output_file);
        end
        
        fprintf(fid, 'N\tTime_ms\tStdDev_ms\n');
        for i = 1:size(results, 1)
            fprintf(fid, '%d\t%.6f\t%.6f\n', results(i, 1), results(i, 2), results(i, 3));
        end
        fclose(fid);
        fprintf('\nResults successfully saved to %s\n', output_file);
        
    catch ME
        fprintf('\nError writing to file: %s\n', ME.message);
    end
end