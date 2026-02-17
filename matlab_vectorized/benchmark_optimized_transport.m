function benchmark_optimized_transport()
% BENCHMARK_OPTIMIZED_TRANSPORT
% Runs a benchmark suite on run_conservative_transport_optimized.m 
% matching the parameters and logic of the provided benchmarking script.

    % --- Configuration ---
    output_file = 'benchmark_optimized_results.tsv';
    
    % Fixed Parameters (matching Python/standard benchmark script)
    isf_val = 0.00001;
    psi_val = 0.02;
    it_val  = 0.001;
    no_val  = 200000; % High interval to disable intermediate Disk IO
    d_coeff = 1e-9;
    
    % Grid sizes: Powers of 2. 
    % Note: The optimized code is efficient enough to run the full range if desired.
    % Adjust this range as needed (e.g., 7:13 for 128 to 8192).
    grid_powers = 7:13; 
    grid_sizes = 2.^grid_powers; 
    
    results = [];

    fprintf('=======================================================\n');
    fprintf(' Benchmarking MATLAB: run_conservative_transport_optimized\n');
    fprintf('=======================================================\n');
    fprintf('%-10s %-10s %-15s %-15s\n', 'N', 'Steps', 'Avg(ms)', 'StdDev(ms)');
    fprintf('%s\n', repmat('-', 1, 60));

    for k = 1:length(grid_sizes)
        n = grid_sizes(k);
        
        % T is inversely proportional to N to keep total workload manageable.
        % Matches the scaling in your previous snippet (including the /12 factor).
        t_val = 15360.0 / double(n) / 12;
        n_steps = floor(t_val / it_val);
        
        % Prepare arguments for run_conservative_transport_optimized
        % Signature: (N, [ISF], [PSI], [IT], [T], [NO], [D_coeff])
        
        try
            % We use evalc to capture the standard output (stdout)
            cmd = sprintf('run_conservative_transport_optimized(%d, %.10f, %.10f, %.10f, %.10f, %d, %.10f);', ...
                          n, isf_val, psi_val, it_val, t_val, no_val, d_coeff);
            
            % Capture output
            sim_output = evalc(cmd);
            
            % --- Regex Parsing ---
            % Looking for: "-> avg step time: 12.3456 ms (+/- 0.1234 ms)"
            pattern = 'avg step time:\s+([0-9.]+)\s+ms\s+\(\+/-\s+([0-9.]+)\s+ms\)';
            tokens = regexp(sim_output, pattern, 'tokens');
            
            if ~isempty(tokens)
                mean_ms_str = tokens{1}{1};
                std_ms_str  = tokens{1}{2};
                
                % Store results
                results = [results; n, str2double(mean_ms_str), str2double(std_ms_str)]; %#ok<AGROW>
                
                % Print to console
                fprintf('%-10d %-10d %-15s %-15s\n', n, n_steps, mean_ms_str, std_ms_str);
            else
                fprintf('%-10d %-10d %-15s %-15s\n', n, n_steps, 'PARSE ERROR', '-');
                % Debug: print tail of output if parse fails
                disp('--- Debug Output Snippet ---');
                disp(sim_output(max(1, end-500):end));
                disp('----------------------------');
            end
            
        catch ME
            fprintf('Error executing N=%d: %s\n', n, ME.message);
        end
    end

    % --- Save Results ---
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
        fprintf('\nResults saved to %s\n', output_file);
        
    catch ME
        fprintf('\nError writing to file: %s\n', ME.message);
    end

end