%% Benchmark Plotting Script (Final Update)
% Plots benchmark results for C++, Julia, and MATLAB.
% Saves the resulting figures as PNG and FIG files to the current folder.

clear; clc; close all;

%% 1. Configuration & Data Loading
% Organized logically by Language and Complexity
fileList = { ...
    'cpp_serial.txt',         'cpp_cpu.txt',            'cpp_cuda.txt', ...
    'julia_cpu_serial.txt',   'julia_cpu_parallel.txt', 'julia_cuda.txt', ...
    'matlab_cpu.txt',         'matlab_vectorized.txt',  'matlab_gpu.txt' ...
};

legendNames = { ...
    'C++ Serial',             'C++ CPU (Parallel)',     'C++ CUDA', ...
    'Julia CPU Serial',       'Julia CPU (Parallel)',   'Julia CUDA', ...
    'MATLAB CPU (Loop)',      'MATLAB Vectorized',      'MATLAB GPU' ...
};

% Visual Settings: 9 distinct markers/colors
lineStyles = {'-+', '-o', '-s', '-x', '-d', '-*', '-^', '-v', '->'};
colors = lines(9); 
lineWidth = 1.5;

numFiles = length(fileList);
data = struct(); 

% Load Data
for i = 1:numFiles
    if exist(fileList{i}, 'file')
        raw = load(fileList{i});
        % Sort by grid size (Column 1)
        [~, idx] = sort(raw(:,1));
        sorted_raw = raw(idx, :);
        
        data(i).N = sorted_raw(:,1);    % Grid Size
        data(i).T = sorted_raw(:,2);    % Time per step (ms)
        data(i).Std = sorted_raw(:,3);  % Std Dev (ms)
        data(i).Name = legendNames{i};
    else
        warning('File %s not found. Generating dummy data.', fileList{i});
        % Robust dummy data generation for demo purposes
        N_dummy = [128, 256, 512, 1024, 2048, 4096, 8192]';
        baseTime = (N_dummy.^2); 
        
        fname = fileList{i};
        if contains(fname, 'serial'), factor = 1.5;
        elseif contains(fname, 'vectorized'), factor = 0.5;
        elseif contains(fname, 'cuda'), factor = 0.002;
        elseif contains(fname, 'parallel'), factor = 0.1;
        else, factor = 0.8; end
        
        data(i).N = N_dummy;
        data(i).T = baseTime * factor; 
        data(i).Std = data(i).T * 0.05; 
        data(i).Name = legendNames{i};
    end
end

% Get Grid Sizes for Axis Ticks
gridSizes = data(1).N;

%% 2. Figure 1: Log-Log Scale (Time vs Grid Size)
f1 = figure('Name', 'Benchmark: Log-Log Scale', 'Color', 'w', 'Position', [100 100 900 600]);
hold on; grid on;

for i = 1:numFiles
    errorbar(data(i).N, data(i).T, data(i).Std, ...
        lineStyles{i}, 'Color', colors(i,:), ...
        'LineWidth', lineWidth, 'MarkerSize', 8, 'CapSize', 8);
end

set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Grid Size (N) [Log Scale]');
ylabel('Time per Step (ms) [Log Scale]');
title('Simulation Performance (Log-Log Plot)');
legend(legendNames, 'Location', 'northwest', 'NumColumns', 1);

% --- Formatting X-Axis ---
xticks(gridSizes);
xticklabels(string(gridSizes)); 
xtickangle(0); 
xlim([min(gridSizes)*0.9, max(gridSizes)*1.1]);
% -------------------------

set(gca, 'FontSize', 12);
hold off;

% --- SAVE FIGURE 1 ---
saveas(f1, 'benchmark_loglog_time.png');
saveas(f1, 'benchmark_loglog_time.fig');
fprintf('Saved Figure 1 to benchmark_loglog_time.png/.fig\n');

%% 3. Figure 2: Speedup Ratio (Log-Log Scale)
% Baseline: C++ CUDA
cudaIdx = find(strcmp(fileList, 'cpp_cuda.txt'));
if isempty(cudaIdx), cudaIdx = 3; end % Fallback

refTime = data(cudaIdx).T; 

f2 = figure('Name', 'Benchmark: Speedup Ratio', 'Color', 'w', 'Position', [150 150 900 600]);
hold on; grid on;

for i = 1:numFiles
    % Ratio = Time_Current / Time_Reference (Slowdown Factor)
    ratio = data(i).T ./ refTime;
    
    plot(data(i).N, ratio, lineStyles{i}, ...
        'Color', colors(i,:), 'LineWidth', lineWidth, 'MarkerSize', 8);
end

yline(1, '--k', 'Baseline (C++ CUDA)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left');

xlabel('Grid Size (N)');
ylabel('Slowdown Factor (Relative to C++ CUDA) [Log Scale]');
title({'Performance Comparison Relative to Fastest Implementation', '(Lower is Better / Closer to 1 is Faster)'});
legend(legendNames, 'Location', 'bestoutside'); 

% --- Formatting Axes ---
set(gca, 'XScale', 'log', 'YScale', 'log'); 
xticks(gridSizes);
xticklabels(string(gridSizes)); 
xtickangle(0); 
xlim([min(gridSizes)*0.9, max(gridSizes)*1.1]);
% -----------------------

set(gca, 'FontSize', 12);
hold off;

% --- SAVE FIGURE 2 ---
saveas(f2, 'benchmark_speedup_ratio.png');
saveas(f2, 'benchmark_speedup_ratio.fig');
fprintf('Saved Figure 2 to benchmark_speedup_ratio.png/.fig\n');