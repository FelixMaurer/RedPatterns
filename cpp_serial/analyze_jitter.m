%% Setup
clc; clear; close all;

% filename = 'sim_serial_20231027_120000/step_times.csv'; % Uncomment and edit if inside a subfolder
filename = 'step_times.csv'; 

if ~isfile(filename)
    error('File "%s" not found. Make sure to run the C++ simulation first!', filename);
end

%% Load Data
data = readtable(filename);
steps = data.Step;
times = data.Time_ms;

% --- Filter Warmup ---
% We exclude the first 20 steps because they are always slow (allocations/cache misses)
warmup_cutoff = 20;
if length(steps) > warmup_cutoff
    steps_stable = steps(warmup_cutoff+1:end);
    times_stable = times(warmup_cutoff+1:end);
else
    steps_stable = steps;
    times_stable = times;
    warning('Not enough steps to filter warmup period.');
end

%% Calculate Statistics
avg_val = mean(times_stable);
std_val = std(times_stable);
min_val = min(times_stable);
max_val = max(times_stable);

fprintf('========================================\n');
fprintf('       PERFORMANCE ANALYSIS             \n');
fprintf('========================================\n');
fprintf('Total Steps: %d\n', length(steps));
fprintf('Mean Time  : %.4f ms\n', avg_val);
fprintf('Std Dev    : %.4f ms\n', std_val);
fprintf('Min Time   : %.4f ms\n', min_val);
fprintf('Max Time   : %.4f ms\n', max_val);
fprintf('Jitter %%   : %.2f %%\n', (std_val/avg_val)*100);
fprintf('========================================\n');

%% Plot 1: Time Series (Evolution)
figure('Name', 'Step Time Evolution', 'Color', 'w', 'Position', [100 100 800 400]);

plot(steps, times, 'Color', [0.2 0.6 0.8], 'LineWidth', 1); hold on;
yline(avg_val, 'r--', 'LineWidth', 1.5, 'Label', sprintf('Mean: %.3f ms', avg_val));

title('Simulation Step Times (Chronological)');
xlabel('Step Index');
ylabel('Compute Time (ms)');
grid on;
legend('Step Time', 'Mean', 'Location', 'best');
xlim([0 max(steps)]);

%% Plot 2: Histogram (Distribution)
figure('Name', 'Jitter Distribution', 'Color', 'w', 'Position', [100 550 800 400]);

% Automatic binning or fixed number
histogram(times_stable, 50, 'FaceColor', [0.8 0.3 0.3], 'EdgeColor', 'none'); hold on;
xline(avg_val, 'k--', 'LineWidth', 2);

title(sprintf('Jitter Distribution (StdDev: %.3f ms)', std_val));
xlabel('Compute Time (ms)');
ylabel('Frequency (Count)');
grid on;
legend('Distribution', 'Mean');

% Add text box with stats
dim = [.7 .6 .2 .2];
str = sprintf('Mean: %.3f ms\nStd: %.3f ms\nMax: %.3f ms', avg_val, std_val, max_val);
annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', 'BackgroundColor', 'w');