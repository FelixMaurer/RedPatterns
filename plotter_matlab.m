close all; clear all; clc;

%% 1. User Parameters
% Adjust this variable to point to your target simulation directory
simDirName = 'sim_20260220_080737'; 

% Formatting constants
fs = 1; % scaling factor
fontName = 'cmss10';
mathFont = 'cmmi10';
rootDir = pwd;

%% 2. Color Model Setup
% Parameters from colorModel.chBfit 
% Columns are RGB channels (1=R, 2=G, 3=B). Rows are coefficients b1 through b5.
chBfit = [
    145.7586  134.7227  130.9048;
   -130.4334 -124.1164 -101.2598;
      0.4656   -0.4106   -0.4582;
      0.7014    0.3524    0.4840;
      2.4949    2.2571    5.9188
];

% Model function (taking real part of complex evaluation)
mdlfun = @(b,x) real(b(1) + b(2) * (x - b(3)) / b(4) ./ (1 + ((x - b(3)) / b(4)).^b(5)).^(1/b(5)));

%% 3. Data Import
targetDir = fullfile(rootDir, simDirName);
filelist = dir(fullfile(targetDir, '**', 'psi*.dat'));

if isempty(filelist)
    fprintf('No psi*.dat files found in %s\n', targetDir);
    return;
end

% Sort files alphabetically/numerically to maintain proper time sequence
[~, idx] = sort({filelist.name});
filelist = filelist(idx);

% Read first file to establish grid dimensions
psi_sample = dlmread(fullfile(filelist(1).folder, filelist(1).name));
psis = zeros(length(psi_sample), length(filelist));

for fileIdx = 1:length(filelist)
    filePath = fullfile(filelist(fileIdx).folder, filelist(fileIdx).name);
    psis(:, fileIdx) = dlmread(filePath);
end

% Convert to % 
plotZ = psis * 100; 

% Define physical axes 
T = linspace(0, 20, size(plotZ, 2)); % 20 minutes
X = linspace(0, 6, size(plotZ, 1));  % 6 cm
[plotT, plotX] = meshgrid(T, X);

%% 4. Plotting & Custom Colormap 
hFig = figure('Units', 'centimeters', 'Position', [5*fs, 5*fs, 13.6*fs, 5.42*fs], 'Color', 'w');

surf(plotT, plotX, plotZ, 'EdgeColor', 'none');
view(2);
hold on;

% Determine data bounds
psi_min = max(0.01, min(plotZ(:)));
psi_max = max(plotZ(:));
PsiVals = linspace(psi_min, psi_max, 256);

% Apply Log Transformation
log10PsiVals = log10(PsiVals / 2.22);
log10PsiVals(isinf(log10PsiVals)) = log10(0.001);

% Calculate RGB channels using the embedded model
chR = mdlfun(chBfit(:, 1), log10PsiVals);
chG = mdlfun(chBfit(:, 2), log10PsiVals);
chB = mdlfun(chBfit(:, 3), log10PsiVals);

% Clean, clip to [0, 255], and normalize to [0, 1]
custom_map = [chR', chG', chB'];
custom_map(custom_map > 255) = 255;
custom_map(custom_map < 0) = 0;
custom_map = custom_map / 255;

colormap(custom_map);
cbar = colorbar;

%% 5. Figure Formatting (Thomas Export Style)
ax = gca;
grid off;
set(ax, 'FontName', fontName, 'LineWidth', 1.2*fs, 'Box', 'on', 'Layer', 'top');

% Axis Limits & Ticks
xlim([0 20]);
ylim([0 6]);
xticks([0 5 10 15 20]);
yticks([0 3 6]);
clim([0 psi_max]);

% Labels with TeX fonts
xlabel(['\fontname{', mathFont, '} t \fontname{', fontName, '} [min]'], 'Interpreter', 'tex', 'FontSize', 11*fs);
ylabel(['\fontname{', mathFont, '} x \fontname{', fontName, '} [cm]'], 'Interpreter', 'tex', 'FontSize', 11*fs);

% Precise Axis Positioning
ax.Units = 'centimeters';
ax.Position = [1.1*fs, 1.1*fs, 8.6460, 3.7238];

% Colorbar Formatting
cbar.Units = 'centimeters';
cbar.LineWidth = ax.LineWidth;
cbar.FontName = fontName;
cbar.FontSize = 11*fs;
cbar.Position = [ax.Position(1) + ax.Position(3) + 0.2, ax.Position(2), 0.5*fs, ax.Position(4)];
cbar.Ticks = linspace(0, floor(psi_max/10)*10, 3);

%% 6. Saving
% Name exactly matches the directory parameter provided at the top
saveName = simDirName;

% Export as SVG for vector quality
print(hFig, [saveName, '.svg'], '-dsvg', '-vector');

fprintf('Processed: %s.svg\n', saveName);