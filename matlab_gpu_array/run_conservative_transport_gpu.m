function run_conservative_transport_gpu(varargin)
% RUN_CONSERVATIVE_TRANSPORT_GPU GPU-Accelerated Finite Volume Transport
% 
% Requirements: Parallel Computing Toolbox and a compatible NVIDIA GPU.
% Performance: optimized via gpuArray, pre-calculated grids, and vectorized convolution.

    %% 1. Parameters and Constants
    args = {256, 1.000, 0.02, 0.005, 1200.0, 1000, 0.0};
    for k = 1:min(length(varargin), length(args))
        if ~isempty(varargin{k}), args{k} = varargin{k}; end
    end
    [N, ISF, PSI, IT, T, NO_interval, D_coeff] = args{:};
    
    % Check for GPU
    if parallel.gpu.GPUDevice.count() < 1
        error('No GPU detected. This script requires a compatible GPU.');
    end
    
    % Derived constants
    subDiv = 65536.0 / double(N);
    M = floor(N * subDiv + 1);
    wingL = floor(N * (34.0 / 256.0));
    SYS_L = M * 1.041412353515625e-6;
    gradL = 0.06;
    zShift = (SYS_L - gradL) / 2.0;
    
    RL = 30.0; RC = 1100.0; P0 = 1100.0;
    IZ = SYS_L / (N - 1);
    IR = RL / (N - 1);
    NT = ceil(T / IT);
    
    h_beta = 7.4e23;
    h_alpha = 12e-5;
    kernelN = 31;
    Rsigma = 2.0; 
    
    %% 2. Setup
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    outDir = sprintf('sim_gpu_%s/', timestamp);
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    
    g = gpuDevice;
    fprintf('\n============================================================\n');
    fprintf('   GPU ACCELERATED FINITE VOLUME TRANSPORT\n');
    fprintf('============================================================\n');
    fprintf('  Device          : %s\n', g.Name);
    fprintf('  grid size (n)   : %d\n', N);
    fprintf('  subgrid (m)     : %d\n', M);
    fprintf('  dt (time step)  : %.5e\n', IT);
    fprintf('------------------------------------------------------------\n');

    %% 3. Initialization (GPU)
    
    % Pre-calculate coordinate vectors on GPU
    j_idx_vec = gpuArray(0:N-1);
    R = RC - RL/2 + RL * (j_idx_vec / (N-1));
    
    % Initialize Density Field (Directly on GPU)
    phi = initPhiGPU(N, wingL, IZ, IR, Rsigma, PSI);
    
    % Load Interaction Kernel
    kernelInFile = 'kernelInput.dat';
    if exist(kernelInFile, 'file') ~= 2
        error('fatal error: could not open kernel file ''%s''', kernelInFile);
    end
    % Kernel processing
    rawKernel = load(kernelInFile); 
    
    % FIX: Ensure rawKernel is a column vector
    if ~iscolumn(rawKernel)
        rawKernel = rawKernel'; 
    end

    if length(rawKernel) < kernelN
        error('kernel file has only %d values (need %d)', length(rawKernel), kernelN);
    end
    
    % Prepare kernel for conv2 (flip for correlation -> convolution equivalence)
    cpuKernel = rawKernel(1:kernelN) * ISF;
    intKernel_gpu = gpuArray(flipud(cpuKernel)); % Now flipud correctly flips the column
    % Save kernel (CPU IO)
    saveNVecToDrive(cpuKernel, outDir, 'intKernel.dat');
    
    % Static Gradient Profiles
    percoll_cpu = computePercollLinear(N, IZ, zShift, gradL);
    gradWing_cpu = computeWing(N, wingL, percoll_cpu);
    
    % Move Masks and Static Fields to GPU
    gradWing = gpuArray(gradWing_cpu);
    percoll = gpuArray(percoll_cpu);
    
    idx_vec = gpuArray((1:N)');
    mask_center = (idx_vec > (wingL + 1)) & (idx_vec < (N - wingL));
    
    P_eff = gradWing;
    P_eff(mask_center) = percoll(mask_center);
    
    % --- PRE-COMPUTATION OPTIMIZATION ---
    % 1. Static portion of Velocity Grid: -alpha * (R + P - P0)
    % This does not change over time. Calculate once.
    rp_grid_term = -h_alpha * (bsxfun(@plus, P_eff, R) - P0);
    
    % 2. Spline Evaluation Map (Constant for fixed M, N, subDiv)
    % We compute the mapping from grid M to grid N once.
    k_spline = gpuArray(0:M-1);
    x_spline = k_spline / subDiv;
    j_spline = floor(x_spline);
    j_idx_map = min(j_spline + 1, N);      % Indices into N
    j_idx_map = j_idx_map(:);
    dx_map = (x_spline(:) - (j_idx_map - 1)); % dx offsets
    
    % 3. Downsampling indices
    ds_indices = floor(gpuArray(0:N-1)' * subDiv) + 1;
    
    % 4. Pre-allocate Flux Arrays (GPU)
    J = gpuArray.zeros(N, N);
    K = gpuArray.zeros(N, N);
    
    %% 4. Simulation Loop
    fprintf('  -> starting simulation loop...\n\n');
    sim_start_time = tic;
    step_times = [];
    t = 0.0;
    
    % Pre-define constants for loop
    halo = floor((kernelN - 1) / 2);
    pad_pre = gpuArray.zeros(halo, 1);
    pad_post = gpuArray.zeros(halo, 1);
    
    for step = 0:NT
        is_output_step = (mod(step-1, NO_interval) == 0) || (step == 1) || (step == NT);
        measure_step = (step > 50) && (step < (NT - 50)) && (~is_output_step);
        
        if measure_step, t_step_start = tic; end
        
        % --- PHYSICS ENGINE (GPU) ---
        
        % 1. Integration
        psi = sum(phi, 2); 
        
        % 2. Spline Coefficients (Hybrid Approach)
        % Thomas algorithm is serial and fast on CPU for small N=256.
        psi_cpu = gather(psi); 
        [b_cpu, c_cpu, d_cpu] = computeSplineCoeffsCPU(psi_cpu, N);
        
        % 3. Spline Evaluation (GPU)
        % Move coeffs to GPU and evaluate on large M grid
        b_gpu = gpuArray(b_cpu);
        c_gpu = gpuArray(c_cpu);
        d_gpu = gpuArray(d_cpu);
        
        % Vectorized evaluation using pre-computed maps
        % psiIntp = y + (b + (c + d*dx)*dx)*dx
        y_val = psi(j_idx_map);
        b_val = b_gpu(j_idx_map);
        c_val = c_gpu(j_idx_map);
        d_val = d_gpu(j_idx_map);
        
        psiIntp = y_val + (b_val + (c_val + d_val .* dx_map) .* dx_map) .* dx_map;
        
        % 4. Convolution (GPU Optimized)
        % Pad
        psi_padded = [pad_pre; psiIntp; pad_post];
        
        % Use conv2 with 'valid'. intKernel_gpu is already flipped.
        % This replaces the manual loop.
        IIntp = conv2(psi_padded, intKernel_gpu, 'valid');
        IIntp = IIntp * (IZ / subDiv);
        
        % 5. Downsampling
        I = IIntp(ds_indices);
        
        % 6. Flux Calculation
        % v = rp_grid_term - beta * I
        v_cell = rp_grid_term - h_beta * repmat(I, 1, N);
        
        % Axial Flux J
        v_face_z = 0.5 * (v_cell(1:end-1, :) + v_cell(2:end, :));
        
        % Upwind Scheme (Branchless GPU optimized)
        phi_curr = phi(1:end-1, :);
        phi_next = phi(2:end, :);
        pos_mask = v_face_z > 0;
        
        % Fused multiply-add is faster than indexed assignment
        phi_up = phi_curr .* pos_mask + phi_next .* (~pos_mask);
        
        J_adv = v_face_z .* phi_up;
        J_diff = -D_coeff * (phi_next - phi_curr) / IZ;
        
        J(1:end-1, :) = J_adv + J_diff;
        
        % Radial Flux K
        if D_coeff ~= 0
            K(:, 1:end-1) = -D_coeff * (phi(:, 2:end) - phi(:, 1:end-1)) / IR;
        end
        
        % 7. Update
        % div_z
        div_z = (J - [gpuArray.zeros(1, N); J(1:end-1, :)]) / IZ;
        
        % div_r
        div_r = (K - [gpuArray.zeros(N, 1), K(:, 1:end-1)]) / IR;
        
        phi = phi - IT * (div_z + div_r);
        phi(phi < 0) = 0;
        
        % --- END PHYSICS ---
        
        if measure_step
            % wait for GPU to finish for accurate timing
            wait(g); 
            step_times(end+1) = toc(t_step_start) * 1000.0;
        end
        
        if is_output_step
            elapsed = toc(sim_start_time);
            printProgressBar(step, NT, elapsed);
            
            % Gather only needed data to CPU for writing
            phi_save = gather(phi);
            psi_save = gather(psi);
            
            saveArrToDrive(phi_save, outDir, sprintf('phi_%010d.dat', step), N);
            saveVecToDrive(psi_save, outDir, sprintf('psi_%010d.dat', step));
            
            % Save static files only once or if needed (optimized to not re-gather)
            % Save static profiles (CPU arrays, so no gather needed)
            saveVecToDrive(gradWing_cpu, outDir, sprintf('gw_%010d.dat', step));
            saveVecToDrive(percoll_cpu, outDir, sprintf('gp_%010d.dat', step));
        end
        t = t + IT;
    end
    
    elapsed = toc(sim_start_time);
    printProgressBar(NT, NT, elapsed);
    fprintf('\n\n  -> simulation complete.\n');
    
    if ~isempty(step_times)
        avg_t = mean(step_times);
        std_t = std(step_times);
        fprintf('  -> avg step time: %.4f ms (+/- %.4f ms)\n', avg_t, std_t);
        fprintf('  -> statistics based on %d pure computation steps\n', length(step_times));
    else
        fprintf('  -> warning: not enough steps for statistics.\n');
    end
end

%% Optimized Helper Functions

function [b, c, d] = computeSplineCoeffsCPU(y, N)
    % Solves the Thomas algorithm for Spline coefficients.
    % Kept on CPU because N=256 is too small for GPU parallelization overhead.
    alp = zeros(N, 1);
    y_diff = diff(y);
    alp(2:N-1) = 3.0 * y_diff(2:end) - 3.0 * y_diff(1:end-1);
    
    mu = zeros(N, 1); 
    ze = zeros(N, 1);
    
    % Serial dependency (Forward sweep)
    for i = 2:N-1
        inv_mu = 1.0 / (4.0 - mu(i-1));
        mu(i) = inv_mu;
        ze(i) = (alp(i) - ze(i-1)) * inv_mu;
    end
    
    c = zeros(N, 1); 
    b = zeros(N, 1); 
    d = zeros(N, 1);
    
    % Serial dependency (Back substitution)
    for j = N-1:-1:1
        c(j) = ze(j) - mu(j) * c(j+1);
        c_j = c(j); c_jp1 = c(j+1);
        b(j) = (y(j+1) - y(j)) - (c_jp1 + 2.0 * c_j) / 3.0;
        d(j) = (c_jp1 - c_j) / 3.0;
    end
end

function phi = initPhiGPU(N, wingL, deltaI, deltaJ, Rsigma, PSI)
    % Fully GPU-vectorized initialization
    edgeZ = wingL + 2; edgeR = wingL;
    
    [j_grid, i_grid] = meshgrid(gpuArray(0:N-1), gpuArray(0:N-1)); 
    zVal = i_grid * deltaI;
    rVal = j_grid * deltaJ;
    
    scaleSigmaZ = 0.6; scaleSigmaR = 0.7;
    posZ1 = 0.25; posR1 = 0.45;
    posZ2 = 0.75; posR2 = 0.55;
    
    totalZ = (N-1)*deltaI; totalR = (N-1)*deltaJ;
    
    baseSigmaIdx = Rsigma / deltaJ;
    sIdxZ = baseSigmaIdx * scaleSigmaZ;
    sIdxR = baseSigmaIdx * scaleSigmaR;
    denZ = 2.0 * sIdxZ^2; denR = 2.0 * sIdxR^2;
    
    % Element-wise exp on GPU
    g1 = exp(-(((zVal - totalZ*posZ1)/deltaI).^2/denZ + ((rVal - totalR*posR1)/deltaJ).^2/denR));
    g2 = exp(-(((zVal - totalZ*posZ2)/deltaI).^2/denZ + ((rVal - totalR*posR2)/deltaJ).^2/denR));
    
    phi = g1 + g2;
    
    mask = (i_grid < edgeZ) | (i_grid > (N-1-edgeZ)) | (j_grid < edgeR) | (j_grid > (N-1-edgeR));
    phi(mask) = 0.0;
    
    phiSum = sum(phi(:));
    if phiSum > 0
        normFactor = PSI * (N - 2 * edgeZ);
        phi = phi * (normFactor / phiSum);
    end
end

% --- Standard Helper Functions (CPU Logic for IO/Setup) ---

function percoll = computePercollLinear(N, IZ, zShift, gradL)
    i = 0:N-1;
    x = IZ * i;
    PL = 8.0;
    percoll = (x - zShift - gradL/2) / (gradL/2) * PL/2;
    percoll = percoll(:);
end

function gradWing = computeWing(N, wingL, percoll)
    gradWing = zeros(N, 1);
    c_wingL = wingL;
    r3 = percoll(c_wingL + 1) - percoll(c_wingL);
    r2 = percoll(c_wingL + 1);
    r1 = r2 - 10;
    x1 = 20; x2 = c_wingL;
    if percoll(x1 + 1) < r1, r1 = percoll(x1 + 1); end
    a = (r1 - r2 + r3*(x2-x1))/((x1-x2)^2);
    b = r3 - 2*a*x2;
    c = r2 - r3*x2 + x2^2*a;
    i = (0:N-1)';
    mask1 = i <= c_wingL;
    gradWing(mask1) = a*i(mask1).^2 + b*i(mask1) + c;
    mask2 = i >= (N - 1 - c_wingL);
    idx_r = N - 1 - i(mask2);
    gradWing(mask2) = -(a*idx_r.^2 + b*idx_r + c);
    clamp_h = (N - 1 - 13) + 1;
    gradWing(i >= N - 14) = gradWing(clamp_h);
    gradWing(i <= 13) = gradWing(14);
end

function saveArrToDrive(f, dirName, fileName, N)
    sampleSkip = ceil(double(N)/256.0);
    % Ensure data is on CPU before writing
    if isa(f, 'gpuArray'), f = gather(f); end
    dlmwrite(fullfile(dirName, fileName), f(1:sampleSkip:end, 1:sampleSkip:end), 'delimiter', '\t', 'precision', '%e');
end

function saveVecToDrive(f, dirName, fileName)
    sampleSkip = ceil(double(length(f))/256.0);
    if isa(f, 'gpuArray'), f = gather(f); end
    f_sub = f(1:sampleSkip:end);
    if iscolumn(f_sub), f_sub = f_sub'; end
    dlmwrite(fullfile(dirName, fileName), f_sub, 'delimiter', '\t', 'precision', '%e');
end

function saveNVecToDrive(f, dirName, fileName)
    if isa(f, 'gpuArray'), f = gather(f); end
    if iscolumn(f), f = f'; end
    dlmwrite(fullfile(dirName, fileName), f, 'delimiter', '\t', 'precision', '%e');
end

function printProgressBar(step, total, elapsed_sec)
    barWidth = 40;
    progress = step / total;
    fprintf('\r[');
    pos = floor(barWidth * progress);
    for i = 0:barWidth-1
        if i < pos, fprintf('='); elseif i == pos, fprintf('>'); else, fprintf(' '); end
    end
    fprintf('] %d %% ', floor(progress * 100));
    if step > 0
        eta_sec = (elapsed_sec / step) * (total - step);
        h = floor(eta_sec / 3600); m = floor(mod(eta_sec, 3600) / 60); s = floor(mod(eta_sec, 60));
        fprintf('| eta: %02d:%02d:%02d', h, m, s);
    end
    if step == total, fprintf('\n'); end
end