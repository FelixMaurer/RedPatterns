function run_conservative_transport_optimized(varargin)
% RUN_CONSERVATIVE_TRANSPORT_OPTIMIZED
% High-Performance Vectorized CPU implementation.
% Uses implicit AVX/Multithreading for maximum single-node throughput.

    %% 1. Parameters and Constants
    args = {256, 1.000, 0.02, 0.005, 1200.0, 1000, 0.0};
    for k = 1:min(length(varargin), length(args))
        if ~isempty(varargin{k}), args{k} = varargin{k}; end
    end
    [N, ISF, PSI, IT, T, NO_interval, D_coeff] = args{:};
    
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
    outDir = sprintf('sim_opt_%s/', timestamp);
    if ~exist(outDir, 'dir'), mkdir(outDir); end
    
    fprintf('\n============================================================\n');
    fprintf('   matlab simulation: optimized vectorized cpu              \n');
    fprintf('============================================================\n');
    fprintf('  grid size (n)   : %d\n', N);
    fprintf('  time steps (nt) : %d\n', NT);
    fprintf('  threads         : %d (Implicit)\n', maxNumCompThreads);
    fprintf('------------------------------------------------------------\n');

    %% 3. Initialization
    j_idx = 0:N-1;
    R = RC - RL/2 + RL * (j_idx / (N-1));
    
    phi = initPhi(N, wingL, IZ, IR, Rsigma, PSI);
    
    % Load Kernel
    kernelInFile = 'kernelInput.dat';
    if exist(kernelInFile, 'file') ~= 2, error('Missing kernelInput.dat'); end
    fid = fopen(kernelInFile, 'r');
    rawKernel = fscanf(fid, '%f');
    fclose(fid);
    intKernel = rawKernel(1:kernelN) * ISF;
    saveNVecToDrive(intKernel, outDir, 'intKernel.dat');
    
    % Preallocate Arrays to prevent reallocation churn
    J = zeros(N, N); 
    K = zeros(N, N); 
    J_in = zeros(N, N);
    K_in = zeros(N, N);
    
    % Static Gradients
    percoll = computePercollLinear(N, IZ, zShift, gradL);
    gradWing = computeWing(N, wingL, percoll);
    
    % Optimization: Pre-calculate Pressure Grid
    idx_vec = (1:N)';
    mask_center = (idx_vec > (wingL + 1)) & (idx_vec < (N - wingL));
    P_eff = gradWing;           
    P_eff(mask_center) = percoll(mask_center); 
    
    % Pre-calculate rp_grid (Pressure + Radius) as it is time-invariant
    % rp_grid(i,j) = P_eff(i) + R(j) - P0
    rp_grid = bsxfun(@plus, P_eff, R) - P0; 
    
    % Pre-calculate Velocity Base Component
    v_base = -h_alpha * rp_grid;
    
    % Convolution Optimization
    % We use 'conv' with 'valid'. To match C++ correlation (sliding dot product),
    % we must flip the kernel.
    % C++: sum(psi[i+k]*kernel[k])  ->  Matlab: conv(psi, flip(kernel))
    intKernelFlipped = flipud(intKernel);
    
    ds_indices = floor((0:N-1)' * subDiv) + 1;
    halo = floor((kernelN - 1) / 2);

    %% 4. Simulation Loop
    fprintf('  -> starting simulation loop...\n\n');
    sim_start_time = tic;
    step_times = [];
    t = 0.0;

    for step = 0:NT
        is_output_step = (mod(step-1, NO_interval) == 0) || (step == 1) || (step == NT);
        measure_step = (step > 50) && (step < (NT - 50)) && (~is_output_step);
        if measure_step, t_step_start = tic; end
        
        % --- PHYSICS ENGINE ---
        
        % 1. Integration (Sum rows)
        psi = sum(phi, 2); 
        
        % 2. Spline Interpolation
        psiIntp = splineInterpolate(psi, N, M, subDiv);
        
        % 3. Convolution (Vectorized compiled C)
        % Pad carefully to match C++ halo
        psi_padded = [zeros(halo, 1); psiIntp; zeros(halo, 1)];
        % 'valid' returns size M (because length(psi_padded) = M + 2*halo)
        IIntp = conv(psi_padded, intKernelFlipped, 'valid'); 
        IIntp = IIntp * (IZ / subDiv);
        
        % 4. Downsampling
        I = IIntp(ds_indices); 
        
        % 5. Flux Calculation
        % v = v_base - beta * I.  (I is broadcasted across columns)
        v_cell = bsxfun(@minus, v_base, h_beta * I);
        
        % Axial Flux J (at interfaces i+1/2)
        % Average velocity at face
        v_face_z = 0.5 * (v_cell(1:end-1, :) + v_cell(2:end, :));
        
        % Optimized Upwind Slicing
        % Avoid creating new zero arrays. Reuse memory where possible.
        phi_curr = phi(1:end-1, :);
        phi_next = phi(2:end, :);
        
        % Default to Next
        phi_up = phi_next; 
        % Overwrite where velocity is positive
        pos_v = v_face_z > 0;
        phi_up(pos_v) = phi_curr(pos_v);
        
        J_adv = v_face_z .* phi_up;
        J_diff = -D_coeff * (phi_next - phi_curr) / IZ;
        
        J(1:end-1, :) = J_adv + J_diff;
        J(end, :) = 0; 
        
        % Radial Flux K
        if D_coeff ~= 0
            K(:, 1:end-1) = -D_coeff * (phi(:, 2:end) - phi(:, 1:end-1)) / IR;
            K(:, end) = 0;
        end
        % Note: If D=0, K is already zeros from pre-allocation.
        
        % 6. Update
        % div_z = (J(i) - J(i-1)) / dz
        J_in(2:end, :) = J(1:end-1, :); % Shift J to get J(i-1)
        div_z = (J - J_in) / IZ;
        
        % div_r = (K(j) - K(j-1)) / dr
        K_in(:, 2:end) = K(:, 1:end-1); % Shift K to get K(j-1)
        div_r = (K - K_in) / IR;
        
        phi = phi - IT * (div_z + div_r);
        phi(phi < 0) = 0; 
        
        % --- END PHYSICS ---

        if measure_step
            step_times(end+1) = toc(t_step_start) * 1000.0;
        end
        
        if is_output_step
            elapsed = toc(sim_start_time);
            printProgressBar(step, NT, elapsed);
            saveArrToDrive(phi, outDir, sprintf('phi_%010d.dat', step), N);
            saveVecToDrive(psi, outDir, sprintf('psi_%010d.dat', step));
            saveVecToDrive(gradWing, outDir, sprintf('gw_%010d.dat', step));
            saveVecToDrive(percoll, outDir, sprintf('gp_%010d.dat', step));
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
    end
end

%% Helper Functions (Identical to before)

function psiIntp = splineInterpolate(y, N, M, subDiv)
    y = y(:);
    alp = zeros(N, 1);
    y_diff = diff(y);
    alp(2:N-1) = 3.0 * y_diff(2:end) - 3.0 * y_diff(1:end-1);
    
    mu = zeros(N, 1); ze = zeros(N, 1);
    for i = 2:N-1
        mu(i) = 1.0 / (4.0 - mu(i-1));
        ze(i) = (alp(i) - ze(i-1)) / (4.0 - mu(i-1));
    end
    
    c_coef = zeros(N, 1); b_coef = zeros(N, 1); d_coef = zeros(N, 1);
    for j = N-1:-1:1
        c_coef(j) = ze(j) - mu(j) * c_coef(j+1);
        b_coef(j) = (y(j+1) - y(j)) - (c_coef(j+1) + 2.0 * c_coef(j)) / 3.0;
        d_coef(j) = (c_coef(j+1) - c_coef(j)) / 3.0;
    end
    
    k = 0:M-1;
    x = k / subDiv;
    j = floor(x);
    j_idx = min(j + 1, N); 
    dx = (x - (j_idx - 1))'; 
    j_idx = j_idx(:);
    psiIntp = y(j_idx) + (b_coef(j_idx) + (c_coef(j_idx) + d_coef(j_idx) .* dx) .* dx) .* dx;
end

function phi = initPhi(N, wingL, deltaI, deltaJ, Rsigma, PSI)
    edgeZ = wingL + 2; edgeR = wingL;
    [j_grid, i_grid] = meshgrid(0:N-1, 0:N-1); 
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
    dlmwrite(fullfile(dirName, fileName), f(1:sampleSkip:end, 1:sampleSkip:end), 'delimiter', '\t', 'precision', '%e');
end

function saveVecToDrive(f, dirName, fileName)
    sampleSkip = ceil(double(length(f))/256.0);
    f_sub = f(1:sampleSkip:end);
    if iscolumn(f_sub), f_sub = f_sub'; end
    dlmwrite(fullfile(dirName, fileName), f_sub, 'delimiter', '\t', 'precision', '%e');
end

function saveNVecToDrive(f, dirName, fileName)
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