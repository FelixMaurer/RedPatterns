using Printf
using Dates
using DelimitedFiles

# ==============================================================================
#                               CONSTANTS & STRUCTS
# ==============================================================================

const RC = 1100.0
const RL = 30.0
const P0 = 1100.0
const KERNEL_N = 31

struct SimParams
    N::Int32
    M::Int32
    wingL::Int32
    kernelN::Int32
    subDiv::Float64
    zShift::Float64
    SYS_L::Float64
    IZ::Float64
    IR::Float64
    IT::Float64
    PSI::Float64
    alpha::Float64
    beta::Float64
    D::Float64
end

# ==============================================================================
#                            SERIAL "KERNEL" FUNCTIONS
# ==============================================================================

# Equivalent to k_inte!
function cpu_inte!(phi, psi, N)
    for i in 1:N
        sum_val = 0.0
        for k in 1:N
            @inbounds sum_val += phi[i, k]
        end
        @inbounds psi[i] = sum_val
    end
    return nothing
end

# Equivalent to k_cmp_a!
function cpu_cmp_a!(y, alp, N)
    for i in 2:(N-1)
        @inbounds alp[i] = 3.0 * (y[i+1] - y[i]) - 3.0 * (y[i] - y[i-1])
    end
    return nothing
end

# Equivalent to k_spline_coeffs!
function cpu_spline_coeffs!(y, alp, b, c, d, scratch, N)
    off_mu = 0
    off_ze = N
    off_locC = 2 * N
    off_locB = 3 * N
    off_locD = 4 * N

    @inbounds scratch[off_mu + 1] = 0.0
    @inbounds scratch[off_ze + 1] = 0.0

    # Forward sweep
    for i in 2:(N-1)
        @inbounds mu_prev = scratch[off_mu + (i-1)]
        @inbounds ze_prev = scratch[off_ze + (i-1)]
        @inbounds alp_curr = alp[i]
        
        val_mu = 1.0 / (4.0 - mu_prev)
        val_ze = (alp_curr - ze_prev) * val_mu
        
        @inbounds scratch[off_mu + i] = val_mu
        @inbounds scratch[off_ze + i] = val_ze
    end

    @inbounds scratch[off_locC + N] = 0.0

    # Backward sweep
    for j in (N-1):-1:1
        @inbounds ze_j = scratch[off_ze + j]
        @inbounds mu_j = scratch[off_mu + j]
        @inbounds c_next = scratch[off_locC + j + 1]
        
        c_curr = ze_j - mu_j * c_next
        @inbounds scratch[off_locC + j] = c_curr

        @inbounds y_next = y[j+1]
        @inbounds y_curr = y[j]
        
        b_curr = (y_next - y_curr) - (c_next + 2.0 * c_curr) / 3.0
        @inbounds scratch[off_locB + j] = b_curr
        
        d_curr = (c_next - c_curr) / 3.0
        @inbounds scratch[off_locD + j] = d_curr
    end

    # Copy out
    for idx in 1:N
        @inbounds b[idx] = scratch[off_locB + idx]
        @inbounds c[idx] = scratch[off_locC + idx]
        @inbounds d[idx] = scratch[off_locD + idx]
    end
    return nothing
end

# Equivalent to k_spline_eval!
function cpu_spline_eval!(y, b, c, d, psiIntp, M, subDiv)
    for k in 1:M
        x = (k - 1) / subDiv
        j_idx = floor(Int32, x)
        dx = x - Float64(j_idx)
        idx = j_idx + 1
        @inbounds psiIntp[k] = y[idx] + (b[idx] + (c[idx] + d[idx] * dx) * dx) * dx
    end
    return nothing
end

# Equivalent to k_conv!
function cpu_conv!(psi, I, convKernel, M, kernelN, subDiv, IZ)
    halo = (kernelN - 1) ÷ 2
    
    for gid in 1:M
        acc = 0.0
        for k in 1:kernelN
            neighbor_idx = gid - halo + (k - 1)
            val = 0.0
            if neighbor_idx >= 1 && neighbor_idx <= M
                @inbounds val = psi[neighbor_idx]
            end
            @inbounds acc += val * convKernel[k]
        end
        @inbounds I[gid] = acc * (IZ / subDiv)
    end
    return nothing
end

# Equivalent to k_dsmp!
function cpu_dsmp!(IIntp, I, N, subDiv)
    for i in 1:N
        j_idx = floor(Int32, (i - 1) * subDiv)
        @inbounds I[i] = IIntp[j_idx + 1]
    end
    return nothing
end

# Equivalent to k_grad!
function cpu_grad!(percoll, t, N, IZ, zShift, gradL)
    PL = 8.0
    for i in 1:N
        x = IZ * Float64(i - 1)
        val = (x - zShift - gradL/2.0) / (gradL/2.0) * PL/2.0
        @inbounds percoll[i] = val
    end
    return nothing
end

# Equivalent to k_wing!
function cpu_wing!(percoll, gradWing, t, N, wingL)
    for i in 1:N
        @inbounds r3 = percoll[wingL + 1] - percoll[wingL]
        @inbounds r2 = percoll[wingL + 1]
        r1 = r2 - 10.0
        x1 = 20.0 
        x2 = Float64(wingL)
        @inbounds p_x1 = percoll[Int(x1) + 1]
        if p_x1 < r1; r1 = p_x1; end
        
        a = (r1 - r2 + r3 * (x2 - x1)) / ((x1 - x2)^2)
        b = r3 - 2.0 * a * x2
        c = r2 - r3 * x2 + x2^2 * a
        
        val = 0.0
        idx_0 = i - 1
        
        if idx_0 <= wingL
            val = a * idx_0^2 + b * idx_0 + c
        elseif idx_0 >= (N - 1 - wingL)
            term = (N - 1 - idx_0)
            val = -(a * term^2 + b * term + c)
        end
        
        if idx_0 >= (N - 1 - 13)
            clamp_idx = N - 1 - 13
            term_c = (N - 1 - clamp_idx)
            val = -(a * term_c^2 + b * term_c + c)
        end
        if idx_0 <= 13
            clamp_idx = 13
            val = a * clamp_idx^2 + b * clamp_idx + c
        end
        @inbounds gradWing[i] = val
    end
    return nothing
end

# Equivalent to k_compute_flux!
function cpu_compute_flux!(phi, J, K, percoll, R, I, gradWing, p::SimParams)
    for j in 1:p.N
        for i in 1:p.N
            # --- J Flux Calculation (Z direction) ---
            if i <= p.N - 1
                function get_v(idx_z)
                    idx_0 = idx_z - 1
                    rp = 0.0
                    if idx_0 > p.wingL && idx_0 < (p.N - 1 - p.wingL)
                        @inbounds rp = R[j] + percoll[idx_z] - P0
                    else
                        @inbounds rp = R[j] + gradWing[idx_z] - P0
                    end
                    @inbounds return (-p.alpha * rp - p.beta * I[idx_z])
                end
                
                v_face = 0.5 * (get_v(i) + get_v(i+1))
                @inbounds phi_up = (v_face > 0.0) ? phi[i, j] : phi[i+1, j]
                @inbounds diff_flux_z = -p.D * (phi[i+1, j] - phi[i, j]) / p.IZ
                @inbounds J[i, j] = v_face * phi_up + diff_flux_z
            elseif i == p.N
                @inbounds J[i, j] = 0.0
            end
            
            # --- K Flux Calculation (R direction) ---
            if j <= p.N - 1
                @inbounds diff_flux_r = -p.D * (phi[i, j+1] - phi[i, j]) / p.IR
                @inbounds K[i, j] = diff_flux_r
            elseif j == p.N
                @inbounds K[i, j] = 0.0
            end
        end
    end
    return nothing
end

# Equivalent to k_update_phi!
function cpu_update_phi!(phi, J, K, p::SimParams)
    for j in 1:p.N
        for i in 1:p.N
            @inbounds flux_in_z = (i == 1) ? 0.0 : J[i-1, j]
            @inbounds flux_out_z = J[i, j]
            div_z = (flux_out_z - flux_in_z) / p.IZ
            
            @inbounds flux_in_r = (j == 1) ? 0.0 : K[i, j-1]
            @inbounds flux_out_r = K[i, j]
            div_r = (flux_out_r - flux_in_r) / p.IR
            
            @inbounds val = phi[i, j] - p.IT * (div_z + div_r)
            if val < 0.0; val = 0.0; end
            @inbounds phi[i, j] = val
        end
    end
    return nothing
end

# ==============================================================================
#                               HOST FUNCTIONS
# ==============================================================================

function init_phi!(phi, R, N, wingL, IZ, IR, PSI)
    edgeZ = wingL + 2
    edgeR = wingL
    scaleSigmaZ = 0.6; scaleSigmaR = 0.7
    posZ1, posR1 = 0.25, 0.45
    posZ2, posR2 = 0.75, 0.55
    totalLengthZ = (N - 1) * IZ
    totalLengthR = (N - 1) * IR
    zCenter1 = totalLengthZ * posZ1; rCenter1 = totalLengthR * posR1
    zCenter2 = totalLengthZ * posZ2; rCenter2 = totalLengthR * posR2
    Rsigma = 2.0
    baseSigmaIdx = Rsigma / IR
    sIdxZ = baseSigmaIdx * scaleSigmaZ; sIdxR = baseSigmaIdx * scaleSigmaR
    denZ = 2.0 * sIdxZ^2; denR = 2.0 * sIdxR^2
    
    fill!(phi, 0.0)
    
    for j in 0:N-1
        for i in 0:N-1
            zVal = i * IZ; rVal = j * IR
            dz1 = (zVal - zCenter1) / IZ; dr1 = (rVal - rCenter1) / IR
            g1 = exp(-(dz1^2 / denZ + dr1^2 / denR))
            dz2 = (zVal - zCenter2) / IZ; dr2 = (rVal - rCenter2) / IR
            g2 = exp(-(dz2^2 / denZ + dr2^2 / denR))
            
            if !(i < edgeZ || i > (N - 1 - edgeZ) || j < edgeR || j > (N - 1 - edgeR))
                @inbounds phi[i+1, j+1] = g1 + g2
            end
        end
    end
    
    phiSum = sum(phi)
    if phiSum > 0
        normFactor = PSI * (N - 2 * edgeZ)
        phi .*= (normFactor / phiSum)
    end
end

function save_n_vec(f_arr, dir, fname)
    open(joinpath(dir, fname), "w") do io
        join(io, f_arr, "\t"); write(io, "\n")
    end
end

function save_arr(f_arr, dir, fname, N)
    sampleSkip = ceil(Int, N / 256.0)
    open(joinpath(dir, fname), "w") do io
        for i in 1:sampleSkip:N
            first = true
            for j in 1:sampleSkip:N
                if !first; write(io, "\t"); end
                print(io, f_arr[i, j]); first = false
            end
            write(io, "\n")
        end
    end
end

function print_progress(step, total, elapsed)
    barWidth = 40
    progress = step / total
    print("\r[")
    pos = floor(Int, barWidth * progress)
    for i in 0:(barWidth-1)
        if i < pos; print("="); elseif i == pos; print(">"); else; print(" "); end
    end
    perc = floor(Int, progress * 100)
    print("] $perc % ")
    if step > 0
        eta = (elapsed / step) * (total - step)
        h = floor(Int, eta / 3600)
        m = floor(Int, (eta - h*3600) / 60)
        s = floor(Int, eta % 60)
        @printf("| eta: %02d:%02d:%02d", h, m, s)
    end
    flush(stdout)
end

# ==============================================================================
#                               MAIN
# ==============================================================================

function main()
    # 1. Parse Arguments
    if length(ARGS) < 1
        println(stderr, "Error: N (grid size) must be provided as the first argument.")
        exit(1)
    end
    N = parse(Int32, ARGS[1])
    argIdx = 2
    ISF = 1.0; PSI_val = 0.02; IT = 0.005; T_val = 1200.0; NO = 1000.0; D_coeff = 0.0
    if length(ARGS) >= argIdx; ISF = parse(Float64, ARGS[argIdx]); argIdx+=1; end
    if length(ARGS) >= argIdx; PSI_val = parse(Float64, ARGS[argIdx]); argIdx+=1; end
    if length(ARGS) >= argIdx; IT = parse(Float64, ARGS[argIdx]); argIdx+=1; end
    if length(ARGS) >= argIdx; T_val = parse(Float64, ARGS[argIdx]); argIdx+=1; end
    if length(ARGS) >= argIdx; NO = parse(Float64, ARGS[argIdx]); argIdx+=1; end
    if length(ARGS) >= argIdx; D_coeff = parse(Float64, ARGS[argIdx]); argIdx+=1; end
    NT = ceil(Int, T_val / IT)

    # 2. Setup Grid & Memory
    subDiv = 65536.0 / Float64(N)
    M = Int32(N * subDiv + 1)
    wingL = Int32(N * (34.0 / 256.0))
    SYS_L = M * 1.041412353515625e-6
    zShift = (SYS_L - 0.06) / 2.0
    IZ = SYS_L / (N - 1)
    IR = RL / (N - 1)
    
    params = SimParams(N, M, wingL, KERNEL_N, subDiv, zShift, SYS_L, IZ, IR, IT, PSI_val, 12e-5, 7.4e23, D_coeff)

    # Output Directory
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    outDir = "sim_serial_$timestamp/"
    mkpath(outDir)
    
    println("\n============================================================")
    println("   julia SERIAL simulation: conservative finite volume transport ")
    println("============================================================")
    println("  grid size (n)   : $N")
    println("  subgrid (m)     : $M")
    println("  time steps (nt) : $NT")
    println("  mode            : SERIAL")
    println("  output dir      : $outDir")
    println("============================================================\n")

    # Allocations (Standard CPU Arrays)
    d_R = zeros(Float64, N)
    for j in 0:N-1
        d_R[j+1] = RC - RL/2 + RL * (Float64(j)/Float64(N-1))
    end
    
    d_phi = zeros(Float64, N, N)
    init_phi!(d_phi, d_R, N, wingL, IZ, IR, PSI_val)
    
    d_J = zeros(Float64, N, N)
    d_K = zeros(Float64, N, N)
    d_I = zeros(Float64, N)
    d_psi = zeros(Float64, N)
    d_percoll = zeros(Float64, N)
    d_gradWing = zeros(Float64, N)
    d_alp = zeros(Float64, N)
    d_b = zeros(Float64, N); d_c = zeros(Float64, N); d_d = zeros(Float64, N)
    d_splineScratch = zeros(Float64, 5 * N)
    d_psiIntp = zeros(Float64, M)
    d_IIntp = zeros(Float64, M)
    
    # Kernel Loading
    kernelInFile = "kernelInput.dat"
    h_intKernel = zeros(Float64, KERNEL_N)
    if isfile(kernelInFile)
        raw_data = readdlm(kernelInFile, Float64)
        vec_data = vec(raw_data)
        for i in 1:min(length(vec_data), KERNEL_N); h_intKernel[i] = vec_data[i] * ISF; end
        print("  -> loading kernel... done.\n")
    else
        println(stderr, "fatal error: could not open kernel file '$kernelInFile'")
        exit(1)
    end
    save_n_vec(h_intKernel, outDir, "intKernel.dat")
    d_intKernel = h_intKernel

    # 3. Warmup (JIT compilation trigger)
    print("  -> warming up (JIT)... ")
    cpu_inte!(d_phi, d_psi, N)
    cpu_cmp_a!(d_psi, d_alp, N)
    cpu_spline_coeffs!(d_psi, d_alp, d_b, d_c, d_d, d_splineScratch, N)
    cpu_spline_eval!(d_psi, d_b, d_c, d_d, d_psiIntp, M, subDiv)
    cpu_conv!(d_psiIntp, d_IIntp, d_intKernel, M, KERNEL_N, subDiv, IZ)
    cpu_dsmp!(d_IIntp, d_I, N, subDiv)
    cpu_grad!(d_percoll, 0.0, N, IZ, zShift, 0.06)
    cpu_wing!(d_percoll, d_gradWing, 0.0, N, wingL)
    cpu_compute_flux!(d_phi, d_J, d_K, d_percoll, d_R, d_I, d_gradWing, params)
    cpu_update_phi!(d_phi, d_J, d_K, params)
    println("done.")

    # 4. Simulation Loop with Step Measurement
    println("  -> starting simulation loop...\n")
    
    SKIP_START = 50
    SKIP_END = 50
    t = 0.0
    n_out = Int(NO)
    
    step_times = Float64[]
    sizehint!(step_times, NT)
    
    sim_start_time = time()

    for step in 1:NT
        is_output_step = (step % n_out == 0) || (step == 1) || (step == NT)
        measure_step = (step > SKIP_START) && (step < (NT - SKIP_END)) && (!is_output_step)

        t0 = time()
        
        # --- Physics Steps ---
        cpu_inte!(d_phi, d_psi, N)
        cpu_cmp_a!(d_psi, d_alp, N)
        cpu_spline_coeffs!(d_psi, d_alp, d_b, d_c, d_d, d_splineScratch, N)
        cpu_spline_eval!(d_psi, d_b, d_c, d_d, d_psiIntp, M, subDiv)
        cpu_conv!(d_psiIntp, d_IIntp, d_intKernel, M, KERNEL_N, subDiv, IZ)
        cpu_dsmp!(d_IIntp, d_I, N, subDiv)
        cpu_grad!(d_percoll, t, N, IZ, zShift, 0.06)
        cpu_wing!(d_percoll, d_gradWing, t, N, wingL)
        cpu_compute_flux!(d_phi, d_J, d_K, d_percoll, d_R, d_I, d_gradWing, params)
        cpu_update_phi!(d_phi, d_J, d_K, params)
        # ---------------------
        
        t1 = time()
        if measure_step
            push!(step_times, (t1 - t0) * 1000.0)
        end

        t += IT
        
        if is_output_step
            elapsed_total = time() - sim_start_time
            print_progress(step, NT, elapsed_total)
            
            s_idx = @sprintf("%010d", step)
            save_arr(d_phi, outDir, "phi_$s_idx.dat", N)
            save_n_vec(d_psi, outDir, "psi_$s_idx.dat")
            save_n_vec(d_gradWing, outDir, "gw_$s_idx.dat")
            save_n_vec(d_percoll, outDir, "gp_$s_idx.dat")
        end
    end
    
    total_duration = time() - sim_start_time
    print_progress(NT, NT, total_duration)
    println("\n\n  -> simulation complete.")

    # 5. Statistics Calculation
    if !isempty(step_times)
        n_meas = length(step_times)
        mean_val = sum(step_times) / n_meas
        
        sq_sum = 0.0
        for val in step_times
            sq_sum += (val - mean_val)^2
        end
        stdev = sqrt(sq_sum / n_meas)
        
        @printf("  -> avg step time: %.4f ms (+/- %.4f ms)\n", mean_val, stdev)
        println("  -> statistics based on $n_meas pure computation steps (SERIAL CPU Wall Time)")
    else
        println("  -> warning: not enough steps for statistics.")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end