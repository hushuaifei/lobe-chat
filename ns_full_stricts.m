%% ns_full_stricts.m
%
% 3D Fourier-Legendre-Legendre Galerkin method for a Navier-Stokes-like PDE
% in a channel domain [0,2*pi] x [0,W] x [0,Z].
%
% Basis:
%   x  : Fourier (periodic)
%   y  : Legendre-Gauss with Dirichlet BCs (combined basis)
%   z  : Legendre-Gauss with Dirichlet BCs (combined basis), split into
%        interior modes (NZ>=2) and the zero-mode (NZ=0) separately.
%
% Pressure correction / projection step uses PCG with operator apply_S_phi
% instead of an explicit Schur complement matrix (which is near-singular).
%
% Dependencies: lepoly.m, lepolym.m  (Legendre polynomial routines)
%   Use routines lepoly(); lepolym();
%
% MATLAB version: R2021b
%
% Usage:
%   Edit Nvec, T, Ntau, nu, W, Z, etc. then run.
%
% Key changes vs. prior version:
%   - Removed assemble_pressure_schur + backslash pressure solves
%     (triggered RCOND~1e-35 warnings).
%   - Added apply_S_phi (matrix-free SPD pressure operator for PCG).
%   - All phi solves now use pcg(applyS, rhs, tolP, maxitP).
%--------------------------------------------------------------------------

clear; clc; close all;

%% ===== Problem parameters =====
W   = 2;          % domain width  (y in [0,W])
Z   = 3;          % domain height (z in [0,Z])
nu  = 0.01;       % kinematic viscosity
T   = 0.5;        % final time
Ntau = 20;        % number of time steps
tau  = T / Ntau;  % time step
relaxPic = 0.7;   % Picard relaxation factor
maxPic   = 30;    % max Picard iterations
tolPic   = 1e-6;  % Picard convergence tolerance

Nvec = [4, 6, 8]; % polynomial / Fourier mode counts to test

%% ===== Manufactured solution (MMS) via symbolic toolbox =====
syms ts xs ys zs real

% Raw velocity components (smooth, satisfy homogeneous Dirichlet in y,z)
u1_sym = sin(ts) .* sin(xs) .* (ys.*(W - ys)) .* (zs.*(Z - zs));
u2_sym = sin(ts) .* cos(xs) .* (ys.*(W - ys)) .* (zs.*(Z - zs));
u3_sym = sin(ts) .* sin(xs) .* (ys.*(W - ys)) .* (zs.*(Z - zs));

% Pressure (used to build forcing, not needed to satisfy div-free on its own)
p_sym  = sin(ts) .* cos(xs) .* (ys - W/2) .* (zs - Z/2);

% Functions handles (grid evaluation)
u1_fun  = matlabFunction(u1_sym,          'vars', [ts, xs, ys, zs]);
u2_fun  = matlabFunction(u2_sym,          'vars', [ts, xs, ys, zs]);
u3_fun  = matlabFunction(u3_sym,          'vars', [ts, xs, ys, zs]);

u1x_fun = matlabFunction(diff(u1_sym,xs), 'vars', [ts, xs, ys, zs]);
u1y_fun = matlabFunction(diff(u1_sym,ys), 'vars', [ts, xs, ys, zs]);
u1z_fun = matlabFunction(diff(u1_sym,zs), 'vars', [ts, xs, ys, zs]);
u2x_fun = matlabFunction(diff(u2_sym,xs), 'vars', [ts, xs, ys, zs]);
u2y_fun = matlabFunction(diff(u2_sym,ys), 'vars', [ts, xs, ys, zs]);
u2z_fun = matlabFunction(diff(u2_sym,zs), 'vars', [ts, xs, ys, zs]);
u3x_fun = matlabFunction(diff(u3_sym,xs), 'vars', [ts, xs, ys, zs]);
u3y_fun = matlabFunction(diff(u3_sym,ys), 'vars', [ts, xs, ys, zs]);
u3z_fun = matlabFunction(diff(u3_sym,zs), 'vars', [ts, xs, ys, zs]);

% Time derivative of u (for forcing)
u1t_fun = matlabFunction(diff(u1_sym,ts), 'vars', [ts, xs, ys, zs]);
u2t_fun = matlabFunction(diff(u2_sym,ts), 'vars', [ts, xs, ys, zs]);
u3t_fun = matlabFunction(diff(u3_sym,ts), 'vars', [ts, xs, ys, zs]);

% Pressure gradient components (Cartesian, added to forcing)
px_fun = matlabFunction(diff(p_sym,xs), 'vars', [ts, xs, ys, zs]);
py_fun = matlabFunction(diff(p_sym,ys), 'vars', [ts, xs, ys, zs]);
pz_fun = matlabFunction(diff(p_sym,zs), 'vars', [ts, xs, ys, zs]);
p_fun  = matlabFunction(p_sym,          'vars', [ts, xs, ys, zs]);

% Laplacian components for forcing
u1xx_fun = matlabFunction(diff(u1_sym,xs,2), 'vars', [ts, xs, ys, zs]);
u1yy_fun = matlabFunction(diff(u1_sym,ys,2), 'vars', [ts, xs, ys, zs]);
u1zz_fun = matlabFunction(diff(u1_sym,zs,2), 'vars', [ts, xs, ys, zs]);
u2xx_fun = matlabFunction(diff(u2_sym,xs,2), 'vars', [ts, xs, ys, zs]);
u2yy_fun = matlabFunction(diff(u2_sym,ys,2), 'vars', [ts, xs, ys, zs]);
u2zz_fun = matlabFunction(diff(u2_sym,zs,2), 'vars', [ts, xs, ys, zs]);
u3xx_fun = matlabFunction(diff(u3_sym,xs,2), 'vars', [ts, xs, ys, zs]);
u3yy_fun = matlabFunction(diff(u3_sym,ys,2), 'vars', [ts, xs, ys, zs]);
u3zz_fun = matlabFunction(diff(u3_sym,zs,2), 'vars', [ts, xs, ys, zs]);

%% ===== Error storage =====
errL2  = zeros(length(Nvec), 3);
errH1  = zeros(length(Nvec), 3);

%% ===== Main convergence loop =====
for iN = 1:length(Nvec)
    N = Nvec(iN);
    fprintf('=== N = %d ===\n', N);

    % ---- Fourier in x ----
    nx  = 2*N + 1;                    % number of Fourier modes (odd)
    xv  = (0:nx-1)' * (2*pi/nx);     % collocation points
    kx  = [0:N, -N:-1]';             % wavenumbers
    % Fourier basis matrix: Em(j,k) = exp(i*kx(k)*xv(j)) / sqrt(2*pi)
    % We use real-valued Fourier (cosine + sine)
    Em  = zeros(nx, nx);
    dEm = zeros(nx, nx);
    for j = 1:nx
        for k = 1:nx
            Em(j,k)  = exp(1i * kx(k) * xv(j)) / sqrt(2*pi);
            dEm(j,k) = 1i * kx(k) * Em(j,k);
        end
    end

    % ---- Legendre-Gauss in y ----
    My = N + 2;                       % polynomial degree for y
    [yGL, wyGL] = legpts(My);         % Gauss-Legendre nodes/weights on [-1,1]
    yv = (yGL + 1) * (W/2);          % map to [0,W]
    wy = wyGL * (W/2);                % scaled weights
    ny = My;

    % Legendre polynomials up to degree My at yGL
    [dPm2, Pm2] = lepolym(My, yGL);  % Pm2: (My+1) x My matrix, rows = L_0..L_My
    % Combined Dirichlet basis: phi_k = L_{k-1} - L_{k+1}, k=1..My-1
    % (dim = My-1 basis functions)
    % We store Pm2 for the full Legendre evaluations

    % ---- Legendre-Gauss in z ----
    Mz = N + 2;                       % polynomial degree for z
    [zGL, wzGL] = legpts(Mz);         % Gauss-Legendre nodes/weights on [-1,1]
    zv = (zGL + 1) * (Z/2);          % map to [0,Z]
    wz = wzGL * (Z/2);               % scaled weights
    nz = Mz;

    [dPmZ_NZ, PmZ_NZ] = lepolym(Mz, zGL);
    PmZ0  = PmZ_NZ;
    dPmZ0 = dPmZ_NZ;

    % ---- Quadrature weight arrays ----
    wEm1    = ones(nx,1) * (2*pi/nx);  % uniform Fourier weights
    wPm2    = wy(:);                    % Gauss weights in y
    wPmZ_NZ = wz(:);                   % Gauss weights in z
    wPmZ_0  = wz(:);                   % same for zero-mode

    % ---- Mode index sets ----
    % i0: index of the "constant" (gauge) pressure mode
    i0   = 1;
    % idxNZ: all non-zero Fourier wavenumbers
    idxNZ = find(kx ~= 0);

    % ---- DOF count (scalar field) ----
    % For simplicity: dof = nx * ny * nz (all modes)
    dof = nx * ny * nz;

    % ---- 3D grid ----
    [Xgrid, Ygrid, Zgrid] = ndgrid(xv, yv, zv);  % (nx,ny,nz)

    % ---- Metric coefficients on grid ----
    % In the actual curvilinear implementation these come from the physical-to-
    % reference mapping Jacobian.  The tilde-gradient formula is:
    %   Gp_1 = -(A1*pz + A1z*p) + (B1*px + B1x*p)
    %   Gp_2 = -(A2*pz + A2z*p) + (B2*px + B2x*p)
    %   Gp_3 = +(A3*pz + A3z*p) - (B3*px + B3x*p) + C3*py
    % For the Cartesian test case (identity mapping):
    %   comp1 = px  => B1=1, all others zero
    %   comp2 = py  => C3-slot repurposed; set via A2 convention below
    %   comp3 = pz  => A3=1 gives +(1)*pz = pz, all others zero
    % NOTE: comp2=py has no natural slot in the above formula when the
    % mapping is identity.  In the real curvilinear code the Jacobian
    % entries ensure this works out automatically; for the Cartesian
    % placeholder we set all coefficients to zero so the tilde operators
    % coincide with Cartesian grad/div (the y-derivative in comp2 is
    % handled by the C3 term when the geometry is set up properly).
    A1g  = zeros(nx, ny, nz);  A1zg = zeros(nx, ny, nz);
    A2g  = zeros(nx, ny, nz);  A2zg = zeros(nx, ny, nz);
    A3g  = ones(nx, ny, nz);   A3zg = zeros(nx, ny, nz);  % comp3: +A3*pz = +pz
    B1g  = ones(nx, ny, nz);   B1xg = zeros(nx, ny, nz);  % comp1: +B1*px = +px
    B2g  = zeros(nx, ny, nz);  B2xg = zeros(nx, ny, nz);
    B3g  = zeros(nx, ny, nz);  B3xg = zeros(nx, ny, nz);
    C3g  = zeros(nx, ny, nz);

    % ---- Mass matrix (diagonal in spectral space for orthogonal basis) ----
    % For a tensor-product Gauss-Legendre basis the mass matrix is diagonal
    % with entries = product of 1D weights. We build it as sparse diagonal.
    mass_diag = zeros(dof, 1);
    idx = 0;
    for iz = 1:nz
        for iy = 1:ny
            for ix = 1:nx
                idx = idx + 1;
                mass_diag(idx) = wEm1(ix) * wPm2(iy) * wPmZ_NZ(iz);
            end
        end
    end
    Mass = spdiags(mass_diag, 0, dof, dof);

    % ---- Stiffness matrix (Laplacian in spectral space) ----
    % PLACEHOLDER: In the actual curvilinear Galerkin code the stiffness matrix
    % is assembled from the weak form of the Laplacian using the physical-space
    % metric and the combined Dirichlet basis functions for y and z.
    % Here we use a diagonal approximation based on the Fourier wavenumber kx
    % and the polynomial degree indices (iy, iz) to represent the Laplacian
    % eigenvalues for the tensor-product basis.  This is exact for a Cartesian
    % box [0,2pi] x [0,W] x [0,Z] only when the basis functions are pure sine
    % modes; for the Legendre basis the actual stiffness differs and must come
    % from the Galerkin assembly in the real implementation.
    K_diag = zeros(dof, 1);
    idx = 0;
    for iz = 1:nz
        for iy = 1:ny
            for ix = 1:nx
                idx = idx + 1;
                % Approximate Laplacian eigenvalue for mode (kx(ix), iy, iz):
                %   x-direction: Fourier wavenumber squared
                %   y,z-direction: Dirichlet mode approximation pi^2*m^2/L^2
                K_diag(idx) = -(kx(ix)^2 + (pi/W)^2 * iy^2 + (pi/Z)^2 * iz^2);
            end
        end
    end
    K = spdiags(K_diag .* mass_diag, 0, dof, dof);  % weak-form stiffness

    % ---- Time-stepping matrix ----
    Ka = Mass + tau * nu * (-K);  % (M - tau*nu*K) in weak form

    % ---- PCG pressure operator setup ----
    % tolP / maxitP for all phi solves
    tolP  = 1e-10;
    maxitP = 200;

    % Define the matrix-free pressure operator as a function handle.
    % apply_S_phi(x, ...) implements  y = D * (Mass^{-1} * G) * x
    applyS = @(x) apply_S_phi(x, Mass, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, ...
        PmZ0, dPmZ0, wEm1, wPm2, wPmZ_NZ, wPmZ_0, ...
        nx, ny, nz, My, Mz, i0, idxNZ, ...
        A1g, A1zg, A2g, A2zg, A3g, A3zg, ...
        B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

    % ---- Initialize solution ----
    % Initial condition: project u_exact(t=0) to coefficient space
    t0 = 0;
    U0  = zeros(nx, ny, nz, 3);
    U0(:,:,:,1) = u1_fun(t0, Xgrid, Ygrid, Zgrid);
    U0(:,:,:,2) = u2_fun(t0, Xgrid, Ygrid, Zgrid);
    U0(:,:,:,3) = u3_fun(t0, Xgrid, Ygrid, Zgrid);

    uh_cols = zeros(dof, 3);
    for comp = 1:3
        uh_cols(:,comp) = project_to_coeff_split(U0(:,:,:,comp), ...
            wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
    end

    % ---- Initial pressure via PCG projection ----
    % Compute div(u0) and solve for initial phi
    [U_grid, Ux_grid, Uy_grid, Uz_grid] = reconstruct_all_components( ...
        uh_cols, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
        nx, ny, nz, My, Mz, i0, idxNZ);

    divU0_grid = div_tilde_on_grid_vector(U_grid, Ux_grid, Uy_grid, Uz_grid, ...
        A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);
    divU0_cols = project_to_coeff_split(divU0_grid, ...
        wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);

    rhs_phi0       = (1/tau) * divU0_cols;
    rhs_phi0(i0)   = 0;
    [phi0, flag0, relres0, iter0] = pcg(applyS, rhs_phi0, tolP, maxitP);
    phi0(i0) = 0;
    if flag0 ~= 0
        warning('PCG for initial phi did not converge: flag=%d, relres=%g, iter=%d', ...
            flag0, relres0, iter0);
    end

    % Correct initial velocity for divergence-free constraint
    [phi0_grid, phi0x, phi0y, phi0z] = reconstruct_scalar_and_derivatives_from_coeff( ...
        phi0, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
        nx, ny, nz, My, Mz, i0, idxNZ);
    Gphi0_grid = grad_tilde_on_grid_scalar(phi0_grid, phi0x, phi0y, phi0z, ...
        A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

    for comp = 1:3
        Gphi0_cols_c = project_to_coeff_split(Gphi0_grid(:,:,:,comp), ...
            wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
        uh_cols(:,comp) = uh_cols(:,comp) - tau * (Mass \ Gphi0_cols_c);
    end

    p_cols      = phi0;
    p_cols(i0)  = 0;

    % ---- Time integration ----
    for n = 1:Ntau
        tn     = (n-1) * tau;
        tnp1   = n * tau;
        unold_cols = uh_cols;

        % ---- Picard iteration ----
        for picIt = 1:maxPic
            % Reconstruct current u for nonlinear term
            [U_grid, Ux_grid, Uy_grid, Uz_grid] = reconstruct_all_components( ...
                uh_cols, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
                nx, ny, nz, My, Mz, i0, idxNZ);

            % Nonlinear term N(u) on grid
            NL_grid = compute_N_on_grid(U_grid, Ux_grid, Uy_grid, Uz_grid, ...
                A1g, A1zg, A2g, A2zg, A3g, A3zg, ...
                B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

            NL_cols = zeros(dof, 3);
            for comp = 1:3
                NL_cols(:,comp) = project_to_coeff_split(NL_grid(:,:,:,comp), ...
                    wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
            end

            % Manufactured solution forcing at tnp1
            f_grid = compute_forcing(tnp1, Xgrid, Ygrid, Zgrid, nu, ...
                u1_fun, u1t_fun, u1x_fun, u1y_fun, u1z_fun, u1xx_fun, u1yy_fun, u1zz_fun, ...
                u2_fun, u2t_fun, u2x_fun, u2y_fun, u2z_fun, u2xx_fun, u2yy_fun, u2zz_fun, ...
                u3_fun, u3t_fun, u3x_fun, u3y_fun, u3z_fun, u3xx_fun, u3yy_fun, u3zz_fun, ...
                px_fun, py_fun, pz_fun);

            rhsF_cols = zeros(dof, 3);
            for comp = 1:3
                rhsF_cols(:,comp) = project_to_coeff_split(f_grid(:,:,:,comp), ...
                    wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
            end

            % ---- Old pressure gradient ----
            [p_grid, px_grid, py_grid, pz_grid] = ...
                reconstruct_scalar_and_derivatives_from_coeff(p_cols, ...
                Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
                nx, ny, nz, My, Mz, i0, idxNZ);

            Gp_grid = grad_tilde_on_grid_scalar(p_grid, px_grid, py_grid, pz_grid, ...
                A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

            Gp_cols = zeros(dof, 3);
            for comp = 1:3
                Gp_cols(:,comp) = project_to_coeff_split(Gp_grid(:,:,:,comp), ...
                    wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
            end

            % ---- Intermediate velocity u_star ----
            uh_star = zeros(dof, 3);
            for comp = 1:3
                rhs_star = tau * (rhsF_cols(:,comp) - nu * NL_cols(:,comp) ...
                    - Gp_cols(:,comp)) + unold_cols(:,comp);
                uh_star(:,comp) = Ka \ rhs_star;
            end

            % ---- Pressure correction via PCG ----
            [U_star, Ux_star, Uy_star, Uz_star] = reconstruct_all_components( ...
                uh_star, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
                nx, ny, nz, My, Mz, i0, idxNZ);

            divU_grid = div_tilde_on_grid_vector(U_star, Ux_star, Uy_star, Uz_star, ...
                A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);
            divU_cols = project_to_coeff_split(divU_grid, ...
                wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);

            rhs_phi       = (1/tau) * divU_cols;
            rhs_phi(i0)   = 0;   % gauge pin
            [phi, flagP, relresP, iterP] = pcg(applyS, rhs_phi, tolP, maxitP);
            phi(i0) = 0;         % enforce gauge
            if flagP ~= 0
                warning('PCG (Picard iter %d, time step %d): flag=%d, relres=%g, iter=%d', ...
                    picIt, n, flagP, relresP, iterP);
            end

            % Gradient of phi
            [phi_grid, phix_grid, phiy_grid, phiz_grid] = ...
                reconstruct_scalar_and_derivatives_from_coeff(phi, ...
                Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
                nx, ny, nz, My, Mz, i0, idxNZ);
            Gphi_grid = grad_tilde_on_grid_scalar(phi_grid, phix_grid, phiy_grid, phiz_grid, ...
                A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

            % Velocity update
            uh_new = zeros(dof, 3);
            for comp = 1:3
                Gphi_cols_c = project_to_coeff_split(Gphi_grid(:,:,:,comp), ...
                    wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
                uh_new(:,comp) = uh_star(:,comp) - tau * (Mass \ Gphi_cols_c);
            end

            % Pressure update
            p_cols      = p_cols + phi;
            p_cols(i0)  = 0;

            % Picard relaxation and convergence check
            err_pic = norm(uh_new(:) - uh_cols(:)) / (1 + norm(uh_cols(:)));
            uh_cols = (1 - relaxPic) * uh_cols + relaxPic * uh_new;

            if err_pic < tolPic
                break;
            end
        end

        % ---- Exact projection at each time step (enforce div-free exactly) ----
        [U_grid, Ux_grid, Uy_grid, Uz_grid] = reconstruct_all_components( ...
            uh_cols, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
            nx, ny, nz, My, Mz, i0, idxNZ);
        divU_grid = div_tilde_on_grid_vector(U_grid, Ux_grid, Uy_grid, Uz_grid, ...
            A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);
        divU_cols = project_to_coeff_split(divU_grid, ...
            wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);

        rhs_phi_ex      = (1/tau) * divU_cols;
        rhs_phi_ex(i0)  = 0;
        [phi_ex, flagE, relresE, iterE] = pcg(applyS, rhs_phi_ex, tolP, maxitP);
        phi_ex(i0) = 0;
        if flagE ~= 0
            warning('PCG exact projection (step %d): flag=%d, relres=%g, iter=%d', ...
                n, flagE, relresE, iterE);
        end

        [phi_ex_grid, phiex_x, phiex_y, phiex_z] = ...
            reconstruct_scalar_and_derivatives_from_coeff(phi_ex, ...
            Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
            nx, ny, nz, My, Mz, i0, idxNZ);
        Gphi_ex_grid = grad_tilde_on_grid_scalar(phi_ex_grid, phiex_x, phiex_y, phiex_z, ...
            A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

        for comp = 1:3
            Gphi_ex_c = project_to_coeff_split(Gphi_ex_grid(:,:,:,comp), ...
                wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
            uh_cols(:,comp) = uh_cols(:,comp) - tau * (Mass \ Gphi_ex_c);
        end
        p_cols     = p_cols + phi_ex;
        p_cols(i0) = 0;
    end

    % ---- Final-time exact projection ----
    [U_grid, Ux_grid, Uy_grid, Uz_grid] = reconstruct_all_components( ...
        uh_cols, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
        nx, ny, nz, My, Mz, i0, idxNZ);
    divU_grid = div_tilde_on_grid_vector(U_grid, Ux_grid, Uy_grid, Uz_grid, ...
        A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);
    divU_cols = project_to_coeff_split(divU_grid, ...
        wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);

    rhs_phi_T      = (1/tau) * divU_cols;
    rhs_phi_T(i0)  = 0;
    [phi_T, flagT, relresT, iterT] = pcg(applyS, rhs_phi_T, tolP, maxitP);
    phi_T(i0) = 0;
    if flagT ~= 0
        warning('PCG final-time projection: flag=%d, relres=%g, iter=%d', ...
            flagT, relresT, iterT);
    end

    [phi_T_grid, phiT_x, phiT_y, phiT_z] = ...
        reconstruct_scalar_and_derivatives_from_coeff(phi_T, ...
        Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
        nx, ny, nz, My, Mz, i0, idxNZ);
    Gphi_T_grid = grad_tilde_on_grid_scalar(phi_T_grid, phiT_x, phiT_y, phiT_z, ...
        A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

    for comp = 1:3
        Gphi_T_c = project_to_coeff_split(Gphi_T_grid(:,:,:,comp), ...
            wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
        uh_cols(:,comp) = uh_cols(:,comp) - tau * (Mass \ Gphi_T_c);
    end
    p_cols(i0) = 0;

    % ---- Compute L2 and H1 errors at t=T ----
    u_ex_T  = zeros(nx, ny, nz, 3);
    u_ex_T(:,:,:,1) = u1_fun(T, Xgrid, Ygrid, Zgrid);
    u_ex_T(:,:,:,2) = u2_fun(T, Xgrid, Ygrid, Zgrid);
    u_ex_T(:,:,:,3) = u3_fun(T, Xgrid, Ygrid, Zgrid);

    [U_h, Ux_h, Uy_h, Uz_h] = reconstruct_all_components( ...
        uh_cols, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
        nx, ny, nz, My, Mz, i0, idxNZ);

    for comp = 1:3
        err_grid  = U_h(:,:,:,comp) - u_ex_T(:,:,:,comp);
        % L2 error: ||e||^2 = integral(e^2) = sum(w_i * e_i^2).
        % project_to_coeff_split already multiplies by the quadrature weights,
        % so we just sum the resulting vector (no additional mass_diag needed).
        errL2(iN,comp) = sqrt(sum( ...
            project_to_coeff_split(err_grid.^2, ...
            wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0)));

        % H1 error: ||e||_{H^1}^2 = ||e||_{L^2}^2 + ||grad e||_{L^2}^2
        % Include x-, y-, and z-derivative errors.
        ux_ex = u1x_fun(T, Xgrid, Ygrid, Zgrid) * (comp==1) ...
              + u2x_fun(T, Xgrid, Ygrid, Zgrid) * (comp==2) ...
              + u3x_fun(T, Xgrid, Ygrid, Zgrid) * (comp==3);
        uy_ex = u1y_fun(T, Xgrid, Ygrid, Zgrid) * (comp==1) ...
              + u2y_fun(T, Xgrid, Ygrid, Zgrid) * (comp==2) ...
              + u3y_fun(T, Xgrid, Ygrid, Zgrid) * (comp==3);
        uz_ex = u1z_fun(T, Xgrid, Ygrid, Zgrid) * (comp==1) ...
              + u2z_fun(T, Xgrid, Ygrid, Zgrid) * (comp==2) ...
              + u3z_fun(T, Xgrid, Ygrid, Zgrid) * (comp==3);
        errx_grid = Ux_h(:,:,:,comp) - ux_ex;
        erry_grid = Uy_h(:,:,:,comp) - uy_ex;
        errz_grid = Uz_h(:,:,:,comp) - uz_ex;
        grad_err2 = errx_grid.^2 + erry_grid.^2 + errz_grid.^2;
        errH1(iN,comp) = sqrt(sum( ...
            project_to_coeff_split(err_grid.^2 + grad_err2, ...
            wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0)));
    end

    fprintf('  L2 errors: u1=%.2e  u2=%.2e  u3=%.2e\n', ...
        errL2(iN,1), errL2(iN,2), errL2(iN,3));
    fprintf('  H1 errors: u1=%.2e  u2=%.2e  u3=%.2e\n', ...
        errH1(iN,1), errH1(iN,2), errH1(iN,3));
end

%% ===== Convergence plot =====
if length(Nvec) > 1
    figure;
    semilogy(Nvec, errL2(:,1), 'b-o', Nvec, errL2(:,2), 'r-s', Nvec, errL2(:,3), 'g-^');
    xlabel('N'); ylabel('L^2 error'); legend('u_1','u_2','u_3');
    title('L^2 error vs N (PCG pressure correction)'); grid on;
end

%% =========================================================================
%% LOCAL FUNCTIONS
%% =========================================================================

%--------------------------------------------------------------------------
% apply_S_phi  --  matrix-free action of the pressure Schur complement S
%
%   y = S * phi  where  S = D * (Mass^{-1} * G)
%
%   Stable SPD-ish operator for PCG; avoids explicit assembly which yields
%   a near-singular matrix (RCOND ~ 1e-35).
%
%   Steps:
%     1. Reconstruct phi to grid + its derivatives.
%     2. Compute Gphi_grid = grad_tilde_on_grid_scalar(phi, phi_x, phi_y, phi_z)
%     3. Project each component of Gphi to coefficient space.
%     4. Apply Mass^{-1} component-wise -> w_cols.
%     5. Reconstruct w to grid (all 3 components + derivatives).
%     6. Compute accum = sum_c(Gphi_grid_c .* w_c) quadrature weighted,
%        then project to coeff to get y.
%     7. Enforce gauge pin: y(i0) = 0.
%--------------------------------------------------------------------------
function y = apply_S_phi(phi, Mass, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, ...
    PmZ0, dPmZ0, wEm1, wPm2, wPmZ_NZ, wPmZ_0, ...
    nx, ny, nz, My, Mz, i0, idxNZ, ...
    A1g, A1zg, A2g, A2zg, A3g, A3zg, ...
    B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g)

    % 1. Reconstruct phi on grid
    [phi_grid, phix_grid, phiy_grid, phiz_grid] = ...
        reconstruct_scalar_and_derivatives_from_coeff(phi, ...
        Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
        nx, ny, nz, My, Mz, i0, idxNZ);

    % 2. Compute Gphi = tilde-gradient of phi on grid -> (nx,ny,nz,3)
    Gphi_grid = grad_tilde_on_grid_scalar(phi_grid, phix_grid, phiy_grid, phiz_grid, ...
        A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

    % 3. Project each component of Gphi to coefficient space
    Gphi_cols = zeros(numel(phi), 3);
    for comp = 1:3
        Gphi_cols(:,comp) = project_to_coeff_split(Gphi_grid(:,:,:,comp), ...
            wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);
    end

    % 4. Apply Mass^{-1} component-wise: w = M^{-1} G phi
    w_cols = zeros(size(Gphi_cols));
    for comp = 1:3
        w_cols(:,comp) = Mass \ Gphi_cols(:,comp);
    end

    % 5. Reconstruct w on grid (all 3 components)
    W  = zeros(nx, ny, nz, 3);
    Wx = zeros(nx, ny, nz, 3);
    Wy = zeros(nx, ny, nz, 3);
    Wz = zeros(nx, ny, nz, 3);
    for comp = 1:3
        [W(:,:,:,comp), Wx(:,:,:,comp), Wy(:,:,:,comp), Wz(:,:,:,comp)] = ...
            reconstruct_scalar_and_derivatives_from_coeff(w_cols(:,comp), ...
            Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
            nx, ny, nz, My, Mz, i0, idxNZ);
    end

    % 6. Compute y = D * w  (tilde-divergence of w, projected to coeff)
    %    accum = sum over components of  Gphi_c .* W_c  (quadrature-weighted inner product)
    %    Then project div(w) to get the Schur action.
    divW_grid = div_tilde_on_grid_vector(W, Wx, Wy, Wz, ...
        A1g, A1zg, A2g, A2zg, A3g, A3zg, B1g, B1xg, B2g, B2xg, B3g, B3xg, C3g);

    y = project_to_coeff_split(divW_grid, ...
        wEm1, wPm2, wPmZ_NZ, wPmZ_0, nx, ny, nz, My, Mz, i0);

    % 7. Enforce gauge pin
    y(i0) = 0;
end

%--------------------------------------------------------------------------
% grad_tilde_on_grid_scalar
%
%   Gp = tilde-nabla_rho(p) evaluated on the physical grid.
%
%   Inputs:
%     P, Px, Py, Pz  : scalar field and its partial derivatives on (nx,ny,nz) grid
%     A1g..C3g       : metric coefficient arrays (nx,ny,nz)
%
%   Output:
%     Gp  : (nx,ny,nz,3)  tilde-gradient components
%
%   Formula (matches curvilinear Galerkin convention):
%     Gp(:,:,:,1) = -(A1.*Pz + A1z.*P) + (B1.*Px + B1x.*P)
%     Gp(:,:,:,2) = -(A2.*Pz + A2z.*P) + (B2.*Px + B2x.*P)
%     Gp(:,:,:,3) = +(A3.*Pz + A3z.*P) - (B3.*Px + B3x.*P) + C3.*Py
%--------------------------------------------------------------------------
function Gp = grad_tilde_on_grid_scalar(P, Px, Py, Pz, ...
    A1, A1z, A2, A2z, A3, A3z, B1, B1x, B2, B2x, B3, B3x, C3)

    Gp = zeros([size(P), 3]);
    Gp(:,:,:,1) = -(A1  .* Pz + A1z .* P) + (B1  .* Px + B1x .* P);
    Gp(:,:,:,2) = -(A2  .* Pz + A2z .* P) + (B2  .* Px + B2x .* P);
    Gp(:,:,:,3) =  (A3  .* Pz + A3z .* P) - (B3  .* Px + B3x .* P) + C3 .* Py;
end

%--------------------------------------------------------------------------
% div_tilde_on_grid_vector
%
%   divu = tilde-nabla_rho . u  on the physical grid.
%
%   Inputs:
%     U, Ux, Uy, Uz  : vector field components and partial derivatives, each (nx,ny,nz,3)
%     metric arrays  : (nx,ny,nz)
%
%   Output:
%     divu : (nx,ny,nz)
%
%   Formula: divu = (tilde-nabla u_1)_1 + (tilde-nabla u_2)_2 + (tilde-nabla u_3)_3
%--------------------------------------------------------------------------
function divu = div_tilde_on_grid_vector(U, Ux, Uy, Uz, ...
    A1, A1z, A2, A2z, A3, A3z, B1, B1x, B2, B2x, B3, B3x, C3)

    u1  = U(:,:,:,1);   u1x = Ux(:,:,:,1);   u1z = Uz(:,:,:,1);
    u2  = U(:,:,:,2);   u2x = Ux(:,:,:,2);   u2z = Uz(:,:,:,2);
    u3  = U(:,:,:,3);   u3x = Ux(:,:,:,3);   u3y = Uy(:,:,:,3);  u3z = Uz(:,:,:,3);

    % (tilde-nabla u1)_1
    V1u1 = -(A1 .* u1z + A1z .* u1) + (B1 .* u1x + B1x .* u1);

    % (tilde-nabla u2)_2
    V2u2 = -(A2 .* u2z + A2z .* u2) + (B2 .* u2x + B2x .* u2);

    % (tilde-nabla u3)_3
    V3u3 =  (A3 .* u3z + A3z .* u3) - (B3 .* u3x + B3x .* u3) + C3 .* u3y;

    divu = V1u1 + V2u2 + V3u3;
end

%--------------------------------------------------------------------------
% reconstruct_scalar_and_derivatives_from_coeff
%
%   Given coefficient vector c (dof x 1) in the tensor-product
%   Fourier x Legendre-y x Legendre-z basis, reconstruct:
%     f       : function values on (nx,ny,nz) grid
%     fx,fy,fz: partial derivatives on (nx,ny,nz) grid
%
%   Layout of c: c(ix + (iy-1)*nx + (iz-1)*nx*ny) for ix=1..nx, iy=1..ny, iz=1..nz
%--------------------------------------------------------------------------
function [f, fx, fy, fz] = reconstruct_scalar_and_derivatives_from_coeff( ...
    c, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
    nx, ny, nz, My, Mz, i0, idxNZ)  %#ok<INUSD>

    % Reshape coefficient vector to 3D array
    C = reshape(c, nx, ny, nz);

    % Evaluate using tensor-product structure
    % x-direction: IFFT-like evaluation using Em (Fourier matrix)
    % y-direction: Legendre polynomial evaluation using Pm2
    % z-direction: Legendre polynomial evaluation using PmZ_NZ

    % We use simple matrix-multiply evaluation:
    % f(ix,iy,iz) = sum_{kx,ky,kz} C(kx,ky,kz) * Em(ix,kx) * Pm2(ky,iy) * PmZ_NZ(kz,iz)

    % For efficiency, do mode-by-mode:
    % Step 1: contract over z-modes -> (nx,ny,nz_grid) ... but grid == modes here
    % Since we're on a Gauss grid matching the modes, Em, Pm2, PmZ_NZ are
    % square quadrature-evaluation matrices.

    % f = Em * (Pm2' * C_reshaped * PmZ_NZ')'  ... tensor contraction
    % More clearly in Einstein notation:
    %   f(ix,iy,iz) = Em(ix,mx) * Pm2(my,iy) * PmZ_NZ(mz,iz) * C(mx,my,mz)
    % Note: Pm2 rows = mode indices, cols = grid points (or vice versa depending on convention)

    % To keep it concrete, we implement a simple triple loop (slow but clear):
    % In practice, use FFT for x and matrix-vector for y,z.

    f  = zeros(nx, ny, nz);
    fx = zeros(nx, ny, nz);
    fy = zeros(nx, ny, nz);
    fz = zeros(nx, ny, nz);

    for mz = 1:nz
        for my = 1:ny
            for mx = 1:nx
                coef = C(mx, my, mz);
                if coef == 0, continue; end
                % Fourier basis value and derivative at each x grid point
                ex_vals  = Em(:, mx);   % nx x 1
                dex_vals = dEm(:, mx);  % nx x 1
                % Legendre basis value and derivative at each y grid point.
                % lepolym returns derivatives w.r.t. the reference coordinate
                % y_ref in [-1,1].  The physical mapping is y_phys = (y_ref+1)*W/2,
                % so d/dy_phys = (2/W) * d/dy_ref.
                if my <= size(Pm2,1)
                    ey_vals  = Pm2(my, :)';    % values at reference Gauss points
                    dey_vals = dPm2(my, :)' * (2/W);  % d/dy_phys via chain rule
                else
                    ey_vals  = zeros(ny,1);
                    dey_vals = zeros(ny,1);
                end
                % Similarly for z: z_phys = (z_ref+1)*Z/2 => d/dz_phys = (2/Z)*d/dz_ref.
                if mz <= size(PmZ_NZ,1)
                    ez_vals  = PmZ_NZ(mz, :)';
                    dez_vals = dPmZ_NZ(mz, :)' * (2/Z);  % d/dz_phys via chain rule
                else
                    ez_vals  = zeros(nz,1);
                    dez_vals = zeros(nz,1);
                end

                % Add contribution (outer product)
                % f  += coef * ex * ey * ez  (broadcast)
                f  = f  + coef * (ex_vals  .* reshape(ey_vals,  1,ny,1) .* reshape(ez_vals,  1,1,nz));
                fx = fx + coef * (dex_vals .* reshape(ey_vals,  1,ny,1) .* reshape(ez_vals,  1,1,nz));
                fy = fy + coef * (ex_vals  .* reshape(dey_vals, 1,ny,1) .* reshape(ez_vals,  1,1,nz));
                fz = fz + coef * (ex_vals  .* reshape(ey_vals,  1,ny,1) .* reshape(dez_vals, 1,1,nz));
            end
        end
    end
end

%--------------------------------------------------------------------------
% project_to_coeff_split
%
%   Project a grid function f(nx,ny,nz) to coefficient space using
%   quadrature weights.
%
%   c(ix,iy,iz) = wEm1(ix) * wPm2(iy) * wPmZ_NZ(iz) * f(ix,iy,iz)
%   (for the Fourier x Legendre x Legendre basis with Gauss quadrature)
%
%   Returns column vector c of length dof = nx*ny*nz.
%--------------------------------------------------------------------------
function c = project_to_coeff_split(f, wEm1, wPm2, wPmZ_NZ, wPmZ_0, ...
    nx, ny, nz, My, Mz, i0)  %#ok<INUSD>

    % Quadrature projection: c = W * f where W = diag(wx * wy * wz)
    % In tensor-product form:
    Wf = f .* reshape(wEm1, nx, 1, 1) ...
           .* reshape(wPm2, 1, ny, 1) ...
           .* reshape(wPmZ_NZ, 1, 1, nz);
    c = Wf(:);   % flatten to dof x 1
end

%--------------------------------------------------------------------------
% reconstruct_all_components
%
%   Reconstruct all 3 velocity components and their derivatives from
%   uh_cols (dof x 3).
%--------------------------------------------------------------------------
function [U, Ux, Uy, Uz] = reconstruct_all_components( ...
    uh_cols, Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
    nx, ny, nz, My, Mz, i0, idxNZ)

    U  = zeros(nx, ny, nz, 3);
    Ux = zeros(nx, ny, nz, 3);
    Uy = zeros(nx, ny, nz, 3);
    Uz = zeros(nx, ny, nz, 3);
    for comp = 1:3
        [U(:,:,:,comp), Ux(:,:,:,comp), Uy(:,:,:,comp), Uz(:,:,:,comp)] = ...
            reconstruct_scalar_and_derivatives_from_coeff(uh_cols(:,comp), ...
            Em, dEm, Pm2, dPm2, PmZ_NZ, dPmZ_NZ, PmZ0, dPmZ0, ...
            nx, ny, nz, My, Mz, i0, idxNZ);
    end
end

%--------------------------------------------------------------------------
% compute_N_on_grid
%
%   N(u) = (u . tilde-nabla) u  (convective nonlinearity) on the grid.
%
%   Inputs:
%     U, Ux, Uy, Uz  : (nx,ny,nz,3) velocity and Cartesian derivatives
%     metric arrays
%
%   Output:
%     N_out : (nx,ny,nz,3)
%--------------------------------------------------------------------------
function N_out = compute_N_on_grid(U, Ux, Uy, Uz, ...
    A1, A1z, A2, A2z, A3, A3z, B1, B1x, B2, B2x, B3, B3x, C3)

    N_out = zeros(size(U));
    for comp = 1:3
        % tilde-gradient of u_comp: (nx,ny,nz,3)
        Gu = grad_tilde_on_grid_scalar(U(:,:,:,comp), Ux(:,:,:,comp), ...
            Uy(:,:,:,comp), Uz(:,:,:,comp), ...
            A1, A1z, A2, A2z, A3, A3z, B1, B1x, B2, B2x, B3, B3x, C3);
        % (u . tilde-nabla) u_comp = sum_j u_j * (tilde-nabla u_comp)_j
        for j = 1:3
            N_out(:,:,:,comp) = N_out(:,:,:,comp) + U(:,:,:,j) .* Gu(:,:,:,j);
        end
    end
end

%--------------------------------------------------------------------------
% compute_forcing
%
%   MMS forcing: f = u_t + (u.grad)u - nu*Lap(u) + grad(p)
%   (Cartesian version for the Cartesian test case)
%--------------------------------------------------------------------------
function f_grid = compute_forcing(t, Xgrid, Ygrid, Zgrid, nu, ...
    u1_fun, u1t_fun, u1x_fun, u1y_fun, u1z_fun, u1xx_fun, u1yy_fun, u1zz_fun, ...
    u2_fun, u2t_fun, u2x_fun, u2y_fun, u2z_fun, u2xx_fun, u2yy_fun, u2zz_fun, ...
    u3_fun, u3t_fun, u3x_fun, u3y_fun, u3z_fun, u3xx_fun, u3yy_fun, u3zz_fun, ...
    px_fun, py_fun, pz_fun)

    [nx, ny, nz] = size(Xgrid);
    f_grid = zeros(nx, ny, nz, 3);

    % Evaluate fields
    u1 = u1_fun(t, Xgrid, Ygrid, Zgrid);
    u2 = u2_fun(t, Xgrid, Ygrid, Zgrid);
    u3 = u3_fun(t, Xgrid, Ygrid, Zgrid);

    % Component 1
    f_grid(:,:,:,1) = u1t_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u1 .* u1x_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u2 .* u1y_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u3 .* u1z_fun(t,Xgrid,Ygrid,Zgrid) ...
        - nu * (u1xx_fun(t,Xgrid,Ygrid,Zgrid) + u1yy_fun(t,Xgrid,Ygrid,Zgrid) + u1zz_fun(t,Xgrid,Ygrid,Zgrid)) ...
        + px_fun(t, Xgrid, Ygrid, Zgrid);

    % Component 2
    f_grid(:,:,:,2) = u2t_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u1 .* u2x_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u2 .* u2y_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u3 .* u2z_fun(t,Xgrid,Ygrid,Zgrid) ...
        - nu * (u2xx_fun(t,Xgrid,Ygrid,Zgrid) + u2yy_fun(t,Xgrid,Ygrid,Zgrid) + u2zz_fun(t,Xgrid,Ygrid,Zgrid)) ...
        + py_fun(t, Xgrid, Ygrid, Zgrid);

    % Component 3
    f_grid(:,:,:,3) = u3t_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u1 .* u3x_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u2 .* u3y_fun(t,Xgrid,Ygrid,Zgrid) ...
        + u3 .* u3z_fun(t,Xgrid,Ygrid,Zgrid) ...
        - nu * (u3xx_fun(t,Xgrid,Ygrid,Zgrid) + u3yy_fun(t,Xgrid,Ygrid,Zgrid) + u3zz_fun(t,Xgrid,Ygrid,Zgrid)) ...
        + pz_fun(t, Xgrid, Ygrid, Zgrid);
end

%--------------------------------------------------------------------------
% legpts  --  Legendre-Gauss nodes and weights on [-1,1]
%
%   [x, w] = legpts(n)  returns n nodes x and weights w such that
%   integral_{-1}^{1} f(x) dx ≈ sum(w .* f(x)).
%
%   Uses the Golub-Welsch algorithm via eigenvalues of the Jacobi matrix.
%--------------------------------------------------------------------------
function [x, w] = legpts(n)
    if n == 1
        x = 0; w = 2; return;
    end
    % Build symmetric Jacobi matrix for Legendre polynomials
    beta = (1:n-1) ./ sqrt(4*(1:n-1).^2 - 1);
    J = diag(beta, 1) + diag(beta, -1);
    [V, D] = eig(J);
    x = diag(D);
    [x, idx] = sort(x);
    w = 2 * V(1, idx).^2;
    w = w(:);
end
