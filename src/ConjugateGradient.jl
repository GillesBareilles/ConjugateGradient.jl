module ConjugateGradient

using LinearAlgebra
using DocStringExtensions
using Printf


function solve_tCG(A::Tm, b; ν=1e-3, ϵ_residual = 1e-13, maxiter=1e5, printlev=0) where {Tm <: AbstractMatrix}
    A!(res, η) = (res .= A * η)
    return solve_tCG(A!, b; ν, ϵ_residual, maxiter, printlev)
end

"""
    $(SIGNATURES)

Conjugate gradient method to solve the linear problem:
    find d s.t. A * x = b

Note that we are looking for a direction d in the tangent space of M at x, and the kernel of
∇²f_M contains at least the orthogonal to this set. CG should naturally give a solution
living in the tangent space.
"""
function solve_tCG(A!::Function, b; ν=1e-3, ϵ_residual = 1e-13, maxiter=1e5, printlev=0)
    rⱼ, rⱼ_prev = similar(b), similar(b)
    vⱼ, vⱼ_prev = similar(b), similar(b)
    Adₖ, Avⱼ = similar(b), similar(b)
    dₖ = similar(b)
    dₖ .= 0
    Adₖ .= 0
    Avⱼ .= 0

    d_type = :Unsolved
    j = 0

    (printlev>0) && @printf "\n    j norm(rⱼ)            norm(vⱼ)             ⟨vⱼ, hessf[vⱼ]⟩   ν * norm(vⱼ)^2\n"
    while true
        # current residual, conjugated direction
        A!(Adₖ, dₖ)
        rⱼ .= Adₖ .- b

        vⱼ = - rⱼ
        if j ≥ 1
            βⱼ = norm(rⱼ)^2 / norm(rⱼ_prev)^2
            vⱼ += βⱼ * vⱼ_prev
        end

        A!(Avⱼ, vⱼ)
        (printlev>0) && @printf "%5i %.10e    %.10e    % .10e     %.10e\n" j norm(rⱼ) norm(vⱼ) dot(vⱼ, Avⱼ) ν * norm(vⱼ)^2
        if norm(rⱼ) < ϵ_residual || dot(vⱼ, Avⱼ) < ν * norm(vⱼ)^2 || j > maxiter
            ## Satisfying point obtained
            if j == 0
                dₖ .= b
            end

            if norm(rⱼ) < ϵ_residual
                d_type = :Solved
                @debug "Exiting: ||rⱼ|| < ϵ : $(norm(rⱼ)) < $ϵ_residual\n"
            elseif j > maxiter
                d_type = :MaxIter
                @debug "Exiting: j > maxiter : $j > $maxiter\n"
            else
                d_type = :QuasiNegCurvature
                a = dot(vⱼ, Avⱼ)
                b = ν * norm(vⱼ)^2
                @debug "Exiting: ⟨vⱼ, hessf_x(vⱼ)⟩ < ν * ||vⱼ||² : $a < $b\n"
            end

            break
        end

        tⱼ = - dot(rⱼ, vⱼ) / dot(vⱼ, Avⱼ)
        dₖ += tⱼ * vⱼ

        rⱼ_prev = deepcopy(rⱼ)
        vⱼ_prev = deepcopy(vⱼ)

        j += 1
    end

    return dₖ, (;
                iter = j,
                d_type
                )

end

export solve_tCG

end # module
