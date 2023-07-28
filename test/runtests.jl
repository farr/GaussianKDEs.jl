using Distributions
using GaussianKDEs
using Random
using Test

@testset "GaussianKDEs.jl Tests" begin
    @testset "1D KDE Tests" begin
        xs = randn(1024)
        ys = rand(1024)
        zs = randexp(1024)

        kn = KDE(xs)
        ku = BoundedKDE(ys, lower=0, upper=1)
        ke = BoundedKDE(zs, lower=0)

        @testset "Unit Normal Density" begin 
            @test isapprox(pdf(kn, 0), 1/sqrt(2*pi), atol=0.1)
        end
        @testset "Uniform Density" begin 
            @test isapprox(pdf(ku, 0.5), 1, atol=0.1)
            @test isapprox(pdf(ku, 0), 1, atol=0.1) 
        end
        @testset "One Sided Positive" begin
            @test isapprox(pdf(ke, 0.5), exp(-0.5), atol=0.1)
        end
    end
end