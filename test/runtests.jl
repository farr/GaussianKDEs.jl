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
            @test isapprox(pdf(kn, 0), 1/sqrt(2*pi), rtol=0.5)
        end
        @testset "Uniform Density" begin 
            @test isapprox(pdf(ku, 0.5), 1, rtol=0.5)
            @test isapprox(pdf(ku, 0), 1, rtol=0.5) 
        end
        @testset "One Sided Positive" begin
            @test isapprox(pdf(ke, 0.5), exp(-0.5), rtol=0.5)
        end
    end

    @testset "multivariate KDE" begin
        nd = 3
        np = 1024

        xs = randn(nd, np)
        k = KDE(xs)

        @testset "Unit Normal Density" begin
            @test isapprox(pdf(k, zeros(nd)), 1/sqrt((2*pi))^nd, rtol=0.5)
        end
    end

    @testset "bandwidth optimization" begin
        nd = 3
        np = 1024
        nt = 128

        xs = randn(nd, np)
        xs_test = randn(nd, nt)
        k = bw_opt_kde(xs, xs_test)

        @testset "Unit Normal Density" begin
            @test isapprox(pdf(k, zeros(nd)), 1/sqrt((2*pi))^nd, rtol=0.5)
        end
    end
end