# CircularArrayBuffers

[![Build Status](https://github.com/JuliaReinforcementLearning/CircularArrayBuffers.jl/workflows/CI/badge.svg)](https://github.com/JuliaReinforcementLearning/CircularArrayBuffers.jl/actions)

`CircularArrayBuffers.jl` is a small package to wrap an `AbstractArray` as a buffer along the last dimension. The main benefit compared to [`CircularBuffer`](https://juliacollections.github.io/DataStructures.jl/latest/circ_buffer/) in [`DataStructures.jl`](https://github.com/JuliaCollections/DataStructures.jl) is that the view of consecutive elements is a `SubArray`.

## Usage

```julia
julia> using CircularArrayBuffers

julia> names(CircularArrayBuffers)
5-element Array{Symbol,1}:
 :CircularArrayBuffer
 :CircularArrayBuffers
 :CircularVectorBuffer
 :capacity
 :isfull

julia> a = CircularArrayBuffer(rand(2,3))
2×3 CircularArrayBuffer{Float64,2}:
 0.0510714  0.0260738  0.0245707
 0.856257   0.571643   0.0189365

julia> b = CircularArrayBuffer{Float64}(2,3)
2×0 CircularArrayBuffer{Float64,2}

julia> push!(b, rand(2))
2×1 CircularArrayBuffer{Float64,2}:
 0.4215856115651755
 0.5485806794787502

julia> push!(b, rand(2))
2×2 CircularArrayBuffer{Float64,2}:
 0.421586  0.640501
 0.548581  0.774729

julia> push!(b, rand(2))
2×3 CircularArrayBuffer{Float64,2}:
 0.421586  0.640501  0.653054
 0.548581  0.774729  0.902611

julia> push!(b, rand(2))
2×3 CircularArrayBuffer{Float64,2}:
 0.640501  0.653054  0.640373
 0.774729  0.902611  0.227435

julia> pop!(b)
2-element view(::CircularArrayBuffer{Float64,2}, :, 3) with eltype Float64:
 0.6403725468830439
 0.22743495787074597

julia> b
2×2 CircularArrayBuffer{Float64,2}:
 0.640501  0.653054
 0.774729  0.902611

julia> size(b)
(2, 2)

julia> capacity(b)
3

julia> isfull(b)
false

julia> push!(b, rand(2))
2×3 CircularArrayBuffer{Float64,2}:
 0.640501  0.653054  0.885887
 0.774729  0.902611  0.0332439

julia> isfull(b)
true

julia> eltype(b)
Float64
```
