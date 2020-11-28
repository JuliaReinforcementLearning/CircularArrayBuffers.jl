module CircularArrayBuffers

export CircularArrayBuffer, CircularVectorBuffer, capacity, isfull

"""
    CircularArrayBuffer{T}(sz::Integer...) -> CircularArrayBuffer{T, N}

`CircularArrayBuffer` uses a `N`-dimension `Array` of size `sz` to serve as a buffer for
`N-1`-dimension `Array`s of the same size.
"""
mutable struct CircularArrayBuffer{T,N} <: AbstractArray{T,N}
    buffer::Array{T,N}
    first::Int
    nframes::Int
    step_size::Int
end

const CircularVectorBuffer{T} = CircularArrayBuffer{T, 1}

CircularVectorBuffer{T}(n::Integer) where T = CircularArrayBuffer{T}(n)

function CircularArrayBuffer{T}(d::Integer...) where {T}
    N = length(d)
    CircularArrayBuffer(Array{T}(undef, d...), 1, 0, N == 1 ? 1 : *(d[1:end-1]...))
end

function CircularArrayBuffer(A::AbstractArray{T,N}) where {T,N}
    CircularArrayBuffer(A, 1, size(A, N), N == 1 ? 1 : *(size(A)[1:end-1]...))
end

Base.IndexStyle(::CircularArrayBuffer) = IndexLinear()

Base.size(cb::CircularArrayBuffer{T,N}, i::Integer) where {T,N} = i == N ? cb.nframes : size(cb.buffer, i)
Base.size(cb::CircularArrayBuffer{T,N}) where {T,N} = ntuple(i -> size(cb, i), N)
Base.getindex(cb::CircularArrayBuffer{T,N}, i::Int) where {T,N} = getindex(cb.buffer, _buffer_index(cb, i))
Base.setindex!(cb::CircularArrayBuffer{T,N}, v, i::Int) where {T,N} = setindex!(cb.buffer, v, _buffer_index(cb, i))

capacity(cb::CircularArrayBuffer{T,N}) where {T,N} = size(cb.buffer, N)
isfull(cb::CircularArrayBuffer) = cb.nframes == capacity(cb)
Base.isempty(cb::CircularArrayBuffer) = cb.nframes == 0

@inline function _buffer_index(cb::CircularArrayBuffer, i::Int)
    ind = (cb.first - 1) * cb.step_size + i
    mod1(ind, length(cb.buffer))
end

@inline function _buffer_frame(cb::CircularArrayBuffer, i::Int)
    n = capacity(cb)
    idx = cb.first + i - 1
    mod1(idx, n)
end

_buffer_frame(cb::CircularArrayBuffer, I::Vector{Int}) = map(i -> _buffer_frame(cb, i), I)

function Base.empty!(cb::CircularArrayBuffer)
    cb.nframes = 0
    cb
end

function Base.push!(cb::CircularArrayBuffer{T, N}, data) where {T,N}
    if cb.nframes == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.nframes += 1
    end
    if N == 1
        cb[cb.nframes] = data
    else
        cb[ntuple(_ -> (:), N - 1)..., cb.nframes] .= data
    end
    cb
end

function Base.pop!(cb::CircularArrayBuffer{T, N}) where {T,N}
    if cb.nframes <= 0
        throw(ArgumentError("buffer must be non-empty"))
    else
        res = @views cb[ntuple(_ -> (:), N - 1)..., cb.nframes]
        cb.nframes -= 1
        res
    end
end

end
