module CircularArrayBuffers

using Adapt

export CircularArrayBuffer, CircularVectorBuffer, capacity, isfull

"""
    CircularArrayBuffer{T}(sz::Integer...) -> CircularArrayBuffer{T, N, Array{T, N}}

`CircularArrayBuffer` uses a `N`-dimension `Array` of size `sz` to serve as a buffer for
`N-1`-dimension `Array`s of the same size.
"""
mutable struct CircularArrayBuffer{T,N,S<:AbstractArray{T,N}} <: AbstractArray{T,N}
    buffer::S
    first::Int
    nframes::Int
    step_size::Int
end

const CircularVectorBuffer{T,S} = CircularArrayBuffer{T,1,S}

CircularVectorBuffer{T}(n::Integer) where {T} = CircularArrayBuffer{T}(n)

function CircularArrayBuffer{T}(d::Integer...) where {T}
    N = length(d)
    CircularArrayBuffer(Array{T}(undef, d...), 1, 0, N == 1 ? 1 : *(d[1:end-1]...))
end

function CircularArrayBuffer(A::AbstractArray{T,N}) where {T,N}
    CircularArrayBuffer(A, 1, size(A, N), N == 1 ? 1 : *(size(A)[1:end-1]...))
end

Adapt.adapt_structure(to, cb::CircularArrayBuffer) =
    CircularArrayBuffer(adapt(to, cb.buffer), cb.first, cb.nframes, cb.step_size)

function Base.show(io::IO, ::MIME"text/plain", cb::CircularArrayBuffer{T}) where {T}
    print(io, ndims(cb) == 1 ? "CircularVectorBuffer(" : "CircularArrayBuffer(")
    Base.showarg(io, cb.buffer, false)
    print(io, ") with eltype $T:\n")
    Base.print_array(io, adapt(Array, cb))
    return nothing
end

Base.IndexStyle(::CircularArrayBuffer) = IndexLinear()

Base.size(cb::CircularArrayBuffer{T,N}, i::Integer) where {T,N} = i == N ? cb.nframes : size(cb.buffer, i)
Base.size(cb::CircularArrayBuffer{T,N}) where {T,N} = ntuple(i -> size(cb, i), N)
Base.getindex(cb::CircularArrayBuffer{T,N}, i::Int) where {T,N} = getindex(cb.buffer, _buffer_index(cb, i))
Base.getindex(cb::CircularArrayBuffer{T,N}, I...) where {T,N} = getindex(cb.buffer, Base.front(I)..., _buffer_frame(cb, Base.last(I)))

# !!!
# strange, but we need this function to show `CircularVectorBuffer` correctly
# `Base.print_array` will try to use `isassigned(cb, i, j)` to print elements
# And, `X::AbstractVector[2, 1]` is valid !!!
# without this line
# ```julia
# julia> cb = CircularArrayBuffer([1., 2.])
# CircularVectorBuffer(::Vector{Float64}) with eltype Float64:
#  1.0
#  2.0

# julia> push!(cb, 3)
# CircularVectorBuffer(::Vector{Float64}) with eltype Float64:
#  #undef
#  #undef
# ```
Base.getindex(cb::CircularVectorBuffer, i, j) = getindex(cb.buffer, _buffer_frame(cb, i), j)

Base.setindex!(cb::CircularArrayBuffer{T,N}, v, i::Int) where {T,N} = setindex!(cb.buffer, v, _buffer_index(cb, i))
Base.setindex!(cb::CircularArrayBuffer{T,N}, v, I...) where {T,N} = setindex!(cb.buffer, v, Base.front(I)..., _buffer_frame(cb, Base.last(I)))

Base.view(cb::CircularArrayBuffer, i::Int) = (@boundscheck checkbounds(cb, i); view(cb.buffer, _buffer_index(cb, i)))
Base.view(cb::CircularArrayBuffer, I...) = (@boundscheck checkbounds(cb, I...); view(cb.buffer, Base.front(I)..., _buffer_frame(cb, Base.last(I))))

capacity(cb::CircularArrayBuffer{T,N}) where {T,N} = size(cb.buffer, N)
isfull(cb::CircularArrayBuffer) = cb.nframes == capacity(cb)
Base.isempty(cb::CircularArrayBuffer) = cb.nframes == 0

"""
    _buffer_index(cb::CircularArrayBuffer, i::Int)

    Return the index of the `i`-th element in the buffer.
"""
@inline function _buffer_index(cb::CircularArrayBuffer, i::Int)
    idx = (cb.first - 1) * cb.step_size + i
    return wrap_index(idx, length(cb.buffer))
end
@inline _buffer_index(cb::CircularArrayBuffer, I::AbstractVector{<:Integer}) = map(Base.Fix1(_buffer_index, cb), I)

"""
    wrap_index(idx, n)

    Return the index of the `idx`-th element in the buffer, if index is one past the size, return 1, else error.
"""
function wrap_index(idx, n)
    if idx <= n
        return idx
    elseif idx <= 2n
        return idx - n
    else
        @info "oops! idx $(idx) > 2n $(2n)"
        return idx - n
    end
end

"""
    _buffer_frame(cb::CircularArrayBuffer, i::Int)

    Return the index of the `i`-th frame in the buffer.
"""
@inline function _buffer_frame(cb::CircularArrayBuffer, i::Int)
    n = capacity(cb)
    idx = cb.first + i - 1
    return wrap_index(idx, n)
end

_buffer_frame(cb::CircularArrayBuffer, I::CartesianIndex) = CartesianIndex(map(i->_buffer_frame(cb, i), Tuple(I)))
_buffer_frame(cb::CircularArrayBuffer, I::AbstractArray{<:Integer}) = map(i -> _buffer_frame(cb, i), I)
_buffer_frame(cb::CircularArrayBuffer, ::Colon) = _buffer_frame(cb, axes(cb)[end])

function Base.empty!(cb::CircularArrayBuffer)
    cb.nframes = 0
    cb
end

function Base.push!(cb::CircularArrayBuffer{T,N}, data) where {T,N}
    if cb.nframes == capacity(cb)
        cb.first = (cb.first == capacity(cb) ? 1 : cb.first + 1)
    else
        cb.nframes += 1
    end
    if N == 1
        i = _buffer_frame(cb, cb.nframes)
        cb.buffer[i:i] .= Ref(data)
    else
        cb.buffer[ntuple(_ -> (:), N - 1)..., _buffer_frame(cb, cb.nframes)] .= data
    end
    cb
end

function Base.append!(cb::CircularArrayBuffer{T,N}, data) where {T,N}
    d, r = divrem(length(data), cb.step_size)
    @assert r == 0
    if length(data) >= length(cb.buffer)
        cb.nframes = capacity(cb)
        cb.first = 1
        cb.buffer[:] .= @view data[end-length(cb.buffer)+1:end]
    else
        start_idx = (cb.first - 1) * cb.step_size + length(cb) + 1
        end_idx = start_idx + length(data) - 1
        if start_idx > length(cb.buffer)
            start_idx -= length(cb.buffer)
            end_idx -= length(cb.buffer)
        end
        if end_idx > length(cb.buffer)
            n_first_part = length(cb.buffer) - start_idx + 1
            n_second_part = length(data) - n_first_part
            cb.buffer[end-n_first_part+1:end] .= @view data[1:n_first_part]
            cb.buffer[1:n_second_part] .= @view data[end-n_second_part+1:end]
        else
            cb.buffer[start_idx:end_idx] .= data
        end

        if cb.nframes + d > capacity(cb)
            cb.first += cb.nframes + d - capacity(cb)
            if cb.first > capacity(cb)
                cb.first -= capacity(cb)
            end
            cb.nframes = capacity(cb)
        else
            cb.nframes += d
        end
    end
    cb
end

function Base.pop!(cb::CircularArrayBuffer{T,N}) where {T,N}
    if cb.nframes <= 0
        throw(ArgumentError("buffer must be non-empty"))
    else
        res = @views cb.buffer[ntuple(_ -> (:), N - 1)..., _buffer_frame(cb, cb.nframes)]
        cb.nframes -= 1
        res
    end
end

function Base.popfirst!(cb::CircularArrayBuffer{T,N}) where {T,N}
    if cb.nframes <= 0
        throw(ArgumentError("buffer must be non-empty"))
    else
        res = @views cb.buffer[ntuple(_ -> (:), N - 1)..., _buffer_frame(cb, 1)]
        cb.nframes -= 1
        cb.first += 1
        if cb.first > capacity(cb)
            cb.first = 1
        end
        res
    end
end

end
