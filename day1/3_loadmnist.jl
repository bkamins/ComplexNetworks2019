using Pkg
Pkg.activate(".")

using Serialization

filenames = (train_image="train-images-idx3-ubyte",
             train_label="train-labels-idx1-ubyte",
             test_image="t10k-images-idx3-ubyte",
             test_label="t10k-labels-idx1-ubyte")

function read_labels(fname)
    f = open(fname)
    magic = bswap(read(f, UInt32))
    @assert magic == 2049
    nobs = bswap(read(f, UInt32))
    @assert filesize(f) == 8 + nobs
    labels = read(f)
    @assert length(labels) == nobs
    close(f)
    return Int.(labels)
end

function read_images(fname)
    f = open(fname)
    magic = bswap(read(f, UInt32))
    @assert magic == 2051
    nobs = bswap(read(f, UInt32))
    @assert bswap(read(f, UInt32)) == 28
    @assert bswap(read(f, UInt32)) == 28
    @assert filesize(f) == 16 + nobs*28*28
    images = Matrix{Float64}[]
    for i in 1:nobs
        data = read(f, 28*28)
        mx = permutedims(reshape(data, 28, 28)) / 255
        push!(images, mx)
    end
    close(f)
    return images
end

train_images = read_images(filenames.train_image)
train_labels = read_labels(filenames.train_label)
test_images = read_images(filenames.test_image)
test_labels = read_labels(filenames.test_label)

serialize("mnist.data", (train_images, train_labels,
                         test_images, test_labels))
