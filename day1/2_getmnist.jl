using Pkg
Pkg.activate(".")

filenames = (train_image="train-images-idx3-ubyte",
             train_label="train-labels-idx1-ubyte",
             test_image="t10k-images-idx3-ubyte",
             test_label="t10k-labels-idx1-ubyte")

url = "http://yann.lecun.com/exdb/mnist/"

using CodecZlib

for fname in filenames
    fnamegz = fname * ".gz"
    isfile(fname) && continue
    @info "Processing $fname"
    if !isfile("$fname.gz")
        @info "Downloading $fnamegz"
        download(url*fnamegz, fnamegz)
    end
    open(fname, "w") do io_out
        open(fnamegz) do io_in
            data = read(GzipDecompressorStream(io_in))
            write(io_out, data)
        end
    end
end
@info "Done!"
