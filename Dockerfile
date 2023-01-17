FROM rust:1.66-bullseye as builder

RUN apt update
RUN apt install -y opencl-c-headers opencl-clhpp-headers libopencl-clang-dev ocl-icd-libopencl1 libopencl-clang11 intel-opencl-icd
RUN ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 /usr/lib/libOpenCL.so

WORKDIR /usr/src/opencl-bf-search
COPY bins bins
COPY crates crates
COPY Cargo.toml .
COPY Cargo.lock .

RUN cargo install --path /usr/src/opencl-bf-search/bins/opencl_bf_search

FROM debian:bullseye-slim
# FROM ubuntu:jammy

COPY --from=builder /usr/local/cargo/bin/opencl-bf-search /usr/local/bin/opencl-bf-search

RUN useradd -ms /bin/bash opencl-bf-search
USER opencl-bf-search

WORKDIR /app
ENTRYPOINT ["opencl-bf-search"]
