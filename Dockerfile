
# can be set at build time with --build-arg BASE_IMAGE=your_image:tag
# for arm64 use: debian:12-slim 
# for x86_64 use: nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 (includes CUDNN for building dlib)
ARG BASE_IMAGE=nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04
FROM $BASE_IMAGE

# Set the working directory in the container
WORKDIR /app

SHELL ["/bin/bash", "-c"] 

# automatically filled by docker build
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-venv python3-pip build-essential \
     ffmpeg git cmake libwebpdemux2 && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY LICENSE /app/
	
# Install any needed packages specified in requirements.txt (if you had one)
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN python -m pip install -U pip  

RUN if [[ "$TARGETARCH" = "amd64" ]]; then \
        echo "Build for AMD64 with CUDA ..." && \
        pip install --pre torch torchvision  --index-url https://download.pytorch.org/whl/nightly/cu130 && \
        pip install "onnxruntime-gpu>=1.17"; \
    elif [[ "$TARGETARCH" = "arm64" ]]; then \
        echo "Build for ARM64 with NO CUDA ..." && \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
        pip install "onnxruntime>=1.17"; \
    else \
        echo "Unsupported architecture: $TARGETARCH" >&2 && exit 1; \
    fi

# Install basicsr with automated patching (PEP 667 + torchvision compat)
COPY scripts /app/scripts
RUN python scripts/install_basicsr.py

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY Dockerfile .dockerignore /app/
COPY pyproject.toml  README.md /app/

COPY *.py /app/
COPY tests /app/tests

# If the cuda 13 built opencv-python is available (cudev module) then install it here 
# This is specific to linux amd64. I had to build the whl myself. 
# Without this, opencv-python will run on CPU
#COPY --from=whl /opencv_contrib_python-4.13.0.90-cp312-cp312-linux_x86_64.whl /app/
#RUN pip install opencv_contrib_python-4.13.0.90-cp312-cp312-linux_x86_64.whl --force-reinstall

RUN pip install -e .

CMD []
 
