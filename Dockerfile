
# can be set at build time with --build-arg BASE_IMAGE=your_image:tag
# for arm64 use: debian:12-slim 
# for x86_64 use: nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 (includes CUDNN for building dlib)
ARG BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04
FROM $BASE_IMAGE

ARG CUDAVER=130

# Set the working directory in the container
WORKDIR /app

SHELL ["/bin/bash", "-c"] 

# automatically filled by docker build
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-venv python3-pip portaudio19-dev build-essential \
     ffmpeg git cmake libwebpdemux2 && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY pyproject.toml LICENSE  README.md /app/
	
# Install any needed packages specified in requirements.txt (if you had one)
RUN python3 -m venv venv
ENV PATH="venv/bin:$PATH"
RUN python -m pip install -U pip  

# on windows we need to install torchcodec separately (after torch) since torchcodec is not yet available on pip for windows, 
# on linux (both arm64 and amd64) we can install torchcodec together with torch (torchcodec will be installed without cuda support on arm64)

# the face_recognition package supports CUDA. But one needs to first build dlib with DLIB_USE_CUDA=1
# Note: that if we didn't need to build dlib for CUDA then we could use the runtime base image instead of devel
RUN if [[ "$TARGETARCH" = "amd64" ]]; then \
        echo "Build for AMD64 with CUDA ..." && \
        if [[ "$CUDAVER" = "130" ]]; then \
          pip install --pre torch torchvision  --index-url https://download.pytorch.org/whl/nightly/cu130; \
        else \
          pip install --pre torch torchvision  --index-url https://download.pytorch.org/whl/nightly/cu128; \
        fi; \
        DLIB_USE_CUDA=1 pip install -v dlib; \
    elif [[ "$TARGETARCH" = "arm64" ]]; then \
        echo "Build for ARM64 with NO CUDA ..." && \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; \ 
    else \
        echo "Unsupported architecture: $TARGETARCH" >&2 && exit 1; \
    fi

# Install basicsr with automated patching (PEP 667 + torchvision compat)
COPY scripts /app/scripts
RUN python scripts/install_basicsr.py

RUN pip install --group base --group youtube --group vl --group chararep

# we have to do this in case the CPU version was installed
# make sure to ignore the unintall in case onnxruntime is not installed
RUN if [[ "$TARGETARCH" = "amd64" ]]; then \
        echo "Build for AMD64 with CUDA ..." && \
        pip uninstall onnxruntime onnxruntime-gpu -y && \    
        pip install onnxruntime-gpu; \
    fi


# If the cuda 13 built opencv-python is available (cudev module) then install it here
# This is specific to linux amd64. I had to build the whl myself.
# Without this, opencv-python will run on CPU
COPY --from=whl /opencv_contrib_python-4.13.0.90-cp312-cp312-linux_x86_64.whl /app/
RUN pip install opencv_contrib_python-4.13.0.90-cp312-cp312-linux_x86_64.whl --force-reinstall

COPY Dockerfile .dockerignore main.py /app/
COPY tests /app/tests
COPY portrait_prep /app/portrait_prep
COPY vicrop /app/vicrop
COPY videsc /app/videsc
COPY chararep /app/chararep
COPY face_ops /app/face_ops


RUN pip install -e .

CMD []
 
