
# default ARCH is amd64, can be overridden at build time with --build-arg ARCH=arm64
ARG ARCH=amd64

# can be set at build time with --build-arg BASE_IMAGE=your_image:tag
# for arm64 use: debian:12-slim 
# for x86_64 use: nvidia/cuda:13.1.1-devel-ubuntu24.04 or nvidia/cuda:13.1.1-runtime-ubuntu24.04
ARG BASE_IMAGE=nvidia/cuda:13.1.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-venv python3-pip portaudio19-dev build-essential \
     ffmpeg git cmake && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY pyproject.toml LICENSE  README.md /app/
	
# Install any needed packages specified in requirements.txt (if you had one)
RUN python3 -m venv venv
ENV PATH="venv/bin:$PATH"
RUN python -m pip install -U pip  

# on windows we need to install torchcodec separately (after torch) since torchcodec is not yet available on pip for windows, 
# on linux (both arm64 and amd64) we can install torchcodec together with torch (torchcodec will be installed without cuda support on arm64)
RUN if [ "${ARCH}" = "amd64" ]; then \
        echo "Build for AMD64 with CUDA ..."; \
        pip install --pre torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/nightly/cu130; \
        pip install --group gpu; \
    else \
        echo "Build for ARM64 with NO GPU ..."; \
        pip install torch torchvision torchaudio torchcodec; \
    fi



RUN pip install --group base --group youtube --group vl

COPY Dockerfile .dockerignore main.py /app/
COPY tests /app/tests
COPY portrait_prep /app/portrait_prep
COPY vicrop /app/vicrop
COPY videsc /app/videsc


RUN pip install -e .

CMD []
 
