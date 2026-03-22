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
RUN pip install -U pip
RUN pip install --pre torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/nightly/cu130
RUN pip install --group gpu --group base --group youtube --group vl

COPY Dockerfile .dockerignore main.py /app/
COPY tests /app/tests
COPY portrait_prep /app/portrait_prep
COPY vicrop /app/vicrop
COPY videsc /app/videsc


RUN pip install -e .

CMD []
 
