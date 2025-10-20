# syntax=docker.io/docker/dockerfile:1.7-labs
ARG APP_PATH=/usr/src/app


FROM continuumio/miniconda3:25.3.1-1 AS builder
ARG APP_PATH
ENV APP_PATH=${APP_PATH}

ENV VENV_PATH=${APP_PATH}/venv LOCAL_BIN_PATH=${APP_PATH}/bin
RUN mkdir -p ${LOCAL_BIN_PATH}
WORKDIR ${APP_PATH}
RUN adduser --disabled-password --gecos '' rxn --uid 1000
RUN chown -R rxn:rxn ${APP_PATH}
RUN chgrp -R 0 ${APP_PATH} && chmod -R g=u  ${APP_PATH}

# build executables
RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
# Note that the legacy source code and scripts are no longer copied here (COPY --chown=rxn:0 . .)
COPY --chown=rxn:0 --exclude=docker/ . . 
# Create conda env
RUN conda create -y -n rxn-env -c conda-forge python=3.8 pip
# Install PyPI dependencies
RUN conda run -n rxn-env pip install wheel
# Install torch, fixing the index url. Make sure requirements.txt does not reinstall this without GPU support
# https://pytorch.org/get-started/previous-versions/
RUN conda run -n rxn-env --no-capture-output pip install torch==1.4.0 --index-url https://download.pytorch.org/whl/cu101
COPY --chown=rxn:0 docker/requirements.txt docker/requirements.txt
# Install repo dependecies, but not local repo's legacy code.
ARG PIP_INSTALL_ARGS="-r docker/requirements.txt ."
RUN conda run -n rxn-env pip install ${PIP_INSTALL_ARGS}

COPY --chown=rxn:0 docker/docker_entrypoint.sh ${LOCAL_BIN_PATH}/docker_entrypoint.sh
# ensure any users of root group to have the same access permissions
RUN chmod -R g=u ${APP_PATH}

# Package up environment
RUN conda install -c conda-forge conda-pack
RUN conda-pack --ignore-missing-files -n rxn-env -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar
RUN /venv/bin/conda-unpack


# BUILD RUNTIME CONTAINER
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu18.04 AS runtime
ARG APP_PATH
ENV APP_PATH=${APP_PATH}

ENV VENV_PATH=${APP_PATH}/venv LOCAL_BIN_PATH=${APP_PATH}/bin
RUN mkdir -p ${LOCAL_BIN_PATH}
WORKDIR ${APP_PATH}
RUN adduser --disabled-password --gecos '' rxn --uid 1000
RUN chown -R rxn:rxn ${APP_PATH}
RUN chgrp -R 0 ${APP_PATH} && chmod -R g=u  ${APP_PATH}

# Install Python environment
COPY --from=builder /venv /venv
RUN echo "source /venv/bin/activate" > ~/.bashrc
ENV PATH="/venv/bin:${PATH}"
ENV PATH="${LOCAL_BIN_PATH}:${PATH}"
RUN pip freeze

# Copy files
COPY --from=builder --chown=rxn:0 --chmod=755 ${LOCAL_BIN_PATH} ${LOCAL_BIN_PATH}
# # all local executables copied or downloaded should be executable
RUN chmod -R +x ${LOCAL_BIN_PATH}
COPY --chown=rxn:0 LICENSE README.md ./

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# Limit to one CPU core by default - this can be overwritten at runtime with
# `docker run --env OMP_NUM_THREADS=4` (or any other number than 4) if needed.
ENV OMP_NUM_THREADS=1

USER 1000
ENTRYPOINT ["docker_entrypoint.sh"]
