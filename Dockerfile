# Official Microsoft Python Dev Container Image
FROM mcr.microsoft.com/devcontainers/python:3.11

# Set Environment Variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HOME=/home/cuser \
    PATH=/home/cuser/.local/bin:${PATH}

# Install system dependencies and cache cleanup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add a non-root user for better security
RUN groupadd -r cuser \
    && useradd -r -g cuser cuser \
    && mkdir -p ${HOME} \
    && chown -R cuser:cuser ${HOME}

# Set working directory
WORKDIR /workspace

# Copy requierements and notebooks
COPY --chown=cuser:cuser requirements.txt ./
COPY --chown=cuser:cuser notebooks/ ./notebooks/
COPY --chown=cuser:cuser data/ ./data/

# Change to non-root user
USER cuser

# Install requierements, additional Jupyter extensions and tools
RUN python -m pip install --upgrade pip \
    && pip install --user --no-cache-dir -r requirements.txt \
    && python -m ipykernel install --user --name python3_cuser --display-name "Python 3 (cuser)"

# Configure Jupyter Lab
RUN mkdir -p ${HOME}/.jupyter && \
    cat <<EOF > ${HOME}/.jupyter/jupyter_server_config.py
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.root_dir = '/workspace'
EOF

# Expose jupyter lab ports
EXPOSE 8888 

# Entry point to start Jupyter Lab
ENTRYPOINT ["jupyter","lab"]
CMD ["--ip=0.0.0.0","--port=8888","--no-browser"]