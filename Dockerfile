# 使用nvidia的CUDA基础镜像
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV LANG=zh_CN.UTF-8
ENV LANGUAGE=zh_CN:zh
ENV LC_ALL=zh_CN.UTF-8

# 更新并安装依赖
# RUN apt-get update && apt-get install -y \
#     curl \
#     tmux \
#     zsh \
#     lsof \
#     net-tools \
#     nvtop \
#     cmake \
#     build-essential \
#     libncurses5-dev \
#     libncursesw5-dev \
#     locales \
#     tzdata \
#     vim \
#     tree \
#     unzip \
#     htop \
#     wget \
#     git \
#     ssh \
#     pciutils \
#     gnupg \
#     tmux \
#     ffmpeg \
#     libsm6 \
#     libxext6 \
#     libgl1 \
#     libglib2.0-0 \
#     libgl1-mesa-glx \
#     net-tools \
#     openssh-server \
#     sudo \
#     software-properties-common \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core Utilities & Shell
    curl \
    tmux \
    zsh \
    lsof \
    iproute2 \ # 替换 net-tools
    locales \
    tzdata \
    vim \
    tree \
    unzip \
    zip \ # 添加 zip
    htop \
    wget \
    git \
    ssh \
    openssh-server \
    sudo \
    pciutils \
    gnupg \
    ca-certificates \ # 添加 ca-certificates
    man-db manpages manpages-dev \ # 添加 man pages
    less \
    file \
    psmisc \
    # Enhanced CLI
    jq \
    ripgrep \
    fd-find \
    bat \
    # Development & Build
    cmake \
    build-essential \
    libncurses5-dev \
    libncursesw5-dev \
    pkg-config \ # 添加 pkg-config
    # GPU/Graphics/Multimedia
    nvtop \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    # System Monitoring
    iotop \
    sysstat \
    # Python (optional, if needed)
    # python3 python3-pip python3-venv python3-dev \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# 配置SSH
RUN mkdir /var/run/sshd && \
    chmod 0755 /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# 生成SSH主机密钥
RUN ssh-keygen -A

# 设置SSH主机密钥的正确权限
RUN chmod 600 /etc/ssh/ssh_host_rsa_key \
    && chmod 600 /etc/ssh/ssh_host_ecdsa_key \
    && chmod 600 /etc/ssh/ssh_host_ed25519_key \
    && chown root:root /etc/ssh/ssh_host_rsa_key /etc/ssh/ssh_host_ecdsa_key /etc/ssh/ssh_host_ed25519_key

# 添加用户sim
RUN useradd -m -s /bin/bash sim && \
    echo 'sim:sim' | chpasswd && \
    adduser sim sudo

# 允许用户sim免密码使用sudo
RUN echo 'sim ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
# ---------- 切换到 sim 用户进行环境安装 ----------
ENV HOME=/home/sim
WORKDIR $HOME

# 设置用户级环境变量
USER sim
ENV PATH="${HOME}/.local/bin:${HOME}/miniconda3/bin:${PATH}"

# 安装 Miniconda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p $HOME/miniconda3 && \
    rm miniconda.sh && \
    conda init bash

# 配置 conda 清华源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ && \
    conda config --set show_channel_urls yes

# 创建虚拟环境
RUN conda create -n transformers python=3.9 -y && \
    echo 'conda activate transformers' >> $HOME/.bashrc

# 安装 PyTorch 套件
RUN /bin/bash -c "source $HOME/miniconda3/etc/profile.d/conda.sh && \
    conda activate transformers && \
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia"

# 安装用户级工具
RUN /bin/bash -c "source $HOME/miniconda3/etc/profile.d/conda.sh && \
    conda activate transformers && \
    pip install transformers deepspeed ipython jupyterlab tqdm  pandas numpy matplotlib"
# RUN pip install --user \
# 安装用户级工具
RUN /bin/bash -c "source $HOME/miniconda3/etc/profile.d/conda.sh && \
    conda activate transformers && \
    pip install peft python-dateutil pytz tzdata six packaging setuptools openpyxl \
         accelerate einops  scikit-learn  \
        scipy statsmodels jinja2 pyarrow fastparquet xlsxwriter \
         ipywidgets seaborn plotly \
        xlrd beautifulsoup4 lxml  tables    \
        timm  opencv-python imgaug simclr scikit-image imageio \
        sympy pyyaml requests pipdeptree tensorboard"

RUN /bin/bash -c "source $HOME/miniconda3/etc/profile.d/conda.sh && \
    conda activate transformers && \
    pip install DiffusionEMD EMD_signal PyEMD PyWavelets  fvcore  layers  numba   \
        pyFFTW pynndescent pyts sampen tftb thop h5py"


# ---------- 切换回 root 完成系统级配置 ----------
USER root
# 暴露SSH端口
EXPOSE 22
RUN rm -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so* || true
# 创建启动脚本
RUN echo '#!/bin/bash\n\
mkdir -p /run/sshd\n\
chmod 0755 /run/sshd\n\
/usr/sbin/sshd\n\
tail -f /dev/null' > /start.sh && \
chmod +x /start.sh

# 设置容器的启动命令
CMD ["/start.sh"]