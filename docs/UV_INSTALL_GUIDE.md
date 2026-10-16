# RoboDojo：使用 uv 安装、验证与排雷

更新日期：2026-10-15。适用分支：`dev`。目标平台：Linux x86_64，优先 Ubuntu 22.04/24.04，NVIDIA GPU。

本文提供一条绕过主仓库 Conda 安装步骤、逐层验证仿真环境的路线。先通过 uv 管理 Python 3.11 和 `.venv`，使用 `uv pip` 分阶段安装；环境实际跑通后再整理 `pyproject.toml` 与锁文件。

**执行规则：按章节顺序执行，每个检查点通过后继续；失败时保存日志，先处理当前层。代码块默认在同一个 Bash 会话、仓库根目录执行。** 文中 `YOUR_POLICY`、`YOUR_CHECKPOINT` 等占位符需替换。不要把整篇文件当作一个脚本执行。

验证状态：本机 RTX 3090 已切换至驱动 580.178.04，Torch CUDA 小计算、CuRobo 所需 API 导入、无相机及启用相机的 AppLauncher 启动均已通过。2026-10-16 已完成资产下载与配置检查：11,611 个唯一 LFS 对象无缺失、15,365 个资产文件均已检出，doctor 为 `pass=12 warn=4 fail=0`。尚未验证真实任务、相机图像采集与 CuRobo 实际规划。**当前本机可从第 13 节策略准备继续；新终端先按第 16 节恢复环境。**

每个 Bash 代码块按一个操作拆分，执行后先核对其下的期望结果，再进行下一步。多行参数、Python 检查片段及必要的条件语句仍作为一次完整操作执行。`tee` 管道同时写日志，不是第二项安装操作。

## 0. 先理解依赖关系和 RTX 3090 驱动选择

### 0.1 项目依赖关系

```text
RoboDojo 评测客户端（项目 .venv，Python 3.11）
├── IsaacLab（Git 子模块，仿真框架）
│   └── Isaac Sim 5.1.0 / Kit 107.3.3（pip 安装，RTX/Vulkan 渲染）
├── CuRobo（Git 子模块，逆运动学、运动规划）
├── PyTorch 2.7.0+cu128（GPU 计算）
└── XPolicyLab 客户端/适配器（Git 子模块）
          ↕ WebSocket
    策略服务（独立环境，模型依赖和权重）

宿主机：NVIDIA 驱动 + EGL/Vulkan + 系统工具
可选工具链：系统 CUDA Toolkit 12.8（与驱动、Python CUDA 包分别管理）
独立下载：Assets 仿真资产、模型权重
```

| 组件 | 本指南固定版本/来源 | 说明 |
| --- | --- | --- |
| Python | 3.11 | uv 创建 `.venv` |
| Isaac Sim | 安装要求 `5.1.0`，元数据 `5.1.0.0` | 两者按 Python 包版本规则等价 |
| IsaacLab | fork 提交 `afca7b09d60d8beb9c1cb28b43066499940b969b` | 核心包 `0.54.3`；框架标识 `2.3.2`；包含 headless 相机修复 |
| CuRobo | fork 提交 `d17b54ce32cba095c0b000c4c58777075d11de0e` | 安装 `[cu12]`，默认运行时编译后端 |
| XPolicyLab | 提交 `53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2` | 服务端依赖按具体模型另行准备 |
| Torch / torchvision / torchaudio | `2.7.0+cu128` / `0.22.0+cu128` / `2.7.0+cu128` | 从 PyTorch cu128 索引安装 |
| Starlette / FastAPI | `0.45.3` / `0.115.7` | Starlette 使用显式 override，保留一条已知声明冲突 |

三个子模块由主仓库 gitlink 固定到具体提交。原 `install.sh` 使用 `--remote` 会跟随远端 main，安装日期不同可能得到不同源码；本文不使用该参数。Conda 不是仿真软件的硬性要求，原脚本主要用它创建 Python 环境。

### 0.2 RTX 3090：当前项目采用的驱动

**本机已通过启动验证的组合：Ubuntu 22.04.5 + RTX 3090 + NVIDIA 580.178.04 + Isaac Sim 5.1 / Kit 107.3.3。** 重启后实际内核为 `6.8.0-138-generic`。这不是说所有 RTX 3090 都必须使用同一驱动，而是本指南的本机复现基线。

| 驱动 | 依据和选择 |
| --- | --- |
| **580.178.04** | 本机实际采用；Torch、无相机和启用相机启动已通过 |
| 580.95.05 | Omniverse 通用技术要求表中 Ampere/Linux GeForce 的 R580 验证版本 |
| 580.65.06 | Isaac Sim 5.1 产品要求页列出的 Linux 测试版本 |
| 595.99.02 | 本机曾在 `app ready` 后于 `librtx.scenedb.plugin.so` 段错误；换为 580 后启动通过 |

参考：[Isaac Sim 5.1 产品要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)、[Omniverse 驱动表](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)、[同类上游故障 #648](https://github.com/isaac-sim/IsaacSim/discussions/648)。通用表针对最新 Omniverse，不保证旧 Kit 适配所有新驱动，也不能笼统说整个 595 分支对所有 Omniverse 都不可用。

驱动注意事项：

- `nvidia-smi` 的 CUDA Version 是驱动支持上限，580 显示 13.0 不表示系统装了 Toolkit 13.0。Toolkit 用 `nvcc --version` 检查。
- 驱动、系统 Toolkit、`.venv` 中 cu128 运行库是三个层面。安装 Toolkit 12.8 优先选 `cuda-toolkit-12-8`，避免 `cuda` 等元包顺带更换已验证驱动；不要默认全局设置 CUDA 的 `LD_LIBRARY_PATH`。
- headless 相机仍使用 RTX/Vulkan；Torch 能计算不等于渲染能启动。Docker 也使用宿主机驱动。
- 本机原 CUDA 13.2、Nsight 和 cuda-keyring 已卸载；卸载 Toolkit 不是驱动切换的必需步骤。当前 Ubuntu 软件源提供 580.178.04。
- 更换驱动前保存工作；DKMS 成功后再重启。保留多个内核及其 DKMS 模块是正常的。不混用 apt 和 `.run`，不全量 purge NVIDIA 包。

仅在驱动未准备好时执行以下切换；已经是 580.178.04 则跳过。先查看候选来源：

```bash
apt-cache policy nvidia-driver-580
```

**期望结果：** Ubuntu 22.04 软件源有 `580.178.04-0ubuntu0.22.04.1`。其他系统/来源不可直接套用下面的包版本。

模拟切换：

```bash
apt-get -s install nvidia-driver-580=580.178.04-0ubuntu0.22.04.1
```

**期望结果：** 仅替换驱动组件；本机模拟为移除 20 个 595 包、安装 21 个 580 包，无依赖错误。若仍有 NVIDIA CUDA 源造成混选，先统一包来源，不继续执行。

确认模拟正常后安装：

```bash
sudo apt-get install nvidia-driver-580=580.178.04-0ubuntu0.22.04.1
```

**期望结果：** DKMS、initramfs 阶段无失败。随后检查：

```bash
dkms status
```

**期望结果：** 580.178.04 在将要启动的内核上为 `installed`；本机在 6.8.0-40 和 6.8.0-138 两个内核上都已安装。失败先处理，勿直接重启。

```bash
sudo reboot
```

**期望结果：** 重新进入桌面。然后检查：

```bash
nvidia-smi
```

**期望结果：** 识别 RTX 3090，Driver Version 为 580.178.04。

```bash
cat /proc/driver/nvidia/version
```

**期望结果：** 内核模块也是 580.178.04。切换后再执行本文 Torch 和 AppLauncher 检查，不只凭 nvidia-smi 判断成功。

回退仅用于新驱动无法工作：进入 TTY/恢复终端，先模拟 `apt-get -s install nvidia-driver-595=595.99.02-0ubuntu0.22.04.1`，确认原版本仍可用，再去掉 `-s`、加 sudo 实际执行，DKMS 成功后重启。旧 595 仅用于恢复原系统，不能据此视为解决 Isaac 崩溃。

## 1. 为什么可以不使用 Conda

当前 `scripts/install.sh` 仅通过 Conda 创建 Python 3.11 环境。PyTorch、Isaac Sim、IsaacLab、CuRobo 和其他库均通过 Python 包安装器安装。系统依赖由 apt 提供。

| 层 | 本文管理方式 |
| --- | --- |
| Python 3.11、虚拟环境 | uv |
| Python 依赖 | uv pip，必要时在同一虚拟环境中兼容使用 pip |
| GPU 驱动、EGL/Vulkan、系统工具 | 宿主机软件包管理器 |
| IsaacLab、CuRobo 源码版本 | Git 子模块固定提交 |
| 仿真资产、模型权重 | 项目下载脚本或模型自身流程 |
| 策略模型环境 | 与仿真 `.venv` 分离 |

本仓库的 `env/` 是项目源码目录，**不可用它作为虚拟环境目录**。

当前不执行以下命令：

```bash
# 不执行：会安装/激活 Conda，并将子模块更新到远端分支。
bash scripts/install.sh -i
```

**不要执行：** 此块仅说明当前方案不使用的入口。

```bash
# 不执行：仍会尝试激活 Conda，且 --from submodules 会更新远端版本。
bash scripts/install.sh --from base_deps
```

**不要执行：** 此块仅说明当前方案不使用的入口。

```bash
# 完整运行依赖尚未声明，不以这两个命令安装本项目。
uv sync
```

**期望结果：** 命令无报错；按本节后续检查确认结果。

```bash
uv run python ...
```

**不要执行：** 此块仅说明当前方案不使用的入口。

`uv run` 默认可能锁定并同步项目环境；现阶段用已激活环境的 `python`、`bash`，或指定 `.venv/bin/python`。不要用 `uv pip sync` 同步不完整的 requirements 清单。

## 2. 固定工作目录，准备日志

```bash
cd /home/void/work/projects/RoboDojo
```

**期望结果：** 无报错；后续相对路径均以仓库根目录为基准。

```bash
git branch --show-current
```

**期望结果：** 输出 `dev`；其他分支先核实当前工作分支。

```bash
git status --short
```

**期望结果：** 列出已有修改；本指南本身可显示为已修改，不应出现整个 .venv 被跟踪。

```bash
export ROBODOJO_ROOT="$PWD"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
mkdir -p "$ROBODOJO_ROOT/.cache/uv-install/logs"
```

**期望结果：** 无报错；目录已存在时也可以成功。

```bash
export ROBODOJO_LOG_DIR="$ROBODOJO_ROOT/.cache/uv-install/logs"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
set -o pipefail
```

**期望结果：** 无输出；后续带 tee 的管道会保留失败状态。

```bash
git rev-parse HEAD | tee "$ROBODOJO_LOG_DIR/robodojo-commit.txt"
```

**期望结果：** 输出当前提交 SHA，并按命令保存记录。

```bash
git submodule status | tee "$ROBODOJO_LOG_DIR/submodules-before.txt"
```

**期望结果：** 保存安装前子模块状态；未初始化时行首 `-` 是正常的，第 7 节完成后再要求提交匹配。

检查点：分支输出为 `dev`。如果有已有修改，先记录，后续不要通过 reset/clean 清理。

当前 `.gitignore` 未忽略 `.venv`。先仅对本地排除，避免误提交环境：

```bash
if ! grep -qxF '/.venv/' .git/info/exclude; then
  printf '\n/.venv/\n' >> .git/info/exclude
fi
```

**期望结果：** 将 .venv 加入本地 Git 排除列表，已存在时不重复添加。

`.cache/` 已在仓库忽略列表中。本文把临时约束和安装记录放在那里；验证后需要长期保留的约束应另行整理进版本控制。

## 3. 检查操作系统、GPU 和空间

```bash
uname -m
```

**期望结果：** `x86_64`。

```bash
cat /etc/os-release
```

**期望结果：** Ubuntu 22.04/24.04；本机为 22.04.5。

```bash
ldd --version | head -n 1
```

**期望结果：** GLIBC 至少 2.35。

```bash
nvidia-smi
```

**期望结果：** RTX 3090 可见，Driver Version 为本机选择的 580.178.04；重定向时结果写入文件。

```bash
df -h "$ROBODOJO_ROOT" /tmp "$HOME"
```

**期望结果：** 检查仓库、临时目录和用户目录空间，预留安装及资产下载余量。

```bash
free -h
```

**期望结果：** 显示可用内存；本机约 64 GB 总内存。

检查点：

- 架构为 `x86_64`。本文不覆盖 ARM 分支。
- Isaac Sim 5.1 的 Python wheels 要求 Python 3.11 和 Linux GLIBC 2.35+。
- `nvidia-smi` 能看到目标 NVIDIA GPU。驱动既需支持 CUDA 12.8，也需兼容 Isaac Sim 5.1 的 RTX/Vulkan 渲染器；不能只检查版本是否足够新。官方 5.1 要求页列出 Linux 测试版本 580.65.06；595.x 有已报告的 RTX 启动崩溃，见第 11 节。项目 Docker 文档中的 CUDA 最低驱动说明不能代替 RTX 兼容性检查。
- 为 wheels、解压文件、扩展缓存和资产预留充足空间；建议先准备 80–100 GB 可用空间，完整权重和数据可能需要更多。这是规划余量，不是安装体积承诺。

`nvidia-smi` 顶部的 CUDA Version 表示驱动可支持的 CUDA 版本，不表示已安装对应 Toolkit。`nvcc --version` 才是系统编译工具的检查；默认 CuRobo 后端是否需要它，见第 10 节。

若 GPU 不可见、GLIBC 不满足或架构不符，停在这里处理；切换 Conda/uv 无法解决这些问题。

## 4. 准备系统工具和图形依赖

以下命令修改系统软件包，适用于 Ubuntu/Debian；其他发行版请按包名映射处理。

```bash
sudo apt-get update
```

**期望结果：** 软件源索引更新成功，无签名或网络错误。

```bash
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  git git-lfs wget curl ca-certificates ffmpeg ncurses-bin \
  libgl1 libglu1-mesa libglib2.0-0 \
  libx11-6 libxext6 libxrender1 libsm6 libice6 libxt6 \
  libxrandr2 libxi6 libxcursor1 libxinerama1 \
  libegl1 libvulkan1 vulkan-tools mesa-utils
```

**期望结果：** 软件包安装完成，无未满足依赖或 dpkg 错误。

```bash
git lfs version
```

**期望结果：** 输出 git-lfs 版本，不是 command not found。

```bash
cmake --version
```

**期望结果：** 输出 cmake 版本。

```bash
ffmpeg -version | head -n 1
```

**期望结果：** 输出 ffmpeg 版本。

这些包参考仓库 Dockerfile，比主安装脚本只装三个包更完整；不包含 NVIDIA 驱动安装。**不要把 Dockerfile 中的容器专用 NVIDIA ICD JSON 配置直接复制到宿主机。** 宿主机驱动应提供匹配的图形库和 ICD。

## 5. 安装 uv，清除环境混用

若已有 uv，先执行 `uv --version`；没有则从官方下载安装脚本，查看后执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh \
  -o "$ROBODOJO_ROOT/.cache/uv-install/install-uv.sh"
```

**期望结果：** 下载安装脚本成功；尚未执行安装。

```bash
less "$ROBODOJO_ROOT/.cache/uv-install/install-uv.sh"
```

**期望结果：** 打开安装器供阅读，按 q 退出。

```bash
sh "$ROBODOJO_ROOT/.cache/uv-install/install-uv.sh"
```

**期望结果：** uv 安装成功；根据实际输出确认可执行文件位置。

```bash
export PATH="$HOME/.local/bin:$PATH"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
uv --version
```

**期望结果：** 输出 uv 版本；不应找不到命令。

`less` 中按 `q` 退出。以安装器实际输出为准确认安装路径。

如果已经激活其他 venv，先运行 `deactivate`。如果已激活 Conda，逐层运行 `conda deactivate`，直到下面输出为空：

```bash
printf 'CONDA_PREFIX=%s\nVIRTUAL_ENV=%s\n' \
  "${CONDA_PREFIX:-}" "${VIRTUAL_ENV:-}"
```

**期望结果：** 创建 uv 环境前，两个值都应为空。

不要只手工 unset `CONDA_PREFIX` 来假装退出 Conda，PATH 和动态库路径可能仍残留。IsaacLab 会优先使用 `CONDA_PREFIX` 对应的 Python；同时激活 Conda 和 uv 会造成解释器选择不一致。

## 6. 创建 Python 3.11 环境

若 `.venv` 已存在，先检查其中 Python 版本。版本不对时按第 16 节归档，不要直接清空。

```bash
uv python install 3.11
```

**期望结果：** Python 3.11 已下载或显示已安装。

```bash
uv venv --python 3.11 --seed "$ROBODOJO_ROOT/.venv"
```

**期望结果：** 创建 .venv，Python 为 3.11，并安装 seed 工具。

```bash
source "$ROBODOJO_ROOT/.venv/bin/activate"
```

**期望结果：** 环境激活；下一步检查解释器，不只看提示符。

```bash
export PYTHONNOUSERSITE=1
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export PIP_USER=0
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export TERM=xterm-256color
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
python - <<'PY'
import os
import sys
from pathlib import Path
expected = Path(os.environ['ROBODOJO_ROOT']) / '.venv'
print('executable:', sys.executable)
print('prefix:', sys.prefix)
assert sys.version_info[:2] == (3, 11)
assert Path(sys.prefix).resolve() == expected.resolve()
assert not os.environ.get('CONDA_PREFIX'), '请先退出 Conda'
PY
```

**期望结果：** 解释器/prefix 指向项目 .venv，版本为 Python 3.11，无 Conda 混用。

```bash
python -m pip --version
```

**期望结果：** pip 路径位于当前项目 .venv。

```bash
uv --version | tee "$ROBODOJO_LOG_DIR/uv-version.txt"
```

**期望结果：** 输出 uv 版本；不应找不到命令。

检查点：Python 和 pip 都位于项目 `.venv`。`--seed` 是为了兼容第三方脚本中的 `python -m pip`，主要安装仍由 uv 执行。

后续 `uv pip` 命令显式携带 `--python`，第三方脚本通过已激活的 `VIRTUAL_ENV` 选择环境。

## 7. 固定并初始化三个子模块

```bash
git submodule sync -- third_party/IsaacLab third_party/curobo XPolicyLab
```

**期望结果：** 同步子模块 URL，无 fatal/error。

```bash
git submodule update --init --recursive -- \
  third_party/IsaacLab third_party/curobo XPolicyLab \
  2>&1 | tee "$ROBODOJO_LOG_DIR/07-submodules.log"
```

**期望结果：** 克隆/检出完成，无 fatal/error；不追踪远端最新提交。

```bash
git submodule status | tee "$ROBODOJO_LOG_DIR/submodules.txt"
```

**期望结果：** 此处记录安装前状态；未初始化时行首为 `-` 是正常的，第 7 节初始化后再检查固定提交。

```bash
test -f third_party/IsaacLab/isaaclab.sh
```

**期望结果：** 文件存在时无输出且成功；失败不要继续。

```bash
test -f third_party/curobo/pyproject.toml
```

**期望结果：** 文件存在时无输出且成功；失败不要继续。

```bash
test -f XPolicyLab/client_server/ws/model_client.py
```

**期望结果：** 文件存在时无输出且成功；失败不要继续。

这里有意不加 `--remote`。本文编写时主仓库记录的提交如下；如果你之后主动更新主仓库，以主仓库 gitlink 为准重新核对本文。

| 子模块 | 固定提交 |
| --- | --- |
| IsaacLab | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| CuRobo | `d17b54ce32cba095c0b000c4c58777075d11de0e` |
| XPolicyLab | `53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2` |

检查点：`git submodule status` 不应以 `-`（未初始化）、`+`（偏离固定提交）、`U`（冲突）开头。认证失败先解决 GitHub 访问权限；不要通过更换任意上游版本绕开。

## 8. 建立统一兼容约束，安装基础依赖

下列约束保留主安装脚本的兼容版本，并补充固定 CUDA 版 Torch。它们是兼容约束，**还不是完整锁文件**。`setuptools<82` 与固定 IsaacLab 安装脚本保持一致。

```bash
export ROBODOJO_CONSTRAINTS="$ROBODOJO_ROOT/.cache/uv-install/constraints.txt"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
cat > "$ROBODOJO_CONSTRAINTS" <<'EOF'
numpy==1.26.0
packaging==23.0
typing_extensions==4.12.2
filelock==3.13.1
websockets==12.0
scipy==1.15.3
warp-lang==1.11.0
click==8.1.7
psutil==5.9.8
wheel==0.45.1
starlette==0.45.3
stable-baselines3<2.8
onnx>=1.18,<1.22
ipython<9
virtualenv==20.30.0
setuptools<82
torch==2.7.0+cu128
torchvision==0.22.0+cu128
torchaudio==2.7.0+cu128
opencv-python-headless==4.11.0.86
EOF
```

**期望结果：** 写入约束文件，无标准输出；这是兼容约束，不是完整锁文件。

```bash
export UV_CONSTRAINT="$ROBODOJO_CONSTRAINTS"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export PIP_CONSTRAINT="$ROBODOJO_CONSTRAINTS"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export ROBODOJO_OVERRIDE="$ROBODOJO_ROOT/.cache/uv-install/overrides.txt"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
printf 'starlette==0.45.3\n' > "$ROBODOJO_OVERRIDE"
```

**期望结果：** 写入 Starlette override 文件，内容为 starlette==0.45.3。

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  'setuptools<82' 'setuptools-scm>=6.2' 'wheel==0.45.1' ninja \
  2>&1 | tee "$ROBODOJO_LOG_DIR/08-build-tools.log"
```

**期望结果：** 依赖解析、构建和安装完成，无 error；已满足时可以显示 Checked/Audited。安装包数量不固定，以下检查仍需执行。

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  -r scripts/requirements.txt \
  opencv-python-headless pillow matplotlib scipy scikit-learn pyyaml \
  2>&1 | tee "$ROBODOJO_LOG_DIR/08-base.log"
```

**期望结果：** 依赖解析、构建和安装完成，无 error；已满足时可以显示 Checked/Audited。安装包数量不固定，以下检查仍需执行。

```bash
python - <<'PY'
import numpy, cv2, yaml, open3d, transforms3d
print('numpy:', numpy.__version__)
print('opencv:', cv2.__version__)
assert numpy.__version__ == '1.26.0'
PY
```

**期望结果：** numpy 为 1.26.0，opencv 为 4.11.0；全部导入通过。

`UV_CONSTRAINT` 和 `PIP_CONSTRAINT` 使用绝对路径，确保子模块切换工作目录后仍可找到约束。它们也影响子脚本的安装命令；在其他项目中工作前应 unset，或开启新终端。

若解析提示某个新版本要求 NumPy 2，不要解除 NumPy 约束。记录包名和冲突链，选择仍支持 NumPy 1.26/Python 3.11 的版本，补入约束后重试。

## 9. 安装并验证 CUDA 版 PyTorch

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 \
  2>&1 | tee "$ROBODOJO_LOG_DIR/09-torch.log"
```

**期望结果：** 依赖解析、构建和安装完成，无 error；已满足时可以显示 Checked/Audited。安装包数量不固定，以下检查仍需执行。

```bash
python - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/09-torch-check.log"
import torch, torchvision, torchaudio
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('torchaudio:', torchaudio.__version__)
print('CUDA build:', torch.version.cuda)
assert torch.__version__ == '2.7.0+cu128'
assert torchvision.__version__ == '0.22.0+cu128'
assert torchaudio.__version__ == '2.7.0+cu128'
assert torch.cuda.is_available(), 'CUDA 不可用，先检查驱动和 GPU 可见性'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones(8, device='cuda')
assert x.sum().item() == 8
torch.cuda.synchronize()
print('CUDA tensor check passed')
PY
```

**期望结果：** Torch/vision/audio 版本为指定 cu128 组合，CUDA 可用，小计算通过。

检查点：版本和 CUDA 小计算全部通过。只有 `import torch` 成功不代表 GPU 可用。

记录 Torch 实际装入的 NVIDIA Python 库版本，防止后续 CuRobo extras 将其替换为不匹配版本：

```bash
python - <<'PY'
from importlib.metadata import distributions
from pathlib import Path
import os
path = Path(os.environ['ROBODOJO_CONSTRAINTS'])
pins = sorted({
    f"{d.metadata['Name']}=={d.version}"
    for d in distributions()
    if (d.metadata['Name'] or '').lower().replace('_', '-').startswith('nvidia-')
})
text = path.read_text()
for pin in pins:
    if pin not in text.splitlines():
        text += pin + '\n'
path.write_text(text)
print('\n'.join(pins))
PY
```

**期望结果：** 打印 NVIDIA Python 包版本并加入约束文件；无 Traceback。

如果之后出现 NVIDIA 库依赖冲突，保留这份记录并分析，不直接用 `--no-deps` 跳过解析。

## 10. 安装 Isaac Sim、IsaacLab 和 CuRobo

### 10.1 Isaac Sim 5.1.0

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaacsim.log"
```

**期望结果：** 依赖解析、构建和安装完成，无 error；已满足时可以显示 Checked/Audited。安装包数量不固定，以下检查仍需执行。

```bash
python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

actual = version('isaacsim')
print('isaacsim:', actual)
assert Version(actual) == Version('5.1.0')
PY
```

**期望结果：** 输出 isaacsim: 5.1.0.0（或等价版本），断言通过，无 Traceback。

这一步先读取元数据，不启动仿真。Isaac Sim 的包元数据可能显示 `5.1.0.0`，按 Python 包版本规则它与 `5.1.0` 等价，因此使用 `Version` 比较。`all,extscache` 下载较大，不要同时启动多份安装。

uv 默认按 `first-index` 解析多索引。如果日志显示“包存在但没有所需版本”，而 PyPI/NVIDIA 确有对应版本，可仅对本次命令添加 `--index-strategy unsafe-best-match` 重试。这会扩大候选来源，仅用于已确认的官方索引，不设置为全局默认。

如果是 uv 与 NVIDIA 包元数据或安装机制的兼容问题，可在**同一个 `.venv`** 使用项目原有 pip 路线：

```bash
"$ROBODOJO_ROOT/.venv/bin/python" -m pip install \
  -c "$ROBODOJO_CONSTRAINTS" \
  'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaacsim-pip-fallback.log"
```

**期望结果：** 命令无报错；按本节后续检查确认结果。

这是包安装器兼容回退，Python 环境仍由 uv 管理。网络、磁盘和 GLIBC 问题不应通过换安装器反复重试。

### 10.2 IsaacLab

从零安装时，本节命令通过 `UV_OVERRIDE` 将第 8 节的 override 传给脚本内所有 uv 安装调用，避免先触发已知 Starlette 冲突。原子模块不修改。此变量用途已对照 [uv 环境变量文档](https://docs.astral.sh/uv/reference/environment/#uv_override)，用户已删除并重建 .venv，重新安装后的版本、CuRobo 导入及两项启动日志检查均通过；已完成安装的本机无需重装。原有失败环境使用本节后面的“补装核心包”路线。

固定提交的 `isaaclab.sh` 已支持 `VIRTUAL_ENV`，并能在 uv 环境内使用 `uv pip`。先核对本地文件与解释器选择：

```bash
rg -n 'VIRTUAL_ENV|CONDA_PREFIX|extract_pip_command|setuptools' \
  third_party/IsaacLab/isaaclab.sh
```

**期望结果：** 显示相关配置或脚本行供核对；它是检查操作，不修改文件。

```bash
bash third_party/IsaacLab/isaaclab.sh --python -c \
  'import sys; print(sys.executable)'
```

**期望结果：** 打印项目 .venv/bin/python，不能指向 Conda。

检查点：解释器是项目 `.venv/bin/python`。若指向 Conda，返回第 5 节清理激活状态。

Isaac Sim 首次启动需要接受 NVIDIA EULA。阅读并同意后，在当前终端设置下面的变量；如果不同意，在这里停止：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
UV_OVERRIDE="$ROBODOJO_OVERRIDE" bash third_party/IsaacLab/isaaclab.sh --install none \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaaclab.log"
```

**期望结果：** 各 IsaacLab 扩展安装完成，核心包不再因 Starlette 解析失败；仍须执行下面的包元数据检查。

```bash
python - <<'PY'
from importlib.metadata import version
import importlib.util
print('isaaclab:', version('isaaclab'))
print('source:', importlib.util.find_spec('isaaclab').origin)
PY
```

**期望结果：** 输出 isaaclab: 0.54.3，source 指向原子模块；Starlette/FastAPI 为 0.45.3/0.115.7（若该片段检查）。

检查点：IsaacLab 指向仓库子模块中的源码。`none` 跳过 RL 框架 extras，不意味着只安装一个包。安装脚本中的 VS Code 配置阶段可能首次启动 Isaac；失败时区分“核心包未安装”与“编辑器配置失败”，不要只凭最后一个错误卸载整个环境。

#### Starlette 已知冲突：当前采用的 override 安装方案

固定提交存在互相冲突的依赖声明：IsaacLab 核心要求 `starlette==0.49.1`；已安装的 `isaacsim-kernel==5.1.0.0` 要求 `fastapi==0.115.7`，后者要求 `starlette>=0.40.0,<0.46.0`。因此本文保留的 `starlette==0.45.3` 无法满足未经修改的 IsaacLab。不要将约束直接改为 0.49.1，也不要通过 `--no-deps` 掩盖冲突。

另外，IsaacLab 安装脚本通过 `find -exec` 调用各扩展安装，子安装失败未必导致整体退出失败；必须查看日志中的错误并检查核心包元数据。`extra named none` 的警告和缺失 VS Code settings 的警告都不是这次核心包缺失的原因。

当前方案：保留项目固定的 IsaacLab、Isaac Sim 和已有环境，用 uv 的显式依赖 override 补装核心包。原项目安装脚本也在安装 IsaacLab 后将 Starlette 固定回 0.45.3；这里用 override 明确表达同一个版本选择，不修改 IsaacLab 源码，也不复制源码树。override 会覆盖依赖声明，而 constraints 只能与原始要求取交集。

采用依据：上游贡献者在 [Isaac Sim 5.1 依赖冲突问题 #4367](https://github.com/isaac-sim/IsaacLab/issues/4367) 中回复，这条冲突可以忽略，应不影响运行。这支持保留当前技术栈继续验证，但不是对 RoboDojo 全部功能的测试保证。**本指南将这一条声明冲突记录为已知例外；依赖解析完全一致与实际评测通过仍是两个不同的验收项。**

适用边界：当前本地 headless 评测。上游提高 Starlette 版本涉及安全修复（IsaacLab 提交 `64ecea24f`），若需要对外提供 FastAPI/Starlette HTTP 或 livestream 服务，应单独评估其安全与兼容要求；上述“不影响运行”的回复不覆盖这些要求。

如果已完成本节补装，并得到下文记录的版本和唯一冲突，无需重复安装，直接继续第 10.3 节。

在已激活 `.venv`、仍保留前文约束变量的终端中执行：

```bash
export ROBODOJO_OVERRIDE="$ROBODOJO_ROOT/.cache/uv-install/overrides.txt"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
printf 'starlette==0.45.3\n' > "$ROBODOJO_OVERRIDE"
```

**期望结果：** 写入 Starlette override 文件，内容为 starlette==0.45.3。

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" --override "$ROBODOJO_OVERRIDE" \
  -e "$ROBODOJO_ROOT/third_party/IsaacLab/source/isaaclab" \
  --dry-run \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaaclab-core-dry-run.log"
```

**期望结果：** 解析成功并预览变更，不安装；不更换已固定的 Torch/Isaac Sim 版本。

检查解析结果：不得替换既有 cu128 Torch、Isaac Sim 或其他兼容约束。若出现新的冲突，停下并记录，不能继续堆叠 override。确认后去掉 `--dry-run` 实际补装：

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" --override "$ROBODOJO_OVERRIDE" \
  -e "$ROBODOJO_ROOT/third_party/IsaacLab/source/isaaclab" \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaaclab-core-recovery.log"
```

**期望结果：** 依赖解析、构建和安装完成，无 error；已满足时可以显示 Checked/Audited。安装包数量不固定，以下检查仍需执行。

```bash
python - <<'PY'
from importlib.metadata import version
import importlib.util
print('isaaclab:', version('isaaclab'))
print('source:', importlib.util.find_spec('isaaclab').origin)
print('starlette:', version('starlette'))
print('fastapi:', version('fastapi'))
PY
```

**期望结果：** 输出 isaaclab: 0.54.3，source 指向原子模块；Starlette/FastAPI 为 0.45.3/0.115.7（若该片段检查）。

```bash
uv pip check --python "$ROBODOJO_ROOT/.venv/bin/python" \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-recovery-pip-check.log"
```

**期望结果：** 当前 override 方案仅允许一条 IsaacLab/Starlette 版本声明冲突；因此非零退出码可预期，其他冲突必须处理。

检查点：核心元数据存在，源码仍指向原子模块。`uv pip check` 此时预计会报告 IsaacLab 要求 Starlette 0.49.1、实际为 0.45.3 的冲突（可能以兼容版本区间表述）。override 不会修改包元数据，所以不能宣称依赖检查完全通过；只有这一条可以作为当前安装方案的已知例外，其他冲突必须处理。

后续若 uv 再次解析已安装的 IsaacLab 依赖，可能仍需显式传入 `--override "$ROBODOJO_OVERRIDE"`。重开终端要恢复该路径变量。不要用 `--no-deps` 跳过所有依赖。

继续第 10.3 节时，使用已带 override 参数的安装命令；随后执行第 11 节版本断言和实际 AppLauncher/渲染启动。任何新的运行时错误都需单独定位。回退可使用 `uv pip uninstall --python "$ROBODOJO_ROOT/.venv/bin/python" isaaclab`，恢复到核心包未安装的状态，已新增的依赖不会随之自动移除；彻底重建按第 16 节归档环境。

2026-10-15 用户本机验证记录：上述补装成功，核心为 `isaaclab==0.54.3`，源码指向原子模块；`isaacsim==5.1.0.0`、`starlette==0.45.3`、`fastapi==0.115.7`。`uv pip check` 检查 274 个包，仅报告上述一条 Starlette 声明冲突。此次解析将 `huggingface-hub` 从 1.16.1 调整为 0.36.2，不应未经解析直接升级回去。**该记录仅验证安装和依赖检查状态，尚未验证 CuRobo、仿真启动、相机或实际任务。**

后续执行 CuRobo、仿真、相机和实际任务验证，并逐项记录结果；安装成功不能代替运行验证。锁环境时必须保留 override 和已知元数据冲突，不能将其描述为完整满足原始依赖声明。上游对新技术栈的修复面向 Isaac Sim 6.x，不能直接视为当前 5.1 组合的修复。参考：[上游修复说明 #4145](https://github.com/isaac-sim/IsaacLab/issues/4145)、[uv override 说明](https://docs.astral.sh/uv/pip/compile/#overriding-dependency-versions)。

### 10.3 CuRobo

项目固定的 CuRobo 默认通过 `cuda.core` 后端进行运行时编译，pybind 扩展是可选路径。因此不能把旧版教程中的 nvcc 要求直接视为默认安装前提。

先核查后端选择，再安装：

```bash
rg -n 'CUROBO_USE_PYBIND|USE_PYBIND|CUDAExtension' third_party/curobo/setup.py
```

**期望结果：** 显示相关配置或脚本行供核对；它是检查操作，不修改文件。

```bash
printf 'CUROBO_USE_PYBIND=%s\n' "${CUROBO_USE_PYBIND:-未设置}"
```

**期望结果：** 命令无报错；按本节后续检查确认结果。

本文走默认后端；如果上面的变量来自你此前的实验，在当前会话清除：

```bash
unset CUROBO_USE_PYBIND
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
test -f "$ROBODOJO_OVERRIDE"
```

**期望结果：** 文件存在时无输出且成功；失败不要继续。

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  --override "$ROBODOJO_OVERRIDE" \
  -e "$ROBODOJO_ROOT/third_party/curobo[cu12]" \
  --no-build-isolation \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-curobo.log"
```

**期望结果：** 依赖解析、构建和安装完成，无 error；已满足时可以显示 Checked/Audited。安装包数量不固定，以下检查仍需执行。

```bash
python - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/10-curobo-check.log"
from importlib.metadata import version
from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ToolPoseCriteria
import curobo.runtime
print('nvidia-curobo:', version('nvidia-curobo'))
print('Required RoboDojo CuRobo APIs are importable')
PY
```

**期望结果：** 输出 CuRobo 版本及 Required RoboDojo CuRobo APIs are importable，无 ImportError。

检查点：全部项目实际使用的 API 可导入。此时只证明安装和接口层通过；首次机器人规划时才会进一步验证运行时 CUDA 编译。

如果主动选择 pybind 后端，才单独准备匹配的 CUDA Toolkit、nvcc 和额外构建依赖；本指南不把这条可选分支混入默认安装。

## 11. 验证依赖一致性与仿真启动

```bash
uv pip check --python "$ROBODOJO_ROOT/.venv/bin/python" \
  2>&1 | tee "$ROBODOJO_LOG_DIR/11-pip-check.log"
```

**期望结果：** 当前 override 方案仅允许一条 IsaacLab/Starlette 版本声明冲突；因此非零退出码可预期，其他冲突必须处理。

```bash
python - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/11-versions.log"
from importlib.metadata import version
from packaging.version import Version

expected = {
    'numpy': '1.26.0',
    'packaging': '23.0',
    'typing_extensions': '4.12.2',
    'filelock': '3.13.1',
    'websockets': '12.0',
    'scipy': '1.15.3',
    'warp-lang': '1.11.0',
    'isaacsim': '5.1.0',
    'torch': '2.7.0+cu128',
    'torchvision': '0.22.0+cu128',
    'torchaudio': '2.7.0+cu128',
}
for name, want in expected.items():
    actual = version(name)
    print(name, actual)
    assert Version(actual) == Version(want), (name, actual, want)
PY
```

**期望结果：** 打印各包版本，所有 Version 断言通过；5.1.0.0 与 5.1.0 等价。

检查点：没有未解释的依赖冲突；第 10.2 节明确说明的 Starlette 声明冲突作为当前 override 方案的已知例外。若出现其他固定版本与子模块声明的硬冲突，保存解析链，先解决；不要继续采用“装完再降级直到能 import”的方式掩盖它。

采用 override 路线时，`uv pip check` 因这一条已知冲突返回非零是预期现象，不能将退出码当作成功；核对完整输出后再单独执行后面的版本检查和启动检查。若出现其他冲突，先处理再继续。

### 11.1 无相机启动检查

已阅读并接受 NVIDIA EULA 后，确认第 10.2 节的 `OMNI_KIT_ACCEPT_EULA=YES` 在当前会话有效。执行：

```bash
python -u - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/11-app-launch-driver580.log"
from isaaclab.app import AppLauncher
print('BEFORE_LAUNCH', flush=True)
launcher = AppLauncher(headless=True)
app = launcher.app
try:
    print('AFTER_LAUNCH', flush=True)
    for i in range(5):
        app.update()
        print(f'UPDATE_{i + 1}_OK', flush=True)
    print('ISAAC_APP_OK', flush=True)
finally:
    print('BEFORE_CLOSE', flush=True)
    app.close()
PY
```

期望：`AFTER_LAUNCH`、`UPDATE_1_OK` 至 `UPDATE_5_OK`、`ISAAC_APP_OK`、关闭日志。第一次建立缓存可能耗时；app ready 单独出现不足以验收。**紧接着执行下一条，不要先运行其他命令：**

```bash
printf 'python_exit=%s tee_exit=%s\n' "${PIPESTATUS[0]}" "${PIPESTATUS[1]}"
```

期望：`python_exit=0 tee_exit=0`。这是上一条管道的状态；中间插入命令会覆盖 PIPESTATUS。

### 11.2 启用相机的渲染启动检查

```bash
python -u - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/11-render-recheck.log"
from isaaclab.app import AppLauncher
print('BEFORE_LAUNCH', flush=True)
launcher = AppLauncher(headless=True, enable_cameras=True)
app = launcher.app
try:
    print('AFTER_LAUNCH', flush=True)
    for i in range(5):
        app.update()
        print(f'UPDATE_{i + 1}_OK', flush=True)
    print('ISAAC_RENDER_APP_OK', flush=True)
finally:
    print('BEFORE_CLOSE', flush=True)
    app.close()
PY
```

期望：五次更新标记、`ISAAC_RENDER_APP_OK`，随后关闭。紧接着执行：

```bash
printf 'python_exit=%s tee_exit=%s\n' "${PIPESTATUS[0]}" "${PIPESTATUS[1]}"
```

期望：两个退出码都是 0。**app.close() 可能直接终止进程，不要求它后面的 Python 语句继续执行。** `python -u` 防止正常标记被缓冲。若缺标记或退出码非零，保留日志，停止后续安装排查。

Kit/Omniverse 模块应在 AppLauncher/SimulationApp 初始化后导入。这两项只验证启动和更新，不生成真实场景、采集相机图像或执行规划；这些由实际任务验证。

### 11.3 本机结果与独立 GUI 红色输出

2026-10-15：580.178.04 驱动下，无相机启动检查通过；相机重测输出 `AFTER_LAUNCH`、五次 `UPDATE_*_OK`、`ISAAC_RENDER_APP_OK`，Python 和 tee 均返回 0。之前 595.99.02 的 RTX 段错误未在这两项检查中复现。这里不证明真实图像内容或完整任务正确。

独立 `isaacsim` GUI 会启用更多扩展。本机日志出现 RequestsDependencyWarning 和 ROS2 Bridge startup failed，但到达 app ready，无原 RTX 段错误。前者是版本警告通过 stderr 显示为红色；后者确实表示 ROS2 中间件未配置，ROS2 功能不可用。当前 RoboDojo 使用 WebSocket，暂不需要为此安装 ROS2；需要 ROS2 时按官方安装说明单独配置，不盲目升级 Python 包来消除颜色。

## 12. 下载资产，运行仓库静态检查

`init_assets.sh` 检测到已有名为 RoboDojo 的 Conda 环境时会尝试激活它。该脚本实际用 Git/Git LFS 下载资产，本文让这个子进程不看到 Conda，同时保留 uv 环境：

```bash
ROBODOJO_SYSTEM_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
env -u BASH_ENV \
  PATH="$ROBODOJO_ROOT/.venv/bin:$ROBODOJO_SYSTEM_PATH" \
  bash --noprofile --norc -c \
  'unset -f conda 2>/dev/null || true; export GIT_LFS_SKIP_SMUDGE=1; exec bash scripts/init_assets.sh' \
  2>&1 | tee "$ROBODOJO_LOG_DIR/12-assets.log"
```

**期望结果：** 仅下载资产，不切换到 Conda；结束时出现 Assets directory is ready。

若 Git/Git LFS 安装在其他目录，把其真实目录加入上面的 PATH；先确认 `command -v conda` 在该子进程中为空。无 Conda 安装的机器可直接执行 `GIT_LFS_SKIP_SMUDGE=1 bash scripts/init_assets.sh`。该变量避免稀疏检出阶段提前下载 LFS 内容，最后仍由显式 lfs pull 下载资产。

检查资产和任务配置：

```bash
readlink -f Assets
```

**期望结果：** 显示 Assets 的真实下载缓存路径。

```bash
ls -ld Assets/Robots Assets/Object/RoboDojo \
  Assets/Material Assets/Eval_Layout/RoboDojo
```

**期望结果：** 列出的文件/目录存在；缺失时先处理，不继续。

```bash
bash scripts/robodojo.sh doctor \
  --skip-conda --skip-isaac --skip-policy \
  --summary "$ROBODOJO_LOG_DIR/12-doctor.json" \
  2>&1 | tee "$ROBODOJO_LOG_DIR/12-doctor.log"
```

**期望结果：** 最后汇总 `fail=0`。跳过 Conda、Isaac 和策略检查产生 WARN 是预期；资产缺失或配置引用失败产生的 FAIL 必须处理。

**这里跳过 Isaac 检查，是因为当前 doctor 内部仍调用 `conda run`，第 11 节已使用真实 uv 环境另行启动验证。** 只加 `--skip-conda` 并不能去掉 doctor 对 Conda 的全部依赖。

资产目录就绪检查主要验证目录存在，不是所有 LFS 内容的完整校验。遇到 USD 无法读取，检查对应文件是否仍是 LFS 指针，并查看资源缓存仓库的 `git lfs status`。`.cache/robodojo_assets_repo` 是 `Assets` 链接的目标，不要当临时垃圾清理。

若出现 LFS batch 请求 EOF，保留缓存并在已开启代理的同一终端直接续传：

```bash
git -C .cache/robodojo_assets_repo -c lfs.concurrenttransfers=4 -c lfs.activitytimeout=120 lfs pull --include="Assets/**" --exclude=""
```

期望：剩余对象下载完成，退出成功。重试进度只统计剩余对象，不是从头下载；若仍 EOF，检查代理分流或更换节点，不能仅凭 EOF 判断服务端或代理哪一端有问题。成功后重跑资产脚本完成 Assets 链接，随后执行 doctor。目录存在不能代替内容检查；本机已检查无缺失缓存对象和残留指针，但未逐个重新计算内容哈希。

仅做已有权重评测通常不需要下载训练数据。

## 13. 策略环境与第一个任务

### 13.1 先确认策略自己的要求

不同策略可能要求不同 Python/Torch/CUDA 组合，不统一安装进仿真 `.venv`。

```bash
ls XPolicyLab/policy
```

**期望结果：** 列出的文件/目录存在；缺失时先处理，不继续。

```bash
# 选择一个实际存在的策略目录后替换：
export ROBODOJO_POLICY=YOUR_POLICY
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export ROBODOJO_CKPT=YOUR_CHECKPOINT
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
ls "XPolicyLab/policy/$ROBODOJO_POLICY"
```

**期望结果：** 列出的文件/目录存在；缺失时先处理，不继续。

```bash
rg -n 'conda|uv|VIRTUAL_ENV|activate|eval_policy|python' \
  "XPolicyLab/policy/$ROBODOJO_POLICY" --glob '*.sh'
```

**期望结果：** 显示相关配置或脚本行供核对；它是检查操作，不修改文件。

阅读该策略 README、安装脚本、`deploy.yml`、`setup_eval_policy_server.sh` 和 `setup_eval_env_client.sh`。特别确认环境参数是否接受绝对 venv 路径；主 CLI 帮助中的“支持 uv”不能保证每个策略脚本都支持。

策略可在其自己的目录下创建独立 `.venv`，按其要求选择 Python。安装时另开终端，避免继承仿真用的 `UV_CONSTRAINT`、`PIP_CONSTRAINT`、`PYTHONPATH`。

仓库权重下载脚本的参数格式可先查看：

```bash
bash scripts/RoboDojo/download_ckpt.sh
```

**期望结果：** 输出数据源和策略参数帮助，不下载权重。

按帮助选择数据源和策略，下载后把 `ROBODOJO_CKPT` 设置为真实权重目录名，并核对 `deploy.yml` 中 processor 等路径。此处不假定某个模型已准备完毕。

### 13.2 推荐先跑分离的 server/client

策略服务端：在它自己的环境/终端运行。若该策略启动器明确支持环境路径，可以使用：

```bash
# 示例；必须替换成策略环境真实绝对路径。
bash scripts/robodojo.sh server \
  --policy-dir "XPolicyLab/policy/$ROBODOJO_POLICY" \
  --task stack_bowls --ckpt "$ROBODOJO_CKPT" \
  --policy-env /absolute/path/to/policy/.venv \
  --policy-port 9000 --bind-host 127.0.0.1 \
  --policy-gpu 0 --dry-run
```

**期望结果：** dry-run 输出服务启动命令；实际启动后监听选定端口。

核对 dry-run 后去掉 `--dry-run` 启动。若策略脚本仅支持 Conda，需先适配该策略启动器，或者在它支持的环境中启动服务；这不影响仿真 `.venv` 的 uv 管理。共享同一 GPU 时注意模型和 Isaac 的显存总量，可按实际机器将服务端和客户端分配到不同 GPU。

客户端：回到仿真终端，确认 `.venv` 激活、第 6 节环境变量仍有效：

```bash
export EVAL_NUM=1
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export ROBODOJO_MAX_BASH_RETRIES=1
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
bash scripts/robodojo.sh client \
  --task stack_bowls \
  --policy-dir "XPolicyLab/policy/$ROBODOJO_POLICY" \
  --policy-host 127.0.0.1 --policy-port 9000 \
  --ckpt "$ROBODOJO_CKPT" \
  --env-gpu 0 --eval-num 1 --dry-run
```

**期望结果：** dry-run 输出客户端命令；实际运行应完成一个 episode 并生成结果。

确认无误后，把最后的 `--dry-run` 去掉再执行，并记录日志：

```bash
bash scripts/robodojo.sh client \
  --task stack_bowls \
  --policy-dir "XPolicyLab/policy/$ROBODOJO_POLICY" \
  --policy-host 127.0.0.1 --policy-port 9000 \
  --ckpt "$ROBODOJO_CKPT" \
  --env-gpu 0 --eval-num 1 \
  2>&1 | tee "$ROBODOJO_LOG_DIR/13-first-eval.log"
```

**期望结果：** dry-run 输出客户端命令；实际运行应完成一个 episode 并生成结果。

当前 `client` 通过 PATH 中的 Python 启动仿真，绕开 `eval` 编排对模型专属客户端环境启动脚本的依赖。排雷阶段降低 Bash 自动重启次数，方便观察首次崩溃；Python 内部仍可能进行自己的恢复。

检查点：策略连接成功、场景和相机正常、需要的规划可执行、任务完成一个 episode 并生成评测结果。任务得分为 0 不必然表示环境失败；要区分策略失败与运行时错误。

只有分离模式通过，并核实 `setup_eval_env_client.sh` 支持 uv 环境路径后，才尝试一体化 `eval`，例如传 `--eval-env "$ROBODOJO_ROOT/.venv"`。参数能传入不等于下游能正确激活，先 dry-run，再检查实际启动日志。此阶段不要立即展开全任务 benchmark。

## 14. 常见问题：按错误所在层排查

| 现象 | 首先检查 | 处理方向 |
| --- | --- | --- |
| 安装到 Conda 或系统 Python | `sys.executable`、`CONDA_PREFIX`、`VIRTUAL_ENV` | 退出旧环境，重新激活 `.venv`，使用显式 `--python` |
| `No module named pip` | `.venv` 是否使用 `--seed` | `uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" pip` |
| `No solution found` | 最先出现的冲突链、约束文件、索引 | 固定兼容版本；不要先解除 Isaac 兼容约束 |
| NVIDIA 包无匹配 wheel | Python 版本、架构、GLIBC、官方索引 | 回到第 3/6/10 节；多索引问题再检查 index strategy |
| SSL、403、下载超时 | 代理、证书、GitHub/PyPI/NVIDIA/PyTorch 访问 | 分别检查来源；不要关闭 TLS 验证 |
| `tabs` / terminal 报错 | `TERM`、`tabs` 命令 | 安装 ncurses-bin，设置 `TERM=xterm-256color` |
| `pkg_resources` 缺失 | setuptools 版本和第三方导入链 | 恢复 `setuptools<82`，保留约束 |
| NumPy ABI / `_ARRAY_API` 报错 | NumPy 是否漂移至 2.x、扩展版本 | 恢复 NumPy 1.26.0，必要时重装对应二进制包 |
| Torch 是 CPU 版或 CUDA 不可用 | Torch 完整版本、`nvidia-smi`、GPU 可见性 | 重跑第 9 节；容器还需正确传入 GPU |
| `python-discovery` 与 filelock 冲突 | 是否装入 virtualenv 21.x | 回到 virtualenv 20.30.0；确认无人依赖后再卸载多余包 |
| CuRobo 的规划类导入失败 | `curobo.__file__`、子模块提交、包名 | 保证安装的是本项目固定 fork，不换成任意 pip curobo |
| CuRobo 首次规划编译失败 | 实际后端、CUDA runtime/NVRTC/头文件查找日志 | 先定位缺失组件；默认 runtime 与可选 pybind 分开处理 |
| `libXt.so.6` 等缺失 | 系统动态库 | 安装第 4 节对应 apt 包 |
| Vulkan/EGL/RTX 初始化失败 | 驱动图形组件、ICD、`vulkaninfo --summary` | 修复宿主机驱动/图形库；headless 相机仍需要渲染能力 |
| `No module named env` / `client_server` | 当前目录和 PYTHONPATH | 使用仓库根目录及第 6 节 PYTHONPATH；确认 XPolicyLab 已初始化 |
| USD/资产无法加载 | 路径、符号链接、LFS 指针、缓存完整性 | 检查具体资产文件；别只看顶层目录存在 |
| doctor 找不到 Conda | 当前 doctor 的固定实现 | 同时跳过 Conda/Isaac，用第 11 节独立验证 |
| WebSocket 连接拒绝/超时 | 服务端日志、地址、端口、协议 | 先确认服务监听；跨机器才改绑定地址和网络配置 |
| 显存不足 | 模型、场景、相机和并行环境总用量 | 减少并行度，分开 GPU，关闭无关 GPU 进程 |
| 退出码 134/139/99，循环重启 | 首次崩溃之前的 PhysX/CUDA/渲染日志 | 保留完整日志；自动重启不是安装成功证据 |

如果出现 `python-discovery` 冲突，可先检查：

```bash
python -m pip show python-discovery virtualenv filelock
```

**期望结果：** 查看包的实际版本及依赖；包缺失也要记录。

只有确认它是升级遗留、无实际调用者后，才执行 `uv pip uninstall --python "$ROBODOJO_ROOT/.venv/bin/python" python-discovery`，随后重跑 `uv pip check`。不要自动卸载所有冲突包。

## 15. 跑通后保存环境，准备真正的 uv 项目管理

```bash
uv pip freeze --python "$ROBODOJO_ROOT/.venv/bin/python" \
  > "$ROBODOJO_LOG_DIR/requirements-observed.txt"
```

**期望结果：** 把当前环境快照写入目标文件，通常无标准输出。

```bash
uv pip list --python "$ROBODOJO_ROOT/.venv/bin/python" \
  > "$ROBODOJO_LOG_DIR/packages.txt"
```

**期望结果：** 把当前环境快照写入目标文件，通常无标准输出。

```bash
git submodule status --recursive > "$ROBODOJO_LOG_DIR/submodules-final.txt"
```

**期望结果：** 三个子模块提交匹配表中记录；行首无 `-`、`+` 或 `U`。

```bash
nvidia-smi > "$ROBODOJO_LOG_DIR/nvidia-smi.txt"
```

**期望结果：** RTX 3090 可见，Driver Version 为本机选择的 580.178.04；重定向时结果写入文件。

```bash
cp "$ROBODOJO_CONSTRAINTS" "$ROBODOJO_LOG_DIR/constraints-final.txt"
```

**期望结果：** 复制记录成功，无输出；源文件必须存在。

```bash
cp "$ROBODOJO_OVERRIDE" "$ROBODOJO_LOG_DIR/overrides-final.txt"
```

**期望结果：** 复制记录成功，无输出；源文件必须存在。

```bash
git status --short
```

**期望结果：** 列出已有修改；本指南本身可显示为已修改，不应出现整个 .venv 被跟踪。

冻结清单可能含本机 editable 绝对路径，只是诊断快照，不能直接当跨机器锁文件。约束与包清单也不锁定系统驱动、资产版本或 Git 子模块。

完成一轮实际任务后，再进行以下仓库改造：

1. 在 `pyproject.toml` 中收紧 Python 为 `>=3.11,<3.12`，声明完整运行依赖，开发工具单独分组。
2. 配置 Torch 的专用 cu128 索引，以及 NVIDIA 包来源；依赖源配置要覆盖实际传递依赖。
3. 显式声明 IsaacLab 的各本地扩展与 CuRobo editable 依赖；不只声明一个顶层 isaaclab 包。
4. 将构建依赖和 CuRobo 构建隔离策略纳入 uv 配置，保留固定子模块提交。
5. 集中维护经过验证的约束，生成 `uv.lock`，在另一个临时环境中验证从零重建。
6. 修改安装脚本和 doctor，支持显式 Python/venv 路径，并在环境检查失败时退出。
7. 按实际策略验证 XPolicyLab 启动器；仅在需要时单独更新该子模块，不直接随意改其内容。
8. 若使用 Docker，再分别迁移 Dockerfile 和 entrypoint；本机 `.venv` 不挂载到容器复用。

本文的手工安装路线已能不依赖 Conda 管理仿真环境；完整 `uv sync` 工作流是后续可复现性改造，不能凭一个 freeze 文件视为完成。

## 16. 中断恢复、重建与日常使用

### 新终端恢复

```bash
cd /home/void/work/projects/RoboDojo
```

**期望结果：** 无报错；后续相对路径均以仓库根目录为基准。

```bash
export ROBODOJO_ROOT="$PWD"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export ROBODOJO_LOG_DIR="$ROBODOJO_ROOT/.cache/uv-install/logs"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export ROBODOJO_CONSTRAINTS="$ROBODOJO_ROOT/.cache/uv-install/constraints.txt"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
# 如果有已激活的 Conda，先逐层 conda deactivate。
export ROBODOJO_OVERRIDE="$ROBODOJO_ROOT/.cache/uv-install/overrides.txt"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
source "$ROBODOJO_ROOT/.venv/bin/activate"
```

**期望结果：** 环境激活；下一步检查解释器，不只看提示符。

```bash
export PYTHONNOUSERSITE=1
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export PIP_USER=0
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export UV_CONSTRAINT="$ROBODOJO_CONSTRAINTS"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export PIP_CONSTRAINT="$ROBODOJO_CONSTRAINTS"
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
export TERM=xterm-256color
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
set -o pipefail
```

**期望结果：** 无输出；后续带 tee 的管道会保留失败状态。

```bash
python -c 'import sys; print(sys.executable)'
```

**期望结果：** 解释器/prefix 指向项目 .venv，版本为 Python 3.11，无 Conda 混用。

```bash
# 已阅读并接受 NVIDIA EULA 后，恢复第 10.2 节对应环境变量。
```

**期望结果：** 命令无报错；按本节后续检查确认结果。

从失败阶段重跑即可，但重跑前核实前一阶段检查点。`tee` 默认覆盖同名日志；重要失败日志先另存，避免丢失首次报错。

### 归档环境后重建

下面仅归档虚拟环境，不清理源码或下载资产。先关闭使用该环境的评测和策略进程。

```bash
deactivate
```

**期望结果：** 退出当前虚拟环境，无报错。

```bash
mv "$ROBODOJO_ROOT/.venv" \
  "$ROBODOJO_ROOT/.cache/uv-install/venv-backup-$(date +%Y%m%d-%H%M%S)"
```

**期望结果：** 归档环境目录，不删除源码或资产；归档的 venv 不能直接换路径运行。

随后从第 6 节重建。**venv 内脚本通常含绝对路径，归档目录不能直接当作可迁移环境运行。** 若要恢复旧环境，先归档当前 `.venv`，再把旧目录移回原来的 `.venv` 路径；不要覆盖已有目录。

离开项目时：

```bash
unset UV_CONSTRAINT PIP_CONSTRAINT
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
unset OMNI_KIT_ACCEPT_EULA ROBODOJO_MAX_BASH_RETRIES EVAL_NUM
```

**期望结果：** 通常无输出；只调整当前终端环境变量，不写入系统配置。

```bash
deactivate
```

**期望结果：** 退出当前虚拟环境，无报错。

本文设置了专用 PYTHONPATH，其他项目请开启新终端或恢复原值。

## 17. 完成标准与问题记录模板

- [ ] `dev` 分支和主仓库提交已记录，子模块与 gitlink 一致。
- [ ] Python/pip 指向 `.venv`，没有同时激活 Conda。
- [ ] Torch cu128 版本正确且 CUDA 小计算通过。
- [ ] Isaac Sim/IsaacLab 包安装完成，CuRobo 所需 API 可导入。
- [ ] `uv pip check` 仅有已记录的 Starlette 声明冲突，兼容版本断言通过；记录 override，不能标记为完整满足原始依赖声明。
- [ ] 无相机和启用相机的 AppLauncher 启动均正常退出。
- [ ] 资产下载完整，静态 doctor `fail=0`。
- [ ] 策略服务在独立环境启动，一个真实任务完成并写出结果。
- [ ] 保存日志、包快照、约束、驱动信息和子模块提交。

遇到问题时按此格式记录，便于继续排雷；日志中不要放访问令牌：

```text
失败章节：
实际执行命令：
第一条有意义的错误（附上下文）：
日志路径：
Python 路径/版本：
uv 版本：
GPU/驱动：
最近一次通过的检查点：
是否修改过 constraints 或子模块提交：
```

## 参考资料

- [当前主安装脚本](../scripts/install.sh)、[doctor](../scripts/internal/verify_install.sh)、[Dockerfile](../Dockerfile)。
- [Isaac Sim 5.1 Python 安装：Python、GLIBC、venv、EULA](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)。
- [Isaac Sim 5.1 系统与 GPU 要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)。
- [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)、[环境管理](https://docs.astral.sh/uv/pip/environments/)、[多索引策略](https://docs.astral.sh/uv/configuration/indexes/)、[同步行为](https://docs.astral.sh/uv/concepts/projects/sync/)。
- [固定 IsaacLab 提交的安装脚本](https://github.com/yuechen0614/IsaacLab/blob/afca7b09d60d8beb9c1cb28b43066499940b969b/isaaclab.sh)：已核查 uv/Conda 环境选择。
- [固定 CuRobo 提交的包配置](https://github.com/yuechen0614/curobo/blob/d17b54ce32cba095c0b000c4c58777075d11de0e/pyproject.toml)、[后端构建选择](https://github.com/yuechen0614/curobo/blob/d17b54ce32cba095c0b000c4c58777075d11de0e/setup.py)：已核查默认后端与可选构建路径。
