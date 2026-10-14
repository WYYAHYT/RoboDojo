# RoboDojo：使用 uv 安装、验证与排雷

编写日期：2026-10-14。适用分支：`dev`。目标平台：Linux x86_64，优先 Ubuntu 22.04/24.04，NVIDIA GPU。

本文提供一条绕过主仓库 Conda 安装步骤、逐层验证仿真环境的路线。先通过 uv 管理 Python 3.11 和 `.venv`，使用 `uv pip` 分阶段安装；环境实际跑通后再整理 `pyproject.toml` 与锁文件。

**执行规则：按章节顺序执行，每个检查点通过后继续；失败时保存日志，先处理当前层。代码块默认在同一个 Bash 会话、仓库根目录执行。** 文中 `YOUR_POLICY`、`YOUR_CHECKPOINT` 等占位符需替换。不要把整篇文件当作一个脚本执行。

验证边界：本文已检查主仓库脚本，并在线核对固定提交的 IsaacLab 安装脚本与 CuRobo 包配置；未在当前机器安装仿真栈或执行 GPU 评测。因此它是可执行的安装与诊断流程，不是已通过本机端到端测试的保证。

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

# 不执行：仍会尝试激活 Conda，且 --from submodules 会更新远端版本。
bash scripts/install.sh --from base_deps

# 完整运行依赖尚未声明，不以这两个命令安装本项目。
uv sync
uv run python ...
```

`uv run` 默认可能锁定并同步项目环境；现阶段用已激活环境的 `python`、`bash`，或指定 `.venv/bin/python`。不要用 `uv pip sync` 同步不完整的 requirements 清单。

## 2. 固定工作目录，准备日志

```bash
cd /home/void/work/projects/RoboDojo
git branch --show-current
git status --short

export ROBODOJO_ROOT="$PWD"
mkdir -p "$ROBODOJO_ROOT/.cache/uv-install/logs"
export ROBODOJO_LOG_DIR="$ROBODOJO_ROOT/.cache/uv-install/logs"
set -o pipefail

git rev-parse HEAD | tee "$ROBODOJO_LOG_DIR/robodojo-commit.txt"
git submodule status | tee "$ROBODOJO_LOG_DIR/submodules-before.txt"
```

检查点：分支输出为 `dev`。如果有已有修改，先记录，后续不要通过 reset/clean 清理。

当前 `.gitignore` 未忽略 `.venv`。先仅对本地排除，避免误提交环境：

```bash
if ! grep -qxF '/.venv/' .git/info/exclude; then
  printf '\n/.venv/\n' >> .git/info/exclude
fi
```

`.cache/` 已在仓库忽略列表中。本文把临时约束和安装记录放在那里；验证后需要长期保留的约束应另行整理进版本控制。

## 3. 检查操作系统、GPU 和空间

```bash
uname -m
cat /etc/os-release
ldd --version | head -n 1
nvidia-smi
df -h "$ROBODOJO_ROOT" /tmp "$HOME"
free -h
```

检查点：

- 架构为 `x86_64`。本文不覆盖 ARM 分支。
- Isaac Sim 5.1 的 Python wheels 要求 Python 3.11 和 Linux GLIBC 2.35+。
- `nvidia-smi` 能看到目标 NVIDIA GPU。驱动需支持所选 CUDA 12.8 栈，并满足 Isaac Sim 的 GPU/驱动要求；可用项目 Docker 文档中的 570.x 或更新驱动作为核查起点，具体硬件仍需对照 NVIDIA 要求。
- 为 wheels、解压文件、扩展缓存和资产预留充足空间；建议先准备 80–100 GB 可用空间，完整权重和数据可能需要更多。这是规划余量，不是安装体积承诺。

`nvidia-smi` 顶部的 CUDA Version 表示驱动可支持的 CUDA 版本，不表示已安装对应 Toolkit。`nvcc --version` 才是系统编译工具的检查；默认 CuRobo 后端是否需要它，见第 10 节。

若 GPU 不可见、GLIBC 不满足或架构不符，停在这里处理；切换 Conda/uv 无法解决这些问题。

## 4. 准备系统工具和图形依赖

以下命令修改系统软件包，适用于 Ubuntu/Debian；其他发行版请按包名映射处理。

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  git git-lfs wget curl ca-certificates ffmpeg ncurses-bin \
  libgl1 libglu1-mesa libglib2.0-0 \
  libx11-6 libxext6 libxrender1 libsm6 libice6 libxt6 \
  libxrandr2 libxi6 libxcursor1 libxinerama1 \
  libegl1 libvulkan1 vulkan-tools mesa-utils

git lfs version
cmake --version
ffmpeg -version | head -n 1
```

这些包参考仓库 Dockerfile，比主安装脚本只装三个包更完整；不包含 NVIDIA 驱动安装。**不要把 Dockerfile 中的容器专用 NVIDIA ICD JSON 配置直接复制到宿主机。** 宿主机驱动应提供匹配的图形库和 ICD。

## 5. 安装 uv，清除环境混用

若已有 uv，先执行 `uv --version`；没有则从官方下载安装脚本，查看后执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh \
  -o "$ROBODOJO_ROOT/.cache/uv-install/install-uv.sh"
less "$ROBODOJO_ROOT/.cache/uv-install/install-uv.sh"
sh "$ROBODOJO_ROOT/.cache/uv-install/install-uv.sh"
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

`less` 中按 `q` 退出。以安装器实际输出为准确认安装路径。

如果已经激活其他 venv，先运行 `deactivate`。如果已激活 Conda，逐层运行 `conda deactivate`，直到下面输出为空：

```bash
printf 'CONDA_PREFIX=%s\nVIRTUAL_ENV=%s\n' \
  "${CONDA_PREFIX:-}" "${VIRTUAL_ENV:-}"
```

不要只手工 unset `CONDA_PREFIX` 来假装退出 Conda，PATH 和动态库路径可能仍残留。IsaacLab 会优先使用 `CONDA_PREFIX` 对应的 Python；同时激活 Conda 和 uv 会造成解释器选择不一致。

## 6. 创建 Python 3.11 环境

若 `.venv` 已存在，先检查其中 Python 版本。版本不对时按第 16 节归档，不要直接清空。

```bash
uv python install 3.11
uv venv --python 3.11 --seed "$ROBODOJO_ROOT/.venv"
source "$ROBODOJO_ROOT/.venv/bin/activate"

export PYTHONNOUSERSITE=1
export PIP_USER=0
export PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab"
export TERM=xterm-256color

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
python -m pip --version
uv --version | tee "$ROBODOJO_LOG_DIR/uv-version.txt"
```

检查点：Python 和 pip 都位于项目 `.venv`。`--seed` 是为了兼容第三方脚本中的 `python -m pip`，主要安装仍由 uv 执行。

后续 `uv pip` 命令显式携带 `--python`，第三方脚本通过已激活的 `VIRTUAL_ENV` 选择环境。

## 7. 固定并初始化三个子模块

```bash
git submodule sync -- third_party/IsaacLab third_party/curobo XPolicyLab
git submodule update --init --recursive -- \
  third_party/IsaacLab third_party/curobo XPolicyLab \
  2>&1 | tee "$ROBODOJO_LOG_DIR/07-submodules.log"

git submodule status | tee "$ROBODOJO_LOG_DIR/submodules.txt"
test -f third_party/IsaacLab/isaaclab.sh
test -f third_party/curobo/pyproject.toml
test -f XPolicyLab/client_server/ws/model_client.py
```

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

export UV_CONSTRAINT="$ROBODOJO_CONSTRAINTS"
export PIP_CONSTRAINT="$ROBODOJO_CONSTRAINTS"

uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  'setuptools<82' 'setuptools-scm>=6.2' 'wheel==0.45.1' ninja \
  2>&1 | tee "$ROBODOJO_LOG_DIR/08-build-tools.log"

uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  -r scripts/requirements.txt \
  opencv-python-headless pillow matplotlib scipy scikit-learn pyyaml \
  2>&1 | tee "$ROBODOJO_LOG_DIR/08-base.log"

python - <<'PY'
import numpy, cv2, yaml, open3d, transforms3d
print('numpy:', numpy.__version__)
print('opencv:', cv2.__version__)
assert numpy.__version__ == '1.26.0'
PY
```

`UV_CONSTRAINT` 和 `PIP_CONSTRAINT` 使用绝对路径，确保子模块切换工作目录后仍可找到约束。它们也影响子脚本的安装命令；在其他项目中工作前应 unset，或开启新终端。

若解析提示某个新版本要求 NumPy 2，不要解除 NumPy 约束。记录包名和冲突链，选择仍支持 NumPy 1.26/Python 3.11 的版本，补入约束后重试。

## 9. 安装并验证 CUDA 版 PyTorch

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 \
  2>&1 | tee "$ROBODOJO_LOG_DIR/09-torch.log"

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

如果之后出现 NVIDIA 库依赖冲突，保留这份记录并分析，不直接用 `--no-deps` 跳过解析。

## 10. 安装 Isaac Sim、IsaacLab 和 CuRobo

### 10.1 Isaac Sim 5.1.0

```bash
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaacsim.log"

python - <<'PY'
from importlib.metadata import version
print('isaacsim:', version('isaacsim'))
assert version('isaacsim') == '5.1.0'
PY
```

这一步先读取元数据，不启动仿真。`all,extscache` 下载较大，不要同时启动多份安装。

uv 默认按 `first-index` 解析多索引。如果日志显示“包存在但没有所需版本”，而 PyPI/NVIDIA 确有对应版本，可仅对本次命令添加 `--index-strategy unsafe-best-match` 重试。这会扩大候选来源，仅用于已确认的官方索引，不设置为全局默认。

如果是 uv 与 NVIDIA 包元数据或安装机制的兼容问题，可在**同一个 `.venv`** 使用项目原有 pip 路线：

```bash
"$ROBODOJO_ROOT/.venv/bin/python" -m pip install \
  -c "$ROBODOJO_CONSTRAINTS" \
  'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaacsim-pip-fallback.log"
```

这是包安装器兼容回退，Python 环境仍由 uv 管理。网络、磁盘和 GLIBC 问题不应通过换安装器反复重试。

### 10.2 IsaacLab

固定提交的 `isaaclab.sh` 已支持 `VIRTUAL_ENV`，并能在 uv 环境内使用 `uv pip`。先核对本地文件与解释器选择：

```bash
rg -n 'VIRTUAL_ENV|CONDA_PREFIX|extract_pip_command|setuptools' \
  third_party/IsaacLab/isaaclab.sh

bash third_party/IsaacLab/isaaclab.sh --python -c \
  'import sys; print(sys.executable)'
```

检查点：解释器是项目 `.venv/bin/python`。若指向 Conda，返回第 5 节清理激活状态。

Isaac Sim 首次启动需要接受 NVIDIA EULA。阅读并同意后，在当前终端设置下面的变量；如果不同意，在这里停止：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
bash third_party/IsaacLab/isaaclab.sh --install none \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaaclab.log"

python - <<'PY'
from importlib.metadata import version
import importlib.util
print('isaaclab:', version('isaaclab'))
print('source:', importlib.util.find_spec('isaaclab').origin)
PY
```

检查点：IsaacLab 指向仓库子模块中的源码。`none` 跳过 RL 框架 extras，不意味着只安装一个包。安装脚本中的 VS Code 配置阶段可能首次启动 Isaac；失败时区分“核心包未安装”与“编辑器配置失败”，不要只凭最后一个错误卸载整个环境。

### 10.3 CuRobo

项目固定的 CuRobo 默认通过 `cuda.core` 后端进行运行时编译，pybind 扩展是可选路径。因此不能把旧版教程中的 nvcc 要求直接视为默认安装前提。

先核查后端选择，再安装：

```bash
rg -n 'CUROBO_USE_PYBIND|USE_PYBIND|CUDAExtension' third_party/curobo/setup.py
printf 'CUROBO_USE_PYBIND=%s\n' "${CUROBO_USE_PYBIND:-未设置}"
```

本文走默认后端；如果上面的变量来自你此前的实验，在当前会话清除：

```bash
unset CUROBO_USE_PYBIND
uv pip install --python "$ROBODOJO_ROOT/.venv/bin/python" \
  -c "$ROBODOJO_CONSTRAINTS" \
  -e "$ROBODOJO_ROOT/third_party/curobo[cu12]" \
  --no-build-isolation \
  2>&1 | tee "$ROBODOJO_LOG_DIR/10-curobo.log"

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

检查点：全部项目实际使用的 API 可导入。此时只证明安装和接口层通过；首次机器人规划时才会进一步验证运行时 CUDA 编译。

如果主动选择 pybind 后端，才单独准备匹配的 CUDA Toolkit、nvcc 和额外构建依赖；本指南不把这条可选分支混入默认安装。

## 11. 验证依赖一致性与仿真启动

```bash
uv pip check --python "$ROBODOJO_ROOT/.venv/bin/python" \
  2>&1 | tee "$ROBODOJO_LOG_DIR/11-pip-check.log"

python - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/11-versions.log"
from importlib.metadata import version
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
    assert actual == want, (name, actual, want)
PY
```

检查点：没有未解释的依赖冲突。若某个固定版本与子模块声明硬冲突，保存解析链，先解决；不要继续采用“装完再降级直到能 import”的方式掩盖它。

启动不含相机的最小 IsaacLab 应用；下面不会启动策略服务：

```bash
python - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/11-app-launch.log"
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
try:
    import isaaclab
    import isaacsim
    for _ in range(5):
        app.update()
    print('ISAAC_APP_OK')
finally:
    app.close()
PY
```

然后单独验证相机所需的渲染启动路径：

```bash
python - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/11-render-launch.log"
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=True)
app = launcher.app
try:
    for _ in range(5):
        app.update()
    print('ISAAC_RENDER_APP_OK')
finally:
    app.close()
PY
```

检查点：正常退出并输出对应标记。第二个检查覆盖渲染启动，不包含真实相机图像采集；实际图像与规划仍需第 13 节评测验证。首次启动可能建立缓存，较慢时观察日志与 GPU/磁盘活动。

Kit/Omniverse 子模块应在 AppLauncher/SimulationApp 初始化之后导入；不要为了检查包就提前导入全部 `omni.*` 模块。

## 12. 下载资产，运行仓库静态检查

`init_assets.sh` 检测到已有名为 RoboDojo 的 Conda 环境时会尝试激活它。该脚本实际用 Git/Git LFS 下载资产，本文让这个子进程不看到 Conda，同时保留 uv 环境：

```bash
ROBODOJO_SYSTEM_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
env -u BASH_ENV \
  PATH="$ROBODOJO_ROOT/.venv/bin:$ROBODOJO_SYSTEM_PATH" \
  bash --noprofile --norc -c \
  'unset -f conda 2>/dev/null || true; exec bash scripts/init_assets.sh' \
  2>&1 | tee "$ROBODOJO_LOG_DIR/12-assets.log"
```

若 Git/Git LFS 安装在其他目录，把其真实目录加入上面的 PATH；先确认 `command -v conda` 在该子进程中为空。无 Conda 安装的机器也可以直接 `bash scripts/init_assets.sh`。

检查资产和任务配置：

```bash
readlink -f Assets
ls -ld Assets/Robots Assets/Object/RoboDojo \
  Assets/Material Assets/Eval_Layout/RoboDojo

bash scripts/robodojo.sh doctor \
  --skip-conda --skip-isaac --skip-policy \
  --summary "$ROBODOJO_LOG_DIR/12-doctor.json" \
  2>&1 | tee "$ROBODOJO_LOG_DIR/12-doctor.log"
```

检查点：doctor 的 `fail=0`，跳过检查产生的 WARN 有明确原因。

**这里跳过 Isaac 检查，是因为当前 doctor 内部仍调用 `conda run`，第 11 节已使用真实 uv 环境另行启动验证。** 只加 `--skip-conda` 并不能去掉 doctor 对 Conda 的全部依赖。

资产目录就绪检查主要验证目录存在，不是所有 LFS 内容的完整校验。遇到 USD 无法读取，检查对应文件是否仍是 LFS 指针，并查看资源缓存仓库的 `git lfs status`。`.cache/robodojo_assets_repo` 是 `Assets` 链接的目标，不要当临时垃圾清理。

仅做已有权重评测通常不需要下载训练数据。

## 13. 策略环境与第一个任务

### 13.1 先确认策略自己的要求

不同策略可能要求不同 Python/Torch/CUDA 组合，不统一安装进仿真 `.venv`。

```bash
ls XPolicyLab/policy
# 选择一个实际存在的策略目录后替换：
export ROBODOJO_POLICY=YOUR_POLICY
export ROBODOJO_CKPT=YOUR_CHECKPOINT

ls "XPolicyLab/policy/$ROBODOJO_POLICY"
rg -n 'conda|uv|VIRTUAL_ENV|activate|eval_policy|python' \
  "XPolicyLab/policy/$ROBODOJO_POLICY" --glob '*.sh'
```

阅读该策略 README、安装脚本、`deploy.yml`、`setup_eval_policy_server.sh` 和 `setup_eval_env_client.sh`。特别确认环境参数是否接受绝对 venv 路径；主 CLI 帮助中的“支持 uv”不能保证每个策略脚本都支持。

策略可在其自己的目录下创建独立 `.venv`，按其要求选择 Python。安装时另开终端，避免继承仿真用的 `UV_CONSTRAINT`、`PIP_CONSTRAINT`、`PYTHONPATH`。

仓库权重下载脚本的参数格式可先查看：

```bash
bash scripts/RoboDojo/download_ckpt.sh
```

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

核对 dry-run 后去掉 `--dry-run` 启动。若策略脚本仅支持 Conda，需先适配该策略启动器，或者在它支持的环境中启动服务；这不影响仿真 `.venv` 的 uv 管理。共享同一 GPU 时注意模型和 Isaac 的显存总量，可按实际机器将服务端和客户端分配到不同 GPU。

客户端：回到仿真终端，确认 `.venv` 激活、第 6 节环境变量仍有效：

```bash
export EVAL_NUM=1
export ROBODOJO_MAX_BASH_RETRIES=1

bash scripts/robodojo.sh client \
  --task stack_bowls \
  --policy-dir "XPolicyLab/policy/$ROBODOJO_POLICY" \
  --policy-host 127.0.0.1 --policy-port 9000 \
  --ckpt "$ROBODOJO_CKPT" \
  --env-gpu 0 --eval-num 1 --dry-run
```

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

只有确认它是升级遗留、无实际调用者后，才执行 `uv pip uninstall --python "$ROBODOJO_ROOT/.venv/bin/python" python-discovery`，随后重跑 `uv pip check`。不要自动卸载所有冲突包。

## 15. 跑通后保存环境，准备真正的 uv 项目管理

```bash
uv pip freeze --python "$ROBODOJO_ROOT/.venv/bin/python" \
  > "$ROBODOJO_LOG_DIR/requirements-observed.txt"
uv pip list --python "$ROBODOJO_ROOT/.venv/bin/python" \
  > "$ROBODOJO_LOG_DIR/packages.txt"
git submodule status --recursive > "$ROBODOJO_LOG_DIR/submodules-final.txt"
nvidia-smi > "$ROBODOJO_LOG_DIR/nvidia-smi.txt"
cp "$ROBODOJO_CONSTRAINTS" "$ROBODOJO_LOG_DIR/constraints-final.txt"
git status --short
```

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
export ROBODOJO_ROOT="$PWD"
export ROBODOJO_LOG_DIR="$ROBODOJO_ROOT/.cache/uv-install/logs"
export ROBODOJO_CONSTRAINTS="$ROBODOJO_ROOT/.cache/uv-install/constraints.txt"
# 如果有已激活的 Conda，先逐层 conda deactivate。
source "$ROBODOJO_ROOT/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab"
export UV_CONSTRAINT="$ROBODOJO_CONSTRAINTS"
export PIP_CONSTRAINT="$ROBODOJO_CONSTRAINTS"
export TERM=xterm-256color
set -o pipefail
python -c 'import sys; print(sys.executable)'
# 已阅读并接受 NVIDIA EULA 后，恢复第 10.2 节对应环境变量。
```

从失败阶段重跑即可，但重跑前核实前一阶段检查点。`tee` 默认覆盖同名日志；重要失败日志先另存，避免丢失首次报错。

### 归档环境后重建

下面仅归档虚拟环境，不清理源码或下载资产。先关闭使用该环境的评测和策略进程。

```bash
deactivate
mv "$ROBODOJO_ROOT/.venv" \
  "$ROBODOJO_ROOT/.cache/uv-install/venv-backup-$(date +%Y%m%d-%H%M%S)"
```

随后从第 6 节重建。**venv 内脚本通常含绝对路径，归档目录不能直接当作可迁移环境运行。** 若要恢复旧环境，先归档当前 `.venv`，再把旧目录移回原来的 `.venv` 路径；不要覆盖已有目录。

离开项目时：

```bash
unset UV_CONSTRAINT PIP_CONSTRAINT
unset OMNI_KIT_ACCEPT_EULA ROBODOJO_MAX_BASH_RETRIES EVAL_NUM
deactivate
```

本文设置了专用 PYTHONPATH，其他项目请开启新终端或恢复原值。

## 17. 完成标准与问题记录模板

- [ ] `dev` 分支和主仓库提交已记录，子模块与 gitlink 一致。
- [ ] Python/pip 指向 `.venv`，没有同时激活 Conda。
- [ ] Torch cu128 版本正确且 CUDA 小计算通过。
- [ ] Isaac Sim/IsaacLab 包安装完成，CuRobo 所需 API 可导入。
- [ ] `uv pip check` 无未解决冲突，兼容版本断言通过。
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
