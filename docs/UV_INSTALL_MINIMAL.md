# RoboDojo 最简 uv 安装

适用：Ubuntu 22.04、RTX 3090、Python 3.11。版本基线：NVIDIA 驱动 **580.178.04**、Torch **2.7.0+cu128**、Isaac Sim **5.1.0**、仓库固定的 IsaacLab/CuRobo/XPolicyLab 子模块。

依赖关系：RoboDojo → IsaacLab → Isaac Sim；CuRobo 提供运动规划；XPolicyLab 提供策略通信和独立模型服务。uv 管理仿真 `.venv`，系统驱动和可选 CUDA Toolkit 单独管理。

2026-10-16 本机阶段验收：重新创建的环境版本/GPU/API/启动检查通过，资产全部检出，doctor `fail=0`；实际策略任务尚未验证。

**每个代码块单独执行，检查通过再继续。已有环境且启动检查通过的用户，直接从第 7 节继续；资产也已验收的当前本机，直接准备策略环境和权重。** 详细故障处理见 [完整指南](UV_INSTALL_GUIDE.md)。本文不执行原 `install.sh`，不使用 `--remote` 或 `uv sync`。

## 1. 系统准备

```bash
nvidia-smi
```

期望：RTX 3090，驱动 `580.178.04`。本机原 `595.99.02` 曾引起 Isaac RTX 崩溃；驱动切换见完整指南。此版本是本机验证基线，并非所有 3090 的硬性要求。

```bash
sudo apt update
```

期望：无软件源错误。

```bash
sudo apt-get install -y build-essential cmake ninja-build pkg-config git git-lfs wget curl ca-certificates ffmpeg ncurses-bin libgl1 libglu1-mesa libglib2.0-0 libx11-6 libxext6 libxrender1 libsm6 libice6 libxt6 libxrandr2 libxi6 libxcursor1 libxinerama1 libegl1 libvulkan1
```

期望：安装完成。系统须为 x86_64、GLIBC ≥2.35。系统 Toolkit 12.8 可选；默认 CuRobo 后端仍需可用的 CUDA runtime/NVRTC 库和头文件。不要全局添加 CUDA `LD_LIBRARY_PATH`，也不要用 CUDA 驱动元包覆盖已验证的驱动。

## 2. uv 和虚拟环境

如果已有 uv，跳过安装器操作。

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/robodojo-install-uv.sh
```

期望：下载成功；执行前查看脚本内容。

```bash
sh /tmp/robodojo-install-uv.sh
```

期望：uv 安装完成。

```bash
export PATH="$HOME/.local/bin:$PATH"
```

期望：无输出，当前终端可找到 uv。

退出其他 venv/Conda 后，进入已克隆的仓库（按实际路径调整）：

```bash
cd ~/work/projects/RoboDojo
```

期望：当前目录为仓库根目录。

```bash
export ROBODOJO_ROOT="$PWD"
```

```bash
mkdir -p "$ROBODOJO_ROOT/.cache/uv-install/logs"
```

```bash
export ROBODOJO_LOG_DIR="$ROBODOJO_ROOT/.cache/uv-install/logs"
```

以上操作期望无报错；后续在同一 Bash 终端执行。

```bash
set -o pipefail
```

期望：无输出，管道中的安装失败不会被 tee 掩盖。

```bash
uv python install 3.11
```

期望：Python 3.11 已安装。

```bash
uv venv --python 3.11 --seed .venv
```

期望：创建环境并包含 pip；不要覆盖已有环境。仓库 `env/` 是源码，不能当作 venv 目录。

```bash
source .venv/bin/activate
```

```bash
export PYTHONNOUSERSITE=1 PIP_USER=0 TERM=xterm-256color PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab"
```

```bash
python -c 'import os,sys; print(sys.executable); assert sys.version_info[:2] == (3,11); assert sys.prefix == os.environ["ROBODOJO_ROOT"] + "/.venv"; assert not os.environ.get("CONDA_PREFIX")'
```

期望：项目 `.venv/bin/python`，断言通过。不要同时激活 Conda。

## 3. 固定子模块

```bash
git submodule update --init --recursive
```

```bash
git submodule status
```

期望：三个子模块已检出，行首无 `-`、`+` 或 `U`。不要加 `--remote`。

## 4. 约束和基础依赖

```bash
export ROBODOJO_CONSTRAINTS="$ROBODOJO_ROOT/.cache/uv-install/constraints.txt" ROBODOJO_OVERRIDE="$ROBODOJO_ROOT/.cache/uv-install/overrides.txt"
```

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

期望：约束文件写入。已有环境不要覆盖原约束文件，其中可能还有已记录的 NVIDIA 库版本。

```bash
printf 'starlette==0.45.3\n' > "$ROBODOJO_OVERRIDE"
```

```bash
export UV_CONSTRAINT="$ROBODOJO_CONSTRAINTS" PIP_CONSTRAINT="$ROBODOJO_CONSTRAINTS"
```

期望：后续子脚本也使用约束。Starlette override 用于处理 IsaacLab 与 Isaac Sim 的已知声明冲突，不修改源码。

```bash
uv pip install --python .venv/bin/python -c "$ROBODOJO_CONSTRAINTS" 'setuptools<82' 'setuptools-scm>=6.2' wheel ninja
```

```bash
uv pip install --python .venv/bin/python -c "$ROBODOJO_CONSTRAINTS" -r scripts/requirements.txt opencv-python-headless pillow matplotlib scipy scikit-learn pyyaml
```

期望：两次安装各自完成，无解析或构建错误。

## 5. Torch、Isaac 和 CuRobo

```bash
uv pip install --python .venv/bin/python -c "$ROBODOJO_CONSTRAINTS" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

```bash
python -c 'import torch; print(torch.__version__); assert torch.__version__ == "2.7.0+cu128"; assert torch.cuda.is_available(); assert torch.ones(8,device="cuda").sum().item() == 8; torch.cuda.synchronize(); print("TORCH_CUDA_OK")'
```

期望：`2.7.0+cu128`、`TORCH_CUDA_OK`。

```bash
uv pip install --python .venv/bin/python -c "$ROBODOJO_CONSTRAINTS" 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
```

期望：安装完成；若多索引解析失败，按完整指南排查，不直接解除约束。

阅读并接受 NVIDIA EULA 后执行：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

```bash
UV_OVERRIDE="$ROBODOJO_OVERRIDE" bash third_party/IsaacLab/isaaclab.sh --install none 2>&1 | tee "$ROBODOJO_LOG_DIR/10-isaaclab.log"
```

期望：各扩展安装完成。该脚本可能漏传子安装错误，必须执行下一项检查。用户已重新创建环境安装，随后版本、API 和两项启动检查均通过。

```bash
python -c 'from importlib.metadata import version; from packaging.version import Version; print("isaaclab:",version("isaaclab")); print("isaacsim:",version("isaacsim")); assert Version(version("isaacsim")) == Version("5.1.0")'
```

期望：`isaaclab: 0.54.3`、`isaacsim: 5.1.0.0`。核心包缺失时按完整指南第 10.2 节补装。

```bash
unset CUROBO_USE_PYBIND
```

```bash
uv pip install --python .venv/bin/python -c "$ROBODOJO_CONSTRAINTS" --override "$ROBODOJO_OVERRIDE" -e './third_party/curobo[cu12]' --no-build-isolation
```

```bash
python -c 'from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg; from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg; from curobo.motion_planner import MotionPlanner; from curobo.types import ToolPoseCriteria; import curobo.runtime; print("CUROBO_IMPORT_OK")'
```

期望：`CUROBO_IMPORT_OK`。这不验证实际规划或首次内核编译。

```bash
uv pip check --python .venv/bin/python
```

期望：仅一条 `isaaclab requires starlette==0.49.1, but 0.45.3 is installed`。该已知例外会返回非零；其他冲突必须处理。上游回复见完整指南；不要将其当作所有声明完全一致，也不要用于未经评估的对外 HTTP/livestream 服务。

## 6. 启动验证

先验证无相机启动：

```bash
python -u - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/11-app-launch.log"
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
try:
    for _ in range(5):
        app.update()
    print('ISAAC_APP_OK', flush=True)
finally:
    app.close()
PY
```

紧接着检查退出码，不插入其他命令：

```bash
printf 'python_exit=%s tee_exit=%s\n' "${PIPESTATUS[0]}" "${PIPESTATUS[1]}"
```

期望：`ISAAC_APP_OK`，两个退出码均为 0。

再验证相机渲染启动：

```bash
python -u - <<'PY' 2>&1 | tee "$ROBODOJO_LOG_DIR/11-render-launch.log"
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=True)
app = launcher.app
try:
    for _ in range(5):
        app.update()
    print('ISAAC_RENDER_APP_OK', flush=True)
finally:
    app.close()
PY
```

紧接着检查：

```bash
printf 'python_exit=%s tee_exit=%s\n' "${PIPESTATUS[0]}" "${PIPESTATUS[1]}"
```

期望：`ISAAC_RENDER_APP_OK`，两个退出码均为 0。`app.close()` 可能直接结束进程，不要求后续 Python 打印。到这里，仿真虚拟环境准备完成。

## 7. 资产和下一步

没有安装 Conda 的机器直接执行：

```bash
GIT_LFS_SKIP_SMUDGE=1 bash scripts/init_assets.sh
```

期望：`Assets directory is ready`。若机器保留 RoboDojo Conda 环境，改用完整指南第 12 节的隔离命令，避免自动切换环境。

```bash
bash scripts/robodojo.sh doctor --skip-conda --skip-isaac --skip-policy
```

期望：`fail=0`；跳过项的 WARN 正常。这里跳过 Isaac 是因为 doctor 仍依赖 conda run，而第 6 节已单独验证。

最后按所选 XPolicyLab 模型准备**独立策略环境和权重**，再跑一个任务。具体参数见完整指南第 13 节。真实相机图像、CuRobo 规划及完整评测到该阶段才能验证。

重开终端按完整指南第 16 节恢复变量。离开项目时清除 `UV_CONSTRAINT`、`PIP_CONSTRAINT`，避免约束影响其他环境。保留 constraints.txt、overrides.txt 和运行日志；它们尚不是完整 uv.lock。
