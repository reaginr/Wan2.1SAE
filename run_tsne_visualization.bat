@echo off
chcp 65001 >nul
echo ============================================
echo SAE激活值 t-SNE 可视化
echo ============================================
echo.

:: 设置miniconda环境路径
set CONDA_PATH=D:\miniconda
set ENV_NAME=WanEnv

:: 激活conda环境
call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_PATH%"
call conda activate %ENV_NAME%

echo 当前环境: %ENV_NAME%
python --version
echo.

:: 检查文件夹是否存在
if not exist "activations" (
    echo 错误: 找不到 activations 文件夹
    echo 请确保已从服务器拷贝 activations 和 activations_nsfw 文件夹到本地
    pause
    exit /b 1
)

echo 发现以下激活值文件夹:
if exist "activations" echo   - activations
dir /b activations\sae_layer* 2>nul
echo.
if exist "activations_nsfw" (
    echo   - activations_nsfw
    dir /b activations_nsfw\sae_layer* 2>nul
)
echo.

:: 默认参数
set CATEGORY=sex
set LAYER_KEY=sae_layer15
set OUTPUT_DIR=visualizations

:: 询问用户参数
echo 默认参数:
echo   类别: %CATEGORY%
echo   层: %LAYER_KEY%
echo   输出: %OUTPUT_DIR%
echo.

set /p CUSTOM="是否使用自定义参数? (y/n, 默认n): "
if /i "%CUSTOM%"=="y" (
    set /p CATEGORY="输入类别 (如 sex, violence): "
    set /p LAYER_KEY="输入层 (如 sae_layer15): "
)

echo.
echo ============================================
echo 开始 t-SNE 可视化
echo ============================================
echo   类别: %CATEGORY%
echo   层: %LAYER_KEY%
echo   输出: %OUTPUT_DIR%
echo.

:: 创建输出目录
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: 执行可视化
python wan/sae/interpretability/visualize_tsne.py ^
    --activation_root "activations" ^
    --category %CATEGORY% ^
    --layer_key %LAYER_KEY% ^
    --output_dir "%OUTPUT_DIR%" ^
    --perplexity 30 ^
    --n_iter 1000 ^
    --n_components 2 ^
    --pca_components 100

if errorlevel 1 (
    echo.
    echo 执行失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo ============================================
echo 可视化完成!
echo ============================================
echo 输出文件:
dir /b "%OUTPUT_DIR%\%CATEGORY%_%LAYER_KEY%*"
echo.

:: 如果存在activations_nsfw，也处理它
if exist "activations_nsfw" (
    echo 检测到 activations_nsfw 文件夹，是否也进行可视化?
    set /p RUN_NSFW="(y/n, 默认n): "
    if /i "!RUN_NSFW!"=="y" (
        echo.
        echo 处理 activations_nsfw...
        python wan/sae/interpretability/visualize_tsne.py ^
            --activation_root "activations_nsfw" ^
            --category %CATEGORY% ^
            --layer_key %LAYER_KEY% ^
            --output_dir "%OUTPUT_DIR%" ^
            --perplexity 30 ^
            --n_iter 1000
    )
)

:: 打开输出目录
echo.
echo 是否打开输出目录?
set /p OPEN_DIR="(y/n, 默认y): "
if /i "%OPEN_DIR%"=="y" (
    start explorer "%OUTPUT_DIR%"
)

echo.
echo 按任意键退出...
pause >nul
