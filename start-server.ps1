# 启动 MewHelp 服务。用法：在仓库根目录执行  pwsh -File .\start-server.ps1
# 关闭窗口或按 Ctrl+C 即停止。
$ErrorActionPreference = 'Stop'

Set-Location -Path $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'src'

Write-Host '正在启动 MewHelp 服务……' -ForegroundColor Cyan
Write-Host '界面地址： http://127.0.0.1:8000/' -ForegroundColor Green
Write-Host '接口文档： http://127.0.0.1:8000/docs' -ForegroundColor Green
Write-Host '按 Ctrl+C 停止。' -ForegroundColor DarkGray

python -m mewhelp.cli serve
