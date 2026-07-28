# 推送 GitHub（使用 F:\Git_Projects\github 推送用 token.txt）
# 用法：在 PowerShell 中执行
#   .\scripts\push_github.ps1

$ErrorActionPreference = "Stop"
$env:Path = "C:\Program Files\GitHub CLI;" + $env:Path
Set-Location $PSScriptRoot\..

$tokenFile = "F:\Git_Projects\github 推送用 token.txt"
if (-not (Test-Path $tokenFile)) {
  Write-Error "找不到 token 文件: $tokenFile"
}
$env:GH_TOKEN = (Get-Content $tokenFile -Raw).Trim()
$env:GITHUB_TOKEN = $env:GH_TOKEN

$name = "write-protected-discrete-bottleneck"
$owner = & gh api user -q .login
Write-Host "账号: $owner"

$exists = $false
try {
  & gh repo view "$owner/$name" 2>$null | Out-Null
  if ($LASTEXITCODE -eq 0) { $exists = $true }
} catch {
  $exists = $false
}

if (-not $exists) {
  Write-Host "创建公开仓库 $owner/$name ..."
  & gh repo create $name --public --source=. --remote=origin --push
} else {
  $remote = git remote get-url origin 2>$null
  if (-not $remote) {
    git remote add origin "https://github.com/$owner/$name.git"
  }
  git push -u origin HEAD
}

Write-Host "完成: https://github.com/$owner/$name"
