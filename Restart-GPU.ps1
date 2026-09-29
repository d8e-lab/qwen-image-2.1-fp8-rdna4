# Restart the AMD display adapter to clear the leaked WSL GPU pool.
#
# WHY: failed hipModuleLoad calls (hipBLASLt's gfx1201 Tensile kernels cannot load under
# WSL2/librocDXG) leak the dxg GPU host-memory pool. Once exhausted, no allocation works
# from inside WSL -- not even 8 bytes -- and the state survives a WSL restart because it
# lives on the Windows graphics driver. Disabling and re-enabling the adapter resets it.
#
# HOW TO RUN:
#   1. Right-click this file -> "Run with PowerShell" as Administrator, OR
#   2. Open PowerShell as Administrator and run:
#        powershell -ExecutionPolicy Bypass -File Restart-GPU.ps1
#
# The screen will blank for a few seconds and windows may move -- expected.

#Requires -RunAsAdministrator

$ErrorActionPreference = 'Stop'

Write-Host "=== AMD display adapters ===" -ForegroundColor Cyan
$gpus = Get-PnpDevice -Class Display | Where-Object { $_.FriendlyName -match 'Radeon|AMD' }
if (-not $gpus) {
    Write-Host "No AMD display adapter found. Aborting." -ForegroundColor Red
    exit 1
}
$gpus | Select-Object Status, FriendlyName, InstanceId | Format-Table -AutoSize

foreach ($gpu in $gpus) {
    Write-Host "`nDisabling: $($gpu.FriendlyName)" -ForegroundColor Yellow
    Disable-PnpDevice -InstanceId $gpu.InstanceId -Confirm:$false
    Start-Sleep -Seconds 5

    Write-Host "Enabling:  $($gpu.FriendlyName)" -ForegroundColor Yellow
    Enable-PnpDevice -InstanceId $gpu.InstanceId -Confirm:$false
    Start-Sleep -Seconds 5

    $state = (Get-PnpDevice -InstanceId $gpu.InstanceId).Status
    Write-Host "Status now: $state" -ForegroundColor $(if ($state -eq 'OK') { 'Green' } else { 'Red' })
}

Write-Host "`n=== Done. Now check from WSL: ===" -ForegroundColor Cyan
Write-Host "  cd ~/workspace/qwen-image"
Write-Host "  source scripts/env.sh && qip_gpu_ok"
Write-Host "`nExpected: 'GPU OK - allocations work'"
