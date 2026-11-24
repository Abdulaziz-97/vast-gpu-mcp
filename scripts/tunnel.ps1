# Run on YOUR Windows computer. Keeps localhost:8765 forwarded to the server on the GPU
# machine, reconnecting automatically and alternating between the routes you give.
#
#   .\tunnel.ps1 root@1.2.3.4:40022 root@ssh5.vast.ai:27389
param([Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)][string[]]$Routes,
      [int]$LocalPort = 8765)

$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"   # works with the Windows ssh-agent
$i = 0
while ($true) {
    $route = $Routes[$i % $Routes.Count]
    $target, $port = $route -split ':(?=\d+$)'
    Write-Host "$(Get-Date -Format s) connecting via $target port $port"
    & $ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 `
           -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new `
           -N -p $port -L "${LocalPort}:127.0.0.1:8765" $target
    Write-Host "$(Get-Date -Format s) tunnel closed; retrying in 5s"
    $i++
    Start-Sleep 5
}
