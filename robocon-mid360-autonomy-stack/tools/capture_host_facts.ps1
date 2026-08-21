param([string]$Output = "")

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path (Get-Location) "host_facts.json"
}

$facts = [ordered]@{
    captured_at_utc = (Get-Date).ToUniversalTime().ToString("o")
    evidence_level = "static_host_inspection"
    computer_name = $env:COMPUTERNAME
    os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber
    cpu = Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
    memory = Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
    disks = Get-CimInstance Win32_LogicalDisk -Filter "DriveType=3" | Select-Object DeviceID, Size, FreeSpace
    network = Get-NetIPConfiguration | Select-Object InterfaceAlias, InterfaceDescription, IPv4Address, IPv6Address
    ros_environment = [ordered]@{
        WSL = (wsl.exe --status 2>$null | Out-String).Trim()
        ROS_DISTRO = $env:ROS_DISTRO
    }
    notes = @(
        "This is a local host inventory, not a sensor or competition acceptance result.",
        "Sensor identity, calibration, bags, and mechanism feedback remain to be measured when connected."
    )
}

$facts | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding UTF8
Write-Output "wrote $Output"

