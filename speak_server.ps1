# Persistent TTS server with best available voice selection
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

# Pick the best available voice (priority order)
$preferred = @("Mark", "Zira", "David", "Hazel", "George")
$installed = $synth.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }

$chosen = $null
foreach ($pref in $preferred) {
    $match = $installed | Where-Object { $_ -match $pref } | Select-Object -First 1
    if ($match) {
        $chosen = $match
        break
    }
}

if ($chosen) {
    $synth.SelectVoice($chosen)
    Write-Host "VOICE: $chosen"
} else {
    Write-Host "VOICE: default"
}

# Speaking rate: -2 to 10 (0 = normal, negative = slower/clearer)
$synth.Rate   = -1   # slightly slower = clearer pronunciation
$synth.Volume = 100  # maximum volume

# Signal ready
Write-Host "READY"
[Console]::Out.Flush()

while ($true) {
    $line = [Console]::ReadLine()
    if ($null -eq $line) { break }
    $line = $line.Trim()
    if ($line -eq "EXIT") { break }
    if ($line.Length -gt 0) {
        $synth.Speak($line)
    }
}
