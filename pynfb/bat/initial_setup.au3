

;Run("C:\Program Files\NOITOM\Axis Neuron\Axis Neuron x64.exe")

If NOT WinExists("Axis Neuron") Then
   Run("C:\Program Files\NOITOM\Axis Neuron\Axis Neuron x64.exe")

   $axisneuron="Axis Neuron"
   ;WinActivate($axisneuron)
   WinWait($axisneuron)
   WinMove($axisneuron,"",725,550,750,320)

   ;WinWait($axisneuron)

   MouseClick("left", 1342, 665, 1)

EndIf

$axisneuron="Axis Neuron"
WinWait($axisneuron)

;WinActivate($axisneuron)
WinMove($axisneuron,"",725,550,750,320)

;WinWait($axisneuron)

If NOT WinExists("Link Perception Neuron") Then
   Run("C:\nfb\pynfb\bat\1_link_PerceptionNeuron.bat")
EndIf

$linkpnname="Link Perception Neuron"
WinWait($linkpnname)
;WinActivate($linkpnname)
WinMove($linkpnname,"",1090,0,380,160)

;WinWait($linkpnname)

If NOT WinExists("EB Neuro Open Stram") Then
   Run("C:\Program Files (x86)\EB Neuro Open Stream\EBNeuro BePlusLTM Amplifier.exe")
   $ebname="EB Neuro Open Stram"
   WinWait($ebname)
   ;WinActivate($ebname)
   WinMove($ebname,"",1090,150,380,400)
   ControlClick($ebname,"","[NAME:_cbHeadCap]")
   ControlClick($ebname,"","[NAME:_cbSamplingRate]")
   Send("{down 4}")
   Send("{enter}")
   ControlSetText($ebname,"","[NAME:_tbIpAddress]","192.168.171.81")
   ControlSetText($ebname,"","[CLASS:WindowsForms10.EDIT.app.0.141b42a_r16_ad1; INSTANCE:1]","68")
   ControlClick($ebname,"","[NAME:_btnLink]")
EndIf






Exit