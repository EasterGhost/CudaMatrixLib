Dim WshShell 
Set WshShell=WScript.CreateObject("WScript.Shell") 
WshShell.Run "powershell.exe"
WScript.Sleep 600
WshShell.SendKeys "ssh LiMuchen@hb.frp.one -p 26562"
WScript.Sleep 200
WshShell.SendKeys "{ENTER}"