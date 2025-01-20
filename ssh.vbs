Dim WshShell 
Set WshShell=WScript.CreateObject("WScript.Shell") 
WshShell.Run "powershell.exe"
WScript.Sleep 540
WshShell.SendKeys "ssh LiMuchen@hb.frp.one -p 26562"
WScript.Sleep 50
WshShell.SendKeys "{ENTER}"