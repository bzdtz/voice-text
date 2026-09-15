Option Explicit

Dim shell, fileSystem, scriptFolder
Set shell = CreateObject("WScript.Shell")
Set fileSystem = CreateObject("Scripting.FileSystemObject")
scriptFolder = fileSystem.GetParentFolderName(WScript.ScriptFullName)

' The shortcut may be started from any working directory.
shell.CurrentDirectory = scriptFolder
shell.Run "cmd.exe /c run.bat", 0, False
