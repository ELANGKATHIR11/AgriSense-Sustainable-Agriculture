Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd.exe /c cd /d f:\agrisense-a-smart-agriculture-solution-for-sustainable-farming && set PATH=F:\FULL-STACK;%PATH% && npm run electron-dev", 0, False
