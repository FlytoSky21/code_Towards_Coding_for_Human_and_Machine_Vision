::for /l %%i in (0 1 5) do (
::	autotrace.exe ..\data\edges\%%i.bmp --output-file ..\data\edges\%%i.svg --centerline --color-count 2
::)
::set  /a startS=%time:~6,5%
::set  /a startM=%time:~3,2%
echo %time%
set start_time=%time%
setlocal enabledelayedexpansion
for /r D:\VGGFace2\test_all_128\edges\  %%i in (*.bmp) do (
    set bmpName=%%i
    set svgName=!bmpName:bmp=svg!
    autotrace.exe !bmpName! --output-file !svgName! --centerline --color-count 2
)
::set  /a endS=%time:~6,5%
::set  /a endM=%time:~3,2%
echo %start_time%
echo %time%
::set  /a diffS_=%endS%-%startS%
::set  /a diffM_=%endM%-%startM%
::echo cost:%diffM_%  %diffS_%
pause