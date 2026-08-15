@echo off
rem == Smoking-detection GUI launcher ==
rem
rem The window has two tabs along the BOTTOM edge, Excel style:
rem
rem     [ Live detection ]  detect / track / alarm / second-stage review
rem     [ Recording ]       save a LIVE stream to disk, no decoding, no lost frames
rem     [ Download ]        save a VIDEO to disk via yt-dlp
rem
rem All three can run at once: recording and downloading never decode, so they
rem use no GPU and do not compete with detection for CPU.
rem
rem Recording vs Download -- the only difference is whether the source ENDS.
rem A live stream has no end, so the recorder treats "nothing to read" as a
rem dropped connection and reconnects. A video does end, and that assumption
rem turns into re-downloading the same file forever. So: live -> Recording,
rem video -> Download. The Download tab refuses live URLs on purpose.
rem
rem ---- Live detection tab ----
rem
rem The status panel is three columns:
rem     Present      people still MOVING (passing / wandering), with
rem                  ID + behaviour + stage / count / facing-away
rem     Waiting      people who have STOPPED and are not flagged as smoking
rem                  yet -- this is the column that matters. Smoking is done
rem                  standing still; someone walking past is just passing by
rem     Smoking      alarm standing, plus the second-stage review verdict
rem Every tracked person lands in exactly one of Present/Waiting, and also in
rem Smoking once an alarm stands. Columns keep a fixed number of rows, so a
rem flickering detection never makes the list jump around.
rem
rem The detection method is chosen in the "method" dropdown of the control row
rem -- all methods share this one GUI so they can be compared under identical
rem input. Default is the pure-rule method, which needs no weight file at all.
rem
rem Arguments are passed through, so a method can be preselected:
rem     launch.bat --method rule+grammar
rem Valid keys live in inference/methods.py; to see them with their
rem descriptions run:  python -m inference.pipeline --method list
rem
rem The "source" box in the GUI takes any of:
rem     0                                     webcam index
rem     demo_videos\clip.avi                  video file (or the Browse button)
rem     rtsp://admin:@pass@10.0.0.1:554       IP camera (raw password is fine)
rem     https://www.youtube.com/watch?v=...   YouTube live or normal video
rem     https://host/live.m3u8                public HLS stream
rem YouTube needs yt-dlp:  pip install yt-dlp
rem
rem On a YouTube/HLS source the video runs about 5 s behind the live edge ON
rem PURPOSE (configs/*.yaml -> stream.prefill_sec). HLS delivers one segment at
rem a time and then goes quiet; buffering that much and playing it back at 1x
rem is what keeps the picture smooth. Raise it for a steadier picture, at the
rem cost of the same amount of extra delay. This is not a bug.
rem
rem The status line right of the buttons reports what is actually happening,
rem in three numbers (labels are on screen; ASCII kept here on purpose):
rem     1st - inference steps per second, target 10
rem     2nd - frames still buffered; sitting at 0 means the source is starving
rem     3rd - stream seconds consumed per wall second; below 1 = losing ground
rem
rem ---- Recording tab ----
rem
rem Stores the stream exactly as the server sent it (ffmpeg -c copy), so no
rem frame can be lost and the CPU cost is near zero. One folder per stream URL,
rem one subfolder per day, oldest days deleted past the retention setting.
rem
rem SPACE: 720p measures ~21 MB/min, about 30 GB PER DAY -- 3 days needs ~90 GB.
rem The tab shows free space before you start and turns red when it is short.
rem Point the folder box at a roomy drive; the default is under the project.
rem "Check retention" lists what the retention setting would delete, without
rem deleting anything.
rem
rem The same thing from the command line:
rem     python scripts/record_stream.py <url> --root E:/recordings
rem
rem ---- Download tab ----
rem
rem Paste a URL, hit "query info" to see title / length / size, then download.
rem Saved as "<title> [<video id>].mp4" -- the id keeps same-titled reuploads
rem from overwriting each other. Cancellable; resumes after an interruption.
rem
rem 720p and above needs audio+video muxing (modern YouTube only ships 360p
rem pre-muxed; higher is DASH with separate streams). The ffmpeg used for that
rem is the one bundled with imageio-ffmpeg, so nothing extra to install.
rem
rem The lower half is not a text log -- it is the folder's video list, with a
rem thumbnail, length and size per file. Double-click (or hit Play) to open it
rem in the player; the list refreshes itself when a download finishes.
rem Messages and errors go to the status line next to the buttons.
rem
rem ---- Player (opens from the video list, or from an alarm entry) ----
rem
rem Built for LABELLING, not for watching. No audio on purpose.
rem     play/pause, draggable seek bar, clock
rem     frame step and 0.25x-2x, for checking the raise/hold/lower turns
rem     skeleton overlay, toggleable -- see what the system sees
rem     "scan" runs the detector over the whole clip and marks every alarm
rem       on the timeline in red; the arrow buttons jump between them
rem     manual marks in yellow; snapshot to jpg with the frame number
rem Keys: space, left/right = 5s, comma/period = one frame, N/P = next or
rem previous mark, S = snapshot, M = mark, F = fullscreen, Esc = leave.
rem The scan uses whichever method the detection tab has selected.
rem
rem Downloaded files feed straight back in: pick one as the "source" on the
rem detection tab.
rem
rem NOTE: keep this file ASCII-only. cmd.exe mangles UTF-8 text under the
rem default codepage, which turns comments and echo output into garbage.

rem cd to this bat file's folder (works even if project moves)
cd /d "%~dp0"

rem Prevent OpenMP duplicate-lib crash (OMP Error #15) on this machine
set KMP_DUPLICATE_LIB_OK=TRUE

rem Interpreter for this machine; fall back to PATH if conda moved or the
rem project was copied to another box
rem No labels/goto in this file on purpose: it has LF-only line endings, and
rem cmd.exe seeks labels by byte offset, which misbehaves without CRLF.
set PY=D:\conda\python.exe
set PYOK=1
if not exist "%PY%" (
    set PY=python
    where python >nul 2>nul || set PYOK=0
)

rem Say plainly that Python is missing. Without this check cmd.exe only prints
rem "'python' is not recognized...", which reads like the project is broken
rem rather than like the interpreter path needs fixing.
if "%PYOK%"=="0" (
    echo.
    echo ===== Python not found. =====
    echo Tried: D:\conda\python.exe  and  "python" on PATH.
    echo Edit the "set PY=" line in this file to point at your interpreter.
    echo.
    pause
    exit /b 1
)

rem Send stderr to a log file instead of the console.
rem
rem Reason: when the source is a YouTube/HLS stream, FFmpeg floods stderr with
rem     [https @ ...] Cannot reuse HTTP connection for different host: ...
rem once per CDN host change. It is harmless (FFmpeg reconnects and playback is
rem fine) but it buries everything else. It is emitted from C code writing to
rem fd 2, so it cannot be filtered from inside Python -- OPENCV_FFMPEG_LOGLEVEL
rem does not work on OpenCV 5.0.0 (measured).
rem
rem Nothing is lost: the app's own messages go to stdout and still show here,
rem and the full stderr (including real tracebacks) is kept in the log.
if not exist "logs" mkdir "logs"
set LOG=logs\gui_stderr.log

rem Rotate at 5 MB, keeping one previous file. The log is append-only across
rem runs and the FFmpeg chatter above is verbose, so without this a few long
rem YouTube sessions grow it into the hundreds of MB.
set LOGMAX=5000000
set SIZE=0
if exist "%LOG%" for %%A in ("%LOG%") do set SIZE=%%~zA
if %SIZE% GTR %LOGMAX% (
    if exist "%LOG%.old" del "%LOG%.old"
    move /y "%LOG%" "%LOG%.old" >nul
    echo [launcher] previous log rotated to %LOG%.old
)

rem %DATE% carries a localized weekday ("2026/08/13" is prefixed by a Chinese
rem weekday here), which turns into mojibake under the console codepage. Take
rem the trailing 10 chars to keep just the yyyy/mm/dd part, which is ASCII.
echo === run %DATE:~-10% %TIME% ===>> "%LOG%"

"%PY%" scripts/gui.py %* 2>> "%LOG%"

rem Keep window open if it crashed. The error went to the log, so show the
rem tail of it -- otherwise "see message above" would point at nothing.
if errorlevel 1 (
    echo.
    echo ===== Program exited with an error. Last lines of %LOG%: =====
    powershell -NoProfile -Command "Get-Content -Tail 25 '%LOG%'"
    echo ===============================================================
    pause
)
