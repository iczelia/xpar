@ECHO OFF
REM   Copyright (C) 2022-2026 Kamila Szewczyk
REM
REM   This program is free software; you can redistribute it and/or modify
REM   it under the terms of the GNU General Public License as published by
REM   the Free Software Foundation; version 3 of the License only.
REM
REM   This program is distributed in the hope that it will be useful,
REM   but WITHOUT ANY WARRANTY; without even the implied warranty of
REM   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
REM   GNU General Public License for more details.
REM
REM   You should have received a copy of the GNU General Public License
REM   along with this program. If not, see <http://www.gnu.org/licenses/>.

REM  D: holds the corpus; E: collects logs and completion tokens.
REM  RUN2 merges stderr into stdout for COMMAND.COM.

D:
CD \

REM  Sanity check.
RUN2.EXE XPAR.EXE --version > E:\VERSION.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO version >> E:\RESULT.TXT

REM  Read a host-generated set.
RUN2.EXE XPAR.EXE verify SIDE.XPA > E:\VERIFY1.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO verify-clean >> E:\RESULT.TXT

RUN2.EXE XPAR.EXE verify --strong SIDE.XPA > E:\STRONG.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO verify-strong >> E:\RESULT.TXT

RUN2.EXE XPAR.EXE info SIDE.XPA > E:\INFO.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO info >> E:\RESULT.TXT

RUN2.EXE XPAR.EXE list SIDE.XPA > E:\LIST.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO list >> E:\RESULT.TXT

RUN2.EXE XPAR.EXE explain SIDE.XPA > E:\EXPLAIN.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO explain >> E:\RESULT.TXT

REM  Detect damage.
COPY /Y BAD.BIN BIG.BIN > NUL
RUN2.EXE XPAR.EXE verify SIDE.XPA > E:\VERIFY2.TXT
IF ERRORLEVEL 2 GOTO DONE
IF NOT ERRORLEVEL 1 GOTO DONE
ECHO verify-damaged >> E:\RESULT.TXT

REM  Repair with scalar kernels.
RUN2.EXE XPAR.EXE repair --in-place SIDE.XPA > E:\REPAIR.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO repair >> E:\RESULT.TXT

RUN2.EXE XPAR.EXE verify --strong SIDE.XPA > E:\VERIFY3.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO verify-repaired >> E:\RESULT.TXT

REM  Write a set for the host to read. Percent signs are special in batch files.
RUN2.EXE XPAR.EXE create -r 6 -o DOSMADE MADE.BIN > E:\CREATE.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO create >> E:\RESULT.TXT

RUN2.EXE XPAR.EXE verify --strong DOSMADE.XPA > E:\VERIFY4.TXT
IF ERRORLEVEL 1 GOTO DONE
ECHO verify-own >> E:\RESULT.TXT

REM  Reject an unrepairable set.
RUN2.EXE XPAR.EXE verify DOOMED.XPA > E:\DOOMED.TXT
REM  ERRORLEVEL n is "n or above", so an exact 2 needs both bounds.
IF ERRORLEVEL 3 GOTO DONE
IF NOT ERRORLEVEL 2 GOTO DONE
ECHO verify-unrepairable >> E:\RESULT.TXT

ECHO ALLDONE >> E:\RESULT.TXT
:DONE
