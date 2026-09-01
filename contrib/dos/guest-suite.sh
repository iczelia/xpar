exec > C:/RESULT.LOG 2>&1

export PATH_SEPARATOR=:
export PATH=/dev/c/bin
export TMPDIR=C:/TMP
export TEMP=C:/TMP
export SHELL=C:/BIN/BASH.EXE
export XPAR_SH=C:/BIN/BASH.EXE
export abs_top_builddir=C:/BLD
export abs_top_srcdir=C:/SRC
export srcdir=C:/SRC/TESTS
export XPAR=C:/BLD/XPAR.EXE
export MKDATA=C:/BLD/MKDATA.EXE
export DAMAGE=C:/BLD/DAMAGE.EXE
export FORGE=C:/BLD/FORGE.EXE
export TIMEIT=C:/BLD/TIMEIT.EXE
export XPAR_TEST_CONFIG_H=C:/BLD/CONFIG.H
export XPAR_TEST_CC=false
export CC=false
export XPAR_TEST_LEVEL=@LEVEL@
export XPAR_TEST_SEED=@SEED@
export XPAR_DOS_TEST=1
test ! -d C:/COMPAT || export XPAR_COMPAT=C:/COMPAT

failed=0
for test_program in TUNIT TCODEC TCENTRAL; do
  echo
  echo "===== $test_program.EXE ====="
  C:/BLD/$test_program.EXE || failed=1
done

while read test_script; do
  test -n "$test_script" || continue
  echo
  echo "===== $test_script ====="
  C:/BIN/BASH.EXE C:/SRC/TESTS/$test_script
  status=$?
  case $status in
    0) ;;
    77) echo "$test_script skipped on DOS" ;;
    *) failed=1; echo "$test_script failed with status $status" ;;
  esac
done < C:/TESTS.LST

echo "$failed" > C:/STATUS.TXT
sync
exit "$failed"
