_prepend_python_nvjitlink_to_ld_path() {
  local py_purelib py_bin="${PYTHON_BIN:-${PYTHON:-python}}"
  if ! py_purelib="$("$py_bin" -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null)"; then
    return 0
  fi
  local nvjit="${py_purelib}/nvidia/nvjitlink/lib"
  if [ -f "${nvjit}/libnvJitLink.so.12" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${nvjit}:"*) ;;
      *) export LD_LIBRARY_PATH="${nvjit}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
}
