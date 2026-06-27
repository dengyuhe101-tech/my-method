from pathlib import Path
import argparse
from datetime import datetime
import os
import re
import site
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CIANNA_DIR = ROOT_DIR / "src"


def _run_sort_key(path):
	match = re.fullmatch(r"run(\d+)", path.name)
	if match:
		return int(match.group(1))
	return -1


def _existing_runs():
	return sorted(
		[p for p in SCRIPT_DIR.iterdir() if p.is_dir() and re.fullmatch(r"run\d+", p.name)],
		key=_run_sort_key,
	)


def _next_run_dir():
	runs = _existing_runs()
	next_id = _run_sort_key(runs[-1]) + 1 if runs else 1
	return SCRIPT_DIR / ("run%d" % next_id)


def _latest_run_dir():
	runs = _existing_runs()
	return runs[-1] if runs else None


def _resolve_run_dir(run_name):
	run_dir = Path(run_name)
	if not run_dir.is_absolute():
		run_dir = SCRIPT_DIR / run_dir
	return run_dir


def activate_run(argv, mode):
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--run", dest="run_name")
	parser.add_argument("--new-run", action="store_true")
	args, remaining = parser.parse_known_args(argv)

	if args.run_name is not None:
		run_dir = _resolve_run_dir(args.run_name)
	elif mode == "train" and remaining and remaining[0].isdigit() and int(remaining[0]) > 0 and not args.new_run:
		run_dir = _latest_run_dir()
		if run_dir is None:
			run_dir = _next_run_dir()
	elif args.new_run or mode == "train":
		run_dir = _next_run_dir()
	else:
		run_dir = _latest_run_dir()
		if run_dir is None:
			run_dir = _next_run_dir()

	run_dir.mkdir(parents=True, exist_ok=True)
	(run_dir / "net_save").mkdir(exist_ok=True)
	(run_dir / "fwd_res").mkdir(exist_ok=True)
	os.chdir(run_dir)
	with open(run_dir / "run_info.txt", "a") as f:
		f.write("[%s] mode=%s argv=%s\n" % (datetime.now().isoformat(timespec="seconds"), mode, " ".join(sys.argv)))
	print("Using run directory: %s" % run_dir)
	return run_dir, remaining


def add_cianna_build_to_path():
	build_libs = []
	for build_root in (CIANNA_DIR / "src" / "build", CIANNA_DIR / "build"):
		for lib_dir in build_root.glob("lib.*"):
			if (lib_dir / "CIANNA.so").is_file():
				build_libs.append(lib_dir)
	if build_libs:
		build_libs = sorted(build_libs, key=lambda p: (p / "CIANNA.so").stat().st_mtime)
		build_path = str(build_libs[-1])
		sys.path.insert(0, build_path)
		return build_path
	return None


def sanitize_user_site():
	os.environ["PYTHONNOUSERSITE"] = "1"
	user_site_paths = []
	try:
		site_paths = site.getusersitepackages()
	except Exception:
		site_paths = []
	if isinstance(site_paths, str):
		site_paths = [site_paths]
	for path in site_paths:
		if path:
			user_site_paths.append(Path(path).resolve())
	user_base = os.environ.get("PYTHONUSERBASE")
	if user_base:
		user_site_paths.append(Path(user_base).expanduser().resolve())

	if not user_site_paths:
		return []

	filtered_sys_path = []
	removed_paths = []
	for path_entry in sys.path:
		if not path_entry:
			filtered_sys_path.append(path_entry)
			continue
		try:
			resolved_entry = Path(path_entry).expanduser().resolve()
		except OSError:
			filtered_sys_path.append(path_entry)
			continue
		if any(
			resolved_entry == user_path or user_path in resolved_entry.parents
			for user_path in user_site_paths
		):
			removed_paths.append(path_entry)
		else:
			filtered_sys_path.append(path_entry)

	sys.path[:] = filtered_sys_path
	return removed_paths


def configure_runtime_paths():
	removed_paths = sanitize_user_site()
	if removed_paths:
		print("Removed user-site paths from sys.path: %s" % ", ".join(removed_paths), flush=True)

	use_env_package = os.environ.get("CIANNA_USE_ENV_PACKAGE", "").strip().lower() in ("1", "true", "yes", "on")
	if use_env_package:
		print("CIANNA_USE_ENV_PACKAGE is set; using environment package instead of local build", flush=True)
		return

	build_path = add_cianna_build_to_path()
	if build_path:
		print("Using local CIANNA build path: %s" % build_path, flush=True)
		return

	raise RuntimeError(
		"No local CIANNA build found under %s/build/lib.*/CIANNA.so or "
		"%s/src/build/lib.*/CIANNA.so. Rebuild CIANNA from this repository before running."
		% (CIANNA_DIR, CIANNA_DIR)
	)
