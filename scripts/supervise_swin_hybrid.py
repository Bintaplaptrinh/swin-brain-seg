"""Own-process supervisor: safe resource recovery and durable monitoring state."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from src.core.paths import configure_run_paths


ROOT = Path(__file__).resolve().parents[1]


def acquire_lock(control):
    stream = (control/"supervisor.lock").open("a+b")
    if stream.seek(0, 2) == 0:
        stream.write(b"\0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError("Another supervisor owns this run; refusing a duplicate")
    return stream


def save(path, data):
    data = {"updated_at": datetime.now().astimezone().isoformat(), **data}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    for attempt in range(5):
        try:
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(.2)


def stop_owned_process(child):
    if child.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
        else:
            child.terminate()
        child.wait(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control", type=Path, default=ROOT/"outputs/logs/swin_hybrid_supervisor")
    parser.add_argument("--calibration", type=Path, default=ROOT/"outputs/logs/calibration/swin_hybrid_512_v2.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume-supervisor", action="store_true")
    args = parser.parse_args()
    args.resume, args.smoke_test = None, False
    configure_run_paths(args, "swin_hybrid")
    args.output, args.control = args.output.resolve(), args.control.resolve()
    if args.output.exists() and not args.resume_supervisor:
        raise FileExistsError("Run directory already exists; refusing to start a duplicate experiment")
    # Never reuse the obsolete 256/frozen/batch-16 calibration for this protocol.
    calibration = json.loads(args.calibration.read_text())
    if (calibration.get("protocol") != "swin_hybrid_512_unfrozen_v2"
            or calibration["size"] != 512 or calibration["peak_reserved_GB"] > 9):
        raise ValueError("A CUDA calibration for the 512/unfrozen v2 protocol is required")
    args.control.mkdir(parents=True, exist_ok=True)
    run_lock = acquire_lock(args.control)
    if (args.control/"STOP_REQUESTED").exists():
        raise RuntimeError("STOP_REQUESTED exists; refusing to restart a deliberately stopped run")
    # Reversible, process-scoped keep-awake: the screen may still turn off.
    # Windows clears this request automatically when this supervisor exits.
    if os.name == "nt":
        import ctypes
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError("Could not request system wakefulness during supervised training")
    batch, workers = calibration["batch_size"], args.workers
    if args.resume_supervisor and (args.control/"supervisor.json").exists():
        previous = json.loads((args.control/"supervisor.json").read_text())
        batch, workers = previous["batch_size"], previous["workers"]
    plan = {"supervisor_pid": os.getpid(), "run_root": str(args.output), "python": sys.executable,
            "batch_size": batch, "effective_batch": 16, "workers": workers, "vram_budget_GB": 9,
            "protocol": "swin_hybrid_512_unfrozen_v2", "size": 512,
            "epochs": 300, "ema_start": 20, "early_start": 100, "patience": 50,
            "parameter_counts": {k: calibration[k] for k in ("total", "trainable", "frozen")}}
    save(args.control/"plan.json", plan)
    first_attempt = len(list(args.control.glob("attempt_*.log")))
    for offset in range(5):
        attempt = first_attempt + offset
        command = [sys.executable, "-u", "-m", "src.trainers.swin_hybrid", "--output", str(args.output),
                   "--batch-size", str(batch), "--effective-batch", "16", "--workers", str(workers), "--max-vram-gb", "9",
                   "--size", "512", "--ema-start", "20", "--early-start", "100", "--patience", "50",
                   "--glioma-sampling-weight", "2", "--pituitary-sampling-weight", "1.3", "--swin-lr-scale", ".3"]
        if (args.checkpoint_dir/"last.pt").exists():
            command += ["--resume", str(args.checkpoint_dir/"last.pt")]
        elif args.output.exists():
            command += ["--retry-empty-run"]
        log_path = args.control/f"attempt_{attempt:02d}.log"
        status = {**plan, "attempt": attempt, "batch_size": batch, "workers": workers, "command": command,
                  "log": str(log_path), "phase": "starting"}
        save(args.control/"supervisor.json", status)
        with log_path.open("w", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            status.update(child_pid=child.pid, phase="running")
            save(args.control/"supervisor.json", status)
            last_activity = time.time()
            last_size = 0
            stalled = False
            while child.poll() is None:
                if (args.control/"STOP_REQUESTED").exists():
                    stop_owned_process(child)
                    save(args.control/"supervisor.json", {**status, "phase": "stopped_by_user"})
                    return
                size = log_path.stat().st_size
                if size != last_size:
                    last_activity, last_size = time.time(), size
                if time.time() - last_activity > 900:
                    stalled = True
                    stop_owned_process(child)
                    break
                status["last_log_activity_seconds_ago"] = time.time()-last_activity
                save(args.control/"supervisor.json", status)
                time.sleep(10)
        code = child.returncode
        if code == 0 and (args.output/"test_metrics.json").exists():
            result = json.loads((args.output/"test_metrics.json").read_text())
            save(args.control/"supervisor.json", {**status, "phase": "completed", "exit_code": code,
                 "completed_epoch": result["completed_epoch"], "stop_reason": result["stop_reason"],
                 "test_glioma_mIoU": result["per_class"]["Glioma"]["mIoU"],
                 "summary": str(args.output/"summary.txt")})
            return
        text = log_path.read_text(encoding="utf-8", errors="replace").lower()
        reason = None
        if "out of memory" in text or ("gb budget" in text and "exceeds" in text):
            if batch > 1:
                batch //= 2
                reason = "reduce physical batch after CUDA memory/budget failure; effective batch remains 16"
        elif stalled or any(s in text for s in ("dataloader worker", "paging file", "winerror 1455", "brokenpipe", "can't start new thread")):
            if workers > 0:
                workers //= 2
                reason = "reduce data workers after worker failure or 15-minute stall"
        event = {"attempt": attempt, "exit_code": code, "reason": reason, "next_batch_size": batch,
                 "next_workers": workers, "log": str(log_path), "time": datetime.now().astimezone().isoformat()}
        with (args.control/"resource_events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event)+"\n")
        if reason is None:
            save(args.control/"supervisor.json", {**status, "phase": "failed", "exit_code": code,
                 "error": "Nonrecoverable failure; inspect the attempt log and training status"})
            raise SystemExit(1)
        save(args.control/"supervisor.json", {**status, "phase": "recovering", **event,
                                              "batch_size": batch, "workers": workers})
        time.sleep(5)
    save(args.control/"supervisor.json", {**status, "phase": "failed", "error": "Resource recovery attempt limit reached"})
    raise SystemExit(1)


if __name__ == "__main__":
    main()
