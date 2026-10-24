"""Run each checkpoint in a separate process and preserve console output."""
import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
out=root/'results';out.mkdir(exist_ok=True)
models=sys.argv[1:] or ['v1','v2','v3']
if any(m not in ['v1','v2','v3'] for m in models):raise SystemExit('Use v1 v2 v3')
for model in models:
    command=[sys.executable,str(root/'experiment_depth.py'),'--model',model,'--root',str(out)]
    result=subprocess.run(command,capture_output=True,text=True)
    (out/f'{model}_console.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    print(model,'exit',result.returncode,flush=True)
    print(result.stdout[-1800:],flush=True)
    if result.returncode:
        print(result.stderr[-5000:],flush=True);raise SystemExit(result.returncode)
