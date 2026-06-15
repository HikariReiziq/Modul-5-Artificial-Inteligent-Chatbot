import json

with open('notebooks/03_evaluation_metrics.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 9: change max_retries
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'invoke_llm_with_retry' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        src = src.replace('stop_after_attempt(5)', 'stop_after_attempt(10)')
        src = src.replace('multiplier=1, min=4, max=60', 'multiplier=2, min=10, max=120')
        cell['source'] = [line + '\n' for line in src.split('\n')][:-1]

# Cell 11: change time.sleep(2) to time.sleep(15)
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'time.sleep(2)' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        src = src.replace('time.sleep(2)', 'time.sleep(15)')
        cell['source'] = [line + '\n' for line in src.split('\n')][:-1]

# Cell 13: import RunConfig
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'from ragas.llms import LangchainLLMWrapper' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        if 'RunConfig' not in src:
            src += "\nfrom ragas.run_config import RunConfig\n"
        cell['source'] = [line + '\n' for line in src.split('\n')][:-1]

# Cell 15: update evaluate to use run_config
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'ragas_results = evaluate(' in ''.join(cell['source']):
        src = ''.join(cell['source'])
        if 'run_config' not in src:
            src = src.replace('llm=evaluator_llm', 'llm=evaluator_llm,\n    run_config=RunConfig(max_workers=1, max_wait=120, max_retries=15)')
        cell['source'] = [line + '\n' for line in src.split('\n')][:-1]

with open('notebooks/03_evaluation_metrics.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
