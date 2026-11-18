"""Verify matched starts/configs/physics, then summarize the offset intervention."""
from pathlib import Path
import json
import statistics

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results/strict'
CASES = ('yibo','yibo_with_our_offsets')
ALLOWED = {'rigid_cloth.rest_offset','rigid_cloth.contact_offset','rigid_cloth.randomization.yibo_rest_offset'}


def differences(a,b,prefix=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return {k for key in a.keys()|b.keys() for k in differences(a.get(key),b.get(key),f'{prefix}.{key}' if prefix else key)}
    return {prefix} if a != b else set()


def read(protocol,case,seed,suffix=''):
    stem=f'{case}{"_no_checks" if protocol=="without_checks" else ""}_seed{seed}'
    directory = ROOT/'results/offset_aware' if protocol=='recipe_checks' else RESULTS
    return json.loads((directory/f'{stem}{suffix}.json').read_text())


def main():
    report={'checkpoint':'pretrained_policy/flexiv_wuji_finetuned/flat7RW_fricition.pth',
            'episodes_per_case':768,'seeds':[0,1,2],'environments_per_seed':256,
            'fold_tolerance_m':.015,'max_steps':600,'protocols':{}}
    for protocol in ('recipe_checks','without_checks'):
        data={case:[] for case in CASES}
        for seed in report['seeds']:
            a,b=[read(protocol,case,seed) for case in CASES]
            configs=[read(protocol,case,seed,'.config') for case in CASES]
            changed=differences(*configs)
            assert changed==ALLOWED,(protocol,seed,changed)
            assert a['initial']==b['initial'],(protocol,seed,'unmatched initial state')
            for one,two in zip(a['startup_physics_readback']['cloth'],b['startup_physics_readback']['cloth']):
                for field in ('env_ids','design','stiffness','damping','armature','material','mass_kg'):
                    assert one[field]==two[field],(protocol,seed,field)
                assert two['all_rest_offset_range']==[0.,0.]
                assert max(abs(x-.002) for x in two['all_contact_offset_range'])<1e-7
                assert min(one['all_rest_offset_range'])>=.004-1e-7
                assert max(one['all_rest_offset_range'])<=.010+1e-7
            for asset in ('robot','table'):
                assert a['startup_physics_readback'][asset]==b['startup_physics_readback'][asset]
            for case,d in zip(CASES,(a,b)):
                assert d['all']['n']==256 and d['all']['completed']==256
                assert d['fold_tolerance_m']==.015
                assert not d['all']['termination_reasons']['nonfinite']
                data[case].append(d)
        summaries={}
        for case,runs in data.items():
            keys=('n','completed','fold_entries','valid_fold_entries','full_releases','guard_ready_envs')
            counts={key:sum(d['all'][key] for d in runs) for key in keys}
            counts['fold_rate']=counts['fold_entries']/counts['n']
            counts['per_seed']=[dict(seed=d['seed'],**{k:d['all'][k] for k in keys}) for d in runs]
            counts['termination_reasons']={reason:sum(d['all']['termination_reasons'].get(reason,0) for d in runs)
                for reason in sorted(set().union(*(d['all']['termination_reasons'] for d in runs)))}
            counts['median_best_fold_error_m']=statistics.median(x for d in runs for x in d['per_env']['best_error'])
            counts['by_design']={design:{k:sum(d['by_design'][design][k] for d in runs) for k in keys}
                                 for design in runs[0]['by_design']}
            summaries[case]=counts
        left=[x for d in data[CASES[0]] for x in d['per_env']['folded']]
        right=[x for d in data[CASES[1]] for x in d['per_env']['folded']]
        summaries['paired']={'both_fold':sum(a and b for a,b in zip(left,right)),
                             'yibo_only':sum(a and not b for a,b in zip(left,right)),
                             'our_offsets_only':sum(b and not a for a,b in zip(left,right)),
                             'neither':sum(not(a or b) for a,b in zip(left,right))}
        report['protocols'][protocol]=summaries
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
