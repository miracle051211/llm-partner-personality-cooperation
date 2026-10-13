"""Offline validation; stdlib + pandas only. Does not import archived study code."""
from pathlib import Path
import hashlib,json,math,sys
import pandas as pd
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
KEYS=['participant_id','treatment','round']
CATS=['EXP_COOP','EXP_UNCOOP','EXP_UNCLEAR','EXP_NONE']
TREAT={'A':'A','B':'O','C':'C','D':'N','E':'E'}
RUNS={
'DeepSeek':('formal_study1_98_v1_1_20261002_172630','11_FORMAL_STAGE2_RAW','08_FORMAL_SESSION_RANKS.csv','formal_study1_98_analysis_v1/08_EXPECTATION_FINAL.csv'),
'Qwen':('qwen98_cross_model_formal_v1_20261007_115200','QWEN98_STAGE2_RAW','08_QWEN_RANKS.csv','analysis_final/QWEN_EXPECTATION_FINAL.csv'),
'GPT-5.6 Terra':('gpt56terra98_cross_model_formal_v1_20261010_220000','GPT56TERRA_STAGE2_RAW','GPT56TERRA_WITHIN_SESSION_RANKS.csv','analysis/gpt56terra98/GPT56TERRA_EXPECTATION_FINAL.csv')}
checks=[]
def check(name,value):checks.append({'check':name,'pass':bool(value)})
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9)
def read(rel):return pd.read_csv(ROOT/rel)
def main():
    manifest=json.loads((ROOT/'PROVENANCE_MANIFEST.json').read_text(encoding='utf-8'))
    for e in manifest['files']:
        p=ROOT/e['package_path'];b=p.read_bytes()
        check('hash:'+e['package_path'],len(b)==e['package_size_bytes'] and hashlib.sha256(b).hexdigest()==e['package_sha256'])
    summary={}
    finalcells=read('analysis_final/03_15CELL_BEHAVIORAL_EFFECTS.csv')
    for model,(run,stem,rankfile,labelsfile) in RUNS.items():
        base='experiment/runs/formal/'+run+'/'
        d=read(base+stem+'.csv');n=d.treatment.eq('F');rankcol='partner_exact_rank' if model=='DeepSeek' else 'partner_rank'
        r=pd.to_numeric(d[rankcol],errors='coerce')
        check(model+':3528 unique trial identifiers',len(d)==3528 and not d[KEYS].duplicated().any())
        check(model+':98 instances x36',d.participant_id.nunique()==98 and d.groupby('participant_id').size().eq(36).all())
        check(model+':7 groups x14',d.groupby('session_id').participant_id.nunique().to_dict()=={f'P{i}':14 for i in range(1,8)})
        check(model+':6 treatments x588',d.treatment.value_counts().to_dict()=={t:588 for t in 'ABCDEF'})
        check(model+':6 rounds per instance treatment',d.groupby(['participant_id','treatment'])['round'].apply(lambda x:set(x)==set(range(1,7))).all())
        check(model+':integer removal/contribution 0..10',d.removal.between(0,10).all() and d.contribution.between(0,10).all() and d.removal.mod(1).eq(0).all() and d.contribution.mod(1).eq(0).all())
        check(model+':contribution=10-removal',(d.contribution+d.removal).eq(10).all())
        check(model+':Visible rank 1..14',r[~n].between(1,14).all() and r[~n].mod(1).eq(0).all())
        check(model+':strategies and rationales complete',d.primary_strategy_code.isin(['S'+str(i) for i in range(1,7)]).all() and d.rationale_text.fillna('').str.strip().ne('').all())
        check(model+':six distinct within-group partners',d.groupby(['participant_id','treatment']).partner_id.nunique().eq(6).all() and d.participant_id.ne(d.partner_id).all() and d.partner_id.map(d.groupby('participant_id').session_id.first()).eq(d.session_id).all())
        ranks=read(base+rankfile)
        check(model+':rank permutations',len(ranks)==98 and all(ranks.groupby('session_id')['session_rank_'+t].apply(lambda x:sorted(x)==list(range(1,15))).all() for t in 'AO CNE'.replace(' ','')))
        for treatment,trait in TREAT.items():
            z=d.treatment.eq(treatment);lookup=ranks.set_index('participant_id')['session_rank_'+trait]
            check(model+':partner rank matches '+trait,d.loc[z,'partner_id'].map(lookup).astype(float).eq(r[z]).all())
        d['bin']=r.map(lambda x:'High' if x<=5 else 'Middle' if x<=9 else 'Low');d.loc[n,'bin']='NoInfo'
        baseline=d[n].groupby('participant_id').contribution.mean();visible=d[~n].groupby('participant_id').contribution.mean();delta=float((visible-baseline).mean())
        s1=d.primary_strategy_code.eq('S1');s1delta=float((s1[~n].groupby(d.loc[~n,'participant_id']).mean()-s1[n].groupby(d.loc[n,'participant_id']).mean()).mean())
        e=read(labelsfile)
        check(model+':3528 valid frozen expectation labels',len(e)==3528 and not e[KEYS].duplicated().any() and e.expectation_code.isin(CATS).all())
        merged=d.merge(e[KEYS+['expectation_code']],on=KEYS,validate='one_to_one');check(model+':coding trial coverage',len(merged)==3528)
        dist=read('analysis/gpt56terra98/04_GPT56TERRA_EXPECTATION_DISTRIBUTION.csv' if model=='GPT-5.6 Terra' else 'analysis_final/15_QWEN_EXPECTATION_DISTRIBUTION.csv')
        dist=dist[dist.model.eq(model)]
        for _,row in dist.iterrows():
            scope=row.scope
            z=merged if scope=='Overall' else merged[n] if scope=='NoInfo' else merged[~n] if str(row.trait) in ('ALL','Visible') else merged[merged.treatment.eq(next(k for k,v in TREAT.items() if v==row.trait)) & merged.bin.eq(row.rank_bin)]
            check(model+':expectation count '+str(row.cell)+'/'+row.expectation_code,int(z.expectation_code.eq(row.expectation_code).sum())==int(row['count']) and len(z)==int(row.n))
        if model=='GPT-5.6 Terra':
            saved=read('analysis/gpt56terra98/01_GPT56TERRA_OVERALL_EFFECT.csv').iloc[0];expected=saved.delta_visible_minus_noinfo;cells=read('analysis/gpt56terra98/02_GPT56TERRA_15CELL_EFFECTS.csv');estimatecol='estimate_delta_visible_minus_noinfo'
        else:
            saved=read('analysis_final/02_OVERALL_VISIBILITY_EFFECTS.csv');expected=saved[saved.model.eq(model)].iloc[0].mean_delta_visible_minus_noinfo;cells=finalcells[finalcells.model.eq(model)];estimatecol='mean_delta'
        check(model+':frozen overall point estimate reproduced',close(delta,expected))
        for _,row in cells.iterrows():
            z=d[d.treatment.eq(next(k for k,v in TREAT.items() if v==row.trait)) & d.bin.eq(row.rank_bin)]
            x=z.groupby('participant_id').contribution.mean()-baseline
            check(model+':cell counts '+row.cell,len(z)=={'High':210,'Middle':168,'Low':210}[row.rank_bin])
            check(model+':cell point estimate '+row.cell,close(x.dropna().mean(),row[estimatecol]) and x.notna().sum()==row.n_participants)
        # CSV/JSONL identity checks only, no assumptions about physical API logs.
        records=[json.loads(line) for line in (ROOT/(base+stem+'.jsonl')).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        j=pd.DataFrame(records)
        check(model+':authoritative JSONL agrees with CSV keys',len(j)==3528 and set(map(tuple,j[KEYS].astype(str).values))==set(map(tuple,d[KEYS].astype(str).values)))
        strategies=read('analysis/gpt56terra98/05_GPT56TERRA_STRATEGY_DISTRIBUTION.csv' if model=='GPT-5.6 Terra' else 'analysis_final/06_STRATEGY_DISTRIBUTION.csv');strategies=strategies[strategies.model.eq(model)]
        for _,row in strategies.iterrows():
            scope=row['scope'] if 'scope' in row.index else row.condition;z=d[n] if scope=='NoInfo' else d[~n];count=row['count'] if 'count' in row.index else row.n_trials
            check(model+':strategy count '+scope+'/'+row.strategy,int(z.primary_strategy_code.eq(row.strategy).sum())==int(count))
        expected_s1={'DeepSeek':0.6170068027210884,'Qwen':0.5772108843537416,'GPT-5.6 Terra':186/2940}[model]
        check(model+':manuscript S1 descriptive difference',close(s1delta,expected_s1))
        summary[model]={'rows':len(d),'instances':int(d.participant_id.nunique()),'noinfo_mean':float(d[n].contribution.mean()),'visible_mean':float(d[~n].contribution.mean()),'instance_delta':delta,'instance_S1_delta':s1delta,'expectation_counts':merged.expectation_code.value_counts().to_dict()}
    # Rebuild the fixed prompts from stored levels, without BFI or inference calls.
    import importlib.util
    sys.path.insert(0,str(ROOT/'experiment_personality_induction'))
    p=ROOT/'experiment_personality_induction/personality_shaping/v3_game_safe/persona_prompt_generator_v3_game_safe.py'
    spec=importlib.util.spec_from_file_location('release_persona_generator',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    personas=[json.loads(line) for line in (ROOT/'experiment/runs/formal/formal_study1_98_v1_1_20261002_172630/02_FORMAL_PERSONAS_FROZEN.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    check('98 fixed persona records',len(personas)==98 and len({p['participant_id'] for p in personas})==98)
    for persona in personas:
        rebuilt=module.build_prompt_from_levels(persona['prompt_levels'])
        check('persona exact reconstruction '+persona['participant_id'],rebuilt['prompt']==persona['persona_text'])
    h=read('analysis/gpt56terra98/03_GPT56TERRA_HUMAN_ALIGNMENT_RANK_HIGHNESS.csv');h=h[h.domain.eq('continuous_rank_highness')]
    check('Terra rank coordinate transformation',len(h)==5 and all(close(row.display_beta,-row.source_beta) and close(row.display_ci_low,-row.source_ci_high) and close(row.display_ci_high,-row.source_ci_low) for _,row in h.iterrows()))
    result={'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','checks_total':len(checks),'failed_checks':[c['check'] for c in checks if not c['pass']],'summaries':summary,'limits':['No API calls or study generation','No bootstrap, p-value, OLS or figure rerun','Expectation labels treated as frozen observations; coding validity not independently verified','Checks reproduce descriptive point estimates and label counts, not complete statistical inference']}
    (ROOT/'VALIDATION_REPORT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2));return 0 if result['status']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
