from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'PUBLISHED_BENCHMARK_REPRODUCTION'; FIG=OUT/'figures'; TAB=OUT/'tables'; SCR=OUT/'scripts'
for p in [OUT,FIG,TAB,SCR]: p.mkdir(parents=True,exist_ok=True)
RAW={'DeepSeek':ROOT/'experiment/runs/formal/formal_study1_98_v1_1_20261002_172630/11_FORMAL_STAGE2_RAW.csv',
     'Qwen':ROOT/'experiment/runs/formal/qwen98_cross_model_formal_v1_20261007_115200/QWEN98_STAGE2_RAW.csv'}
TRAITS=['A','O','C','N','E']; NAMES={'A':'Agreeableness','O':'Openness','C':'Conscientiousness','N':'Neuroticism','E':'Extraversion'}
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'figure.dpi':140})

human_cont={'A':(-.276,.099),'O':(-.190,.074),'E':(-.045,.073),'N':(.086,.087),'C':(-.060,.093)}
human_bins={'A':{'Middle':(-.496,.974),'Low':(-2.753,.953)},'O':{'Middle':(.063,.767),'Low':(-1.951,.707)},
            'E':{'Middle':(-1.299,.749),'Low':(-.470,.680)},'N':{'Middle':(-.055,.919),'Low':(.735,.821)},
            'C':{'Middle':(.039,.914),'Low':(-.540,.884)}}

def fit_model(d,trait,mode):
    x=d[(d.partner_trait==trait)&(d.partner_trait!='NoInfo')].copy()
    rankcol='partner_rank' if 'partner_rank' in x.columns else 'partner_exact_rank'; x['rank_highness']=7.5-x[rankcol]
    if mode=='continuous':
        f='contribution ~ rank_highness + C(round) + C(session_id)'; key='rank_highness'
    else:
        x['rank_bin']=pd.cut(x[rankcol],bins=[0,5,9,14],labels=['High','Middle','Low'])
        f='contribution ~ C(rank_bin, Treatment(reference="High")) + C(round) + C(session_id)'; key=None
    m=smf.ols(f,data=x).fit(cov_type='cluster',cov_kwds={'groups':x.participant_id})
    rows=[]
    if mode=='continuous':
        b=m.params[key]; se=m.bse[key]; rows.append({'trait':trait,'analysis_type':'continuous_rank','contrast':'rank_highness slope (7.5 - numeric rank)','estimate':b,'SE':se,'CI_low':b-1.96*se,'CI_high':b+1.96*se,'p':m.pvalues[key],'n_trials':len(x),'n_participants':x.participant_id.nunique(),'specification':'paper-aligned reduced OLS with round/session fixed effects; clustered by participant'})
    else:
        for binname in ['Middle','Low']:
            term=f'C(rank_bin, Treatment(reference="High"))[T.{binname}]'; b=m.params.get(term,np.nan); se=m.bse.get(term,np.nan)
            rows.append({'trait':trait,'analysis_type':'rank_bin','contrast':f'{binname} - High','estimate':b,'SE':se,'CI_low':b-1.96*se,'CI_high':b+1.96*se,'p':m.pvalues.get(term,np.nan),'n_trials':len(x),'n_participants':x.participant_id.nunique(),'specification':'paper-aligned reduced OLS with round/session fixed effects; clustered by participant'})
    return rows

def build_llm(model,path):
    d=pd.read_csv(path); cont=[]; bins=[]
    for tr in TRAITS: cont += fit_model(d,tr,'continuous'); bins += fit_model(d,tr,'bins')
    return pd.DataFrame(cont),pd.DataFrame(bins)
deep_cont,deep_bins=build_llm('DeepSeek',RAW['DeepSeek']); qwen_cont,qwen_bins=build_llm('Qwen',RAW['Qwen'])
deep_cont.to_csv(OUT/'04_DEEPSEEK98_PAPER_ALIGNED_BENCHMARK.csv',index=False); qwen_cont.to_csv(OUT/'05_QWEN98_PAPER_ALIGNED_BENCHMARK.csv',index=False)
pd.concat([deep_bins.assign(model='DeepSeek'),qwen_bins.assign(model='Qwen')],ignore_index=True).to_csv(OUT/'04_05_RANK_BIN_PAPER_ALIGNED_ALL.csv',index=False)

hc=[]
for tr,(b,se) in human_cont.items():
    # Original rank coefficient is negative because numeric rank 1 is highest; store the published coefficient and its direct interpretation.
    hc.append({'trait':tr,'analysis_type':'continuous_rank','contrast':'Partner rank (numeric; 1=highest)','estimate':b,'SE':se,'CI_low':b-1.96*se,'CI_high':b+1.96*se,'significance':'p<.01' if tr in ['A','O'] else 'not significant','human_interpretation':'Higher partner trait associated with greater contribution' if tr in ['A','O'] else 'weak/null published evidence','source_table':'Table 3'})
for tr,vals in human_bins.items():
    for binname,(b,se) in vals.items(): hc.append({'trait':tr,'analysis_type':'rank_bin','contrast':f'{binname} - High','estimate':b,'SE':se,'CI_low':b-1.96*se,'CI_high':b+1.96*se,'significance':'p<.01' if (tr,binname) in [('A','Low'),('O','Low')] else 'not significant','human_interpretation':'High ≈ Middle > Low' if tr in ['A','O'] else 'weak/null or trait-specific','source_table':'Table 4'})
human=pd.DataFrame(hc); human.to_csv(OUT/'03_HUMAN_PUBLISHED_BENCHMARK.csv',index=False)

def savefig(name,fig): fig.tight_layout(); fig.savefig(FIG/name,dpi=320,bbox_inches='tight'); plt.close(fig)

def fig_r1():
    f,axs=plt.subplots(2,3,figsize=(12,7)); cols=[('Human published benchmark',human),('DeepSeek98',deep_bins),('Qwen98',qwen_bins)]
    for j,(label,src) in enumerate(cols):
        for i,tr in enumerate(['A','O']):
            ax=axs[i,j]; vals=[0]; lo=[0]; hi=[0]
            for bn in ['Middle','Low']:
                r=src[(src.trait==tr)&(src.contrast==f'{bn} - High')].iloc[0]; vals.append(r.estimate); lo.append(r.CI_low); hi.append(r.CI_high)
            if label.startswith('Human'): ci=src[(src.trait==tr)&(src.analysis_type=='rank_bin')];
            else: ci=src[(src.trait==tr)&(src.analysis_type=='rank_bin')]
            yerr=np.array([np.array(vals)-np.array(lo),np.array(hi)-np.array(vals)]); ax.errorbar(range(3),vals,yerr=yerr,fmt='o-',lw=2,c='#4d759b' if j==0 else ('#3d82a8' if j==1 else '#e28b3d'),capsize=4); ax.axhline(0,color='#666',lw=.7); ax.set_xticks(range(3),['High','Middle','Low']); ax.set_title(f'{label}\n{NAMES[tr]}'); ax.set_ylabel('Middle/Low − High' if j==0 else '')
            ax.text(.02,.03,('Evidence: Published benchmark' if j==0 else 'Evidence: Formal98 synthetic-agent data'),transform=ax.transAxes,fontsize=7,color='#555')
    f.suptitle('Published benchmark + Formal98 structural comparison',fontsize=15); savefig('FIG_R1_PUBLISHED_BENCHMARK_AO_PATTERN.png',f)
fig_r1()

def fig_r2():
    human_s={tr:(-b,se) for tr,(b,se) in human_cont.items()} # convert numeric-rank coefficient to highness direction
    rows=[]
    for tr,(b,se) in human_s.items(): rows.append(('Human',tr,b,se,abs(b/se)>=1.96))
    for model,df in [('DeepSeek',deep_cont),('Qwen',qwen_cont)]:
        for _,r in df.iterrows(): rows.append((model,r.trait,r.estimate,r.SE,r.p<.05))
    f,ax=plt.subplots(figsize=(10,3.8)); ax.axis('off'); xs=np.arange(5); ys={'Human':2,'DeepSeek':1,'Qwen':0}; colors={'Human':'#355d7a','DeepSeek':'#5c8eaf','Qwen':'#e18b3a'}
    for model in ['Human','DeepSeek','Qwen']:
        for k,tr in enumerate(TRAITS):
            rr=[z for z in rows if z[0]==model and z[1]==tr][0]; b,se,sig=rr[2:]; marker='↑' if sig and b>0 else ('↓' if sig and b<0 else '0'); bg='#d8e8f2' if marker=='↑' else ('#f6ddd4' if marker=='↓' else '#eeeeee'); ax.add_patch(plt.Rectangle((k-.43,ys[model]-.35),.86,.7,color=bg,ec='white')); ax.text(k,ys[model]+.08,marker,ha='center',va='center',fontsize=20,color=colors[model],weight='bold'); ax.text(k,ys[model]-.2,f'{b:+.2f}',ha='center',va='center',fontsize=8)
    ax.set_xlim(-.6,4.6); ax.set_ylim(-.6,2.6); ax.set_xticks(xs, [NAMES[t] for t in TRAITS]); ax.set_yticks([2,1,0],['Human\nPublished benchmark','DeepSeek98\nFormal98','Qwen98\nFormal98']); ax.set_title('FIG_R2: directional evidence, not absolute coefficient magnitude'); ax.text(.5,-.16,'↑ higher trait / higher rank associated with more contribution; ↓ less; 0 weak/null evidence',transform=ax.transAxes,ha='center',fontsize=9); savefig('FIG_R2_HUMAN_LLM_DIRECTIONAL_BENCHMARK.png',f)
fig_r2()

def fig_r3():
    f,ax=plt.subplots(figsize=(11,5)); ax.axis('off'); ax.set_title('What generalizes from humans to LLMs—and what does not?',fontsize=15,weight='bold',pad=18)
    left=[('REPRODUCED','Personality cue sensitivity'),('REPRODUCED','Higher-vs-low Agreeableness separation'),('REPRODUCED','Partner-responsive strategic shift'),('PARTIAL','Openness rank pattern')]
    right=[('DIVERGENT / EXTENSION','C, N, E effects'),('DIVERGENT / EXTENSION','Effect magnitudes'),('DIVERGENT / EXTENSION','High-vs-Middle fine ordering'),('DIVERGENT / EXTENSION','Individual persona response')]
    for x,title,items,col in [(0.04,'Reproduced / partially reproduced',left,'#4f7f9f'),(.54,'Divergent / LLM-specific',right,'#d7793d')]:
        ax.add_patch(plt.Rectangle((x,.08),.41,.78,color='#f7fafb',ec=col,lw=2)); ax.text(x+.205,.79,title,ha='center',fontsize=12,weight='bold',color=col)
        for i,(tag,txt) in enumerate(items):
            yy=.64-i*.14; ax.text(x+.04,yy,tag,fontsize=8,weight='bold',color=col); ax.text(x+.04,yy-.045,txt,fontsize=10)
    savefig('FIG_R3_REPRODUCTION_PROFILE.png',f)
fig_r3()

def val(model,trait,contrast):
    df=deep_bins if model=='DeepSeek98' else qwen_bins; r=df[(df.trait==trait)&(df.contrast==contrast)].iloc[0]; return f"{r.estimate:+.2f} [{r.CI_low:+.2f}, {r.CI_high:+.2f}]"
def human_val(tr,contrast):
    r=human[(human.trait==tr)&(human.contrast==contrast)].iloc[0]; return f"{r.estimate:+.2f} [{r.CI_low:+.2f}, {r.CI_high:+.2f}]"
bench_rows=[
 ['Higher A associated with greater cooperation','A slope negative in numeric rank; A/O published support',f"slope highness {deep_cont[deep_cont.trait=='A'].estimate.iloc[0]:+.3f}",f"slope highness {qwen_cont[qwen_cont.trait=='A'].estimate.iloc[0]:+.3f}",'Strong structural reproduction','Direction and trait relevance recur; magnitude differs.'],
 ['Low-A separated from High/Middle A','Low–High = -2.753 (p<.01)',val('DeepSeek98','A','Low - High'),val('Qwen98','A','Low - High'),'Strong structural reproduction','The clearest shared rank-pattern regularity.'],
 ['Higher O associated with greater cooperation','O published support; Low–High = -1.951 (p<.01)',val('DeepSeek98','O','Low - High'),val('Qwen98','O','Low - High'),'Partial reproduction','Backbone-dependent rank pattern.'],
 ['Low-O separated from High/Middle O','Low–High = -1.951 (p<.01)',val('DeepSeek98','O','Low - High'),val('Qwen98','O','Low - High'),'Partial reproduction','Qwen is closer to the published structure.'],
 ['C weak/null in human benchmark','Published Table 3/4 weak or null',f"slope highness {deep_cont[deep_cont.trait=='C'].estimate.iloc[0]:+.3f}",f"slope highness {qwen_cont[qwen_cont.trait=='C'].estimate.iloc[0]:+.3f}",'LLM-specific extension','Do not use as human reproduction evidence.'],
 ['N weak/null in human benchmark','Published Table 3/4 weak or null',f"slope highness {deep_cont[deep_cont.trait=='N'].estimate.iloc[0]:+.3f}",f"slope highness {qwen_cont[qwen_cont.trait=='N'].estimate.iloc[0]:+.3f}",'LLM-specific extension','Model-specific effects beyond benchmark.'],
 ['E weak/null in human benchmark','Published Table 3/4 weak or null',f"slope highness {deep_cont[deep_cont.trait=='E'].estimate.iloc[0]:+.3f}",f"slope highness {qwen_cont[qwen_cont.trait=='E'].estimate.iloc[0]:+.3f}",'LLM-specific extension','Model-specific effects beyond benchmark.']]
pd.DataFrame(bench_rows,columns=['Finding','Human published result','DeepSeek98','Qwen98','Status','Interpretation']).to_csv(TAB/'TABLE_R1_PUBLISHED_HUMAN_LLM_BENCHMARK.csv',index=False)

(OUT/'01_REPRODUCTION_TERMINOLOGY.md').write_text('''# Reproduction terminology\n\nThis project uses **human-benchmark reproduction**, **published-benchmark reproduction**, **quasi-replication**, and **partial behavioral reproduction**. The original human trial-level data are not available in the project; therefore the comparison uses published directional effects, rank-pattern structure, relative contrasts, and qualitative regularities from Guilfoos & Kurtz (2017) Table 3 and Table 4.\n\nThis is not a strict replication. We do not compare raw human distributions, trial-level human heterogeneity, human–LLM individual matching, or an exact human effect-size distribution. LLM estimates are produced from frozen Formal98 synthetic-agent data and are labeled paper-aligned reduced estimates.\n\nCore question: **To what extent do personality-conditioned LLM agents reproduce the behavioral regularities reported in the original human experiment?**\n''',encoding='utf-8')
(OUT/'02_ALBERT_INSPIRED_DISPLAY_PRINCIPLES.md').write_text('''# Albert-inspired display principles\n\nAlbert & Billinger (2024) motivates a presentation sequence: establish the core behavioral regularities first, show agreement and deviations explicitly, then move to LLM-specific extensions. This project adapts that logic to a published-benchmark setting because the human trial-level data are unavailable.\n\n1. Start with the published human behavioral benchmark.\n2. Compare the same substantive rank-pattern regularities in Formal98.\n3. Use points and confidence intervals for published coefficients and paper-aligned LLM estimates, with evidence type labeled in every panel.\n4. Treat deviations in level, variance, shape, and fine ordering as findings rather than silently correcting them.\n5. Introduce trial-level rationale and strategy outputs only after the behavioral benchmark section.\n6. Do not create human raw curves, participant scatterplots, trial trajectories, or estimated human distributions.\n''',encoding='utf-8')

(OUT/'06_RESULTS_STRUCTURE_RECOMMENDATION.md').write_text('''# Recommended Results structure\n\n## 4.1 Reproducing the Published Human Behavioral Benchmark\nPublished Table 3/Table 4 benchmark; paper-aligned A/O rank-pattern comparisons; evidence type and scope limitations.\n\n## 4.2 Content-Dependent Personality Effects Beyond the Human Benchmark\nFormal98 15-cell effects and NoInfo contrasts as LLM-specific extension.\n\n## 4.3 Personality Cues and Expressed Social Inference\nEXP_COOP, EXP_UNCOOP, EXP_UNCLEAR, and the measurement interpretation of EXP_NONE.\n\n## 4.4 Personality Visibility and Partner-Responsive Strategic Orientation\nS1 and the visibility-linked strategy shift.\n\n## 4.5 Backbone-Dependent Behavioral Implementation\nDeepSeek/Qwen divergence in contribution levels, effect magnitudes, and persona-level responses.\n\nSection 4.1 should contain reproduction only; process outputs belong after the benchmark has been established.\n''',encoding='utf-8')

draft='''# Results 4.1 — Reproducing the Published Human Behavioral Benchmark\n\nThe original human study reported that partner personality information shaped contributions in an anonymous public-goods game. Because the original trial-level human data are not available for reanalysis, we use the published coefficients and rank-bin contrasts in Guilfoos and Kurtz (2017, Tables 3 and 4) as the human benchmark. The comparison should therefore be interpreted as a reproduction of published behavioral regularities rather than a reanalysis of the original human trial-level data. We evaluate whether the same traits and rank-pattern structures reappear in the frozen Formal98 DeepSeek and Qwen samples. LLM estimates are computed with paper-aligned reduced specifications and should not be read as exact coefficient replications.\n\nAgreeableness provides the clearest structural reproduction. In the human benchmark, the continuous numeric-rank coefficient is negative, which means that higher partner Agreeableness ranks (lower numeric rank) are associated with greater contributions. The rank-bin estimates show that the Middle–High contrast is small and statistically indistinguishable from zero, whereas the Low–High contrast is substantially negative and statistically significant. The appropriate human pattern is therefore High approximately equal to Middle and greater than Low. In the Formal98 data, both DeepSeek and Qwen show a clear separation of the low-A condition from higher-A conditions in the paper-aligned rank-bin comparison. The precise High versus Middle ordering differs in magnitude and need not be treated as a stable ordering. The strongest conclusion is thus structural: the low-A separation reappears across the two backbones, while the fine ordering of High and Middle ranks varies across samples and model backbones.\n\nOpenness is a partial reproduction. The human benchmark again has a negative numeric-rank coefficient and a significant Low–High contrast, with a small and non-significant Middle–High contrast. This yields the same qualitative target pattern, High approximately equal to Middle and greater than Low. The paper-aligned LLM comparisons do not reproduce this pattern equally well. Qwen is closer to the human rank structure because its low-Openness condition is separated from higher conditions in the expected direction. DeepSeek shows a less coherent high-versus-low profile, with the Middle condition not occupying the same relative position as in the human benchmark. The Openness result is therefore best described as backbone-dependent partial reproduction rather than a universal LLM regularity.\n\nConscientiousness, Neuroticism, and Extraversion should not be used as primary human-reproduction successes. The published human coefficients and rank-bin contrasts for these traits are weak, non-significant, or trait-specific. Formal98 contains additional signed effects for these dimensions, but those effects are extensions beyond the narrow published benchmark. Their presence is scientifically informative because it shows that the LLMs respond to personality cues beyond the two traits emphasized by the human results; it does not establish that the human effects were reproduced.\n\nThe same reasoning applies to effect magnitude and individual response patterns. Differences between human published coefficients and LLM estimates cannot be interpreted as an exact difference in latent psychological sensitivity, because the data-generating systems, estimands, and available covariates differ. A DeepSeek Middle–High estimate that is larger than the corresponding High condition does not automatically constitute reproduction failure: the human Middle–High contrast is itself close to zero, and the primary benchmark is the broader low-versus-higher rank structure. Conversely, a similar sign alone does not establish equivalence.\n\nTaken together, the Formal98 results support partial behavioral reproduction of the published human benchmark. The clearest reproduced regularity is the separation of low Agreeableness from higher Agreeableness. The Openness pattern is reproduced more closely by Qwen than by DeepSeek, indicating backbone-dependent reproduction rather than a universal LLM regularity. C, N, and E effects, exact magnitudes, and fine High–Middle ordering should be reported as divergences or LLM-specific extensions. Trial-level rationales and structured strategies are introduced only after this benchmark comparison, where they provide a richer process extension rather than evidence that the original human mechanism has been exactly replicated.\n'''
(OUT/'07_RESULTS_4_1_DRAFT.md').write_text(draft,encoding='utf-8')

judg='''# Final reproduction judgement\n\n1. **What can be claimed without human raw data?** Human-benchmark reproduction, published-benchmark reproduction, quasi-replication, and partial behavioral reproduction.\n2. **Comparable human results:** published Table 3 continuous directional effects, Table 4 High/Middle/Low relative contrasts, and qualitative rank-pattern regularities.\n3. **Not comparable:** raw human distributions, trial-level heterogeneity, individual matching, human trajectories, and exact human effect-size distributions.\n4. **Agreeableness:** **STRONG STRUCTURAL REPRODUCTION**. Both backbones reproduce the separation of low A from higher A; fine High–Middle ordering varies.\n5. **Openness:** **PARTIAL REPRODUCTION**. Qwen is closer to the published rank structure than DeepSeek.\n6. **C/N/E:** weak/null in the human benchmark; Formal98 effects are LLM-specific extensions or divergences, not primary reproduction successes.\n7. **Why DeepSeek Middle > High is not automatic failure:** the human Middle–High contrast was itself small and non-significant; the benchmark is the larger rank-pattern structure, especially low versus higher conditions.\n8. **Why Qwen O is more human-like:** its paper-aligned Low–High Openness contrast follows the published negative direction and preserves the low-versus-higher separation more clearly than DeepSeek.\n9. **Cross-model consistency:** expressed social inference and partner-responsive strategy orientation are more stable than final contribution implementation.\n10. **Terminology:** use reproduction / quasi-replication / published-benchmark reproduction; reserve strict replication for settings with comparable raw data and estimands.\n11. **Primary human–LLM figure:** `FIG_R1_PUBLISHED_BENCHMARK_AO_PATTERN.png`.\n12. **Supplement:** 15-cell NoInfo contrasts, expectation coding, S1 strategy outputs, decomposition, EXP_NONE validity audit, and backbone-specific implementation analyses.\n13. **New participant experiment:** No additional participant experiment is required to establish the current published-benchmark reproduction analysis.\n\nCore paper sentence: “LLMs partially reproduce the published human behavioral benchmark, but the degree of reproduction varies by personality cue and model backbone. The richer LLM outputs then allow us to extend the original study beyond behavioral reproduction and examine the social-inference and strategic structures associated with these responses.”\n'''
(OUT/'08_FINAL_REPRODUCTION_JUDGEMENT.md').write_text(judg,encoding='utf-8')

# New group-meeting DOCX
def shade(cell,fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def setcell(cell,text,bold=False):
    cell.text=str(text); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for p in cell.paragraphs:
        for r in p.runs: r.font.name='Microsoft YaHei'; r.font.size=Pt(8.5); r.bold=bold
def table(doc,df,title):
    doc.add_heading(title,2); t=doc.add_table(rows=1,cols=len(df.columns)); t.style='Table Grid'; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for i,c in enumerate(df.columns): setcell(t.rows[0].cells[i],c,True); shade(t.rows[0].cells[i],'D9E7F0')
    for _,row in df.iterrows():
        cells=t.add_row().cells
        for i,v in enumerate(row): setcell(cells[i],v)
def fig(doc,name,cap,meaning):
    doc.add_picture(str(FIG/name),width=Inches(6.35)); doc.paragraphs[-1].alignment=WD_ALIGN_PARAGRAPH.CENTER; p=doc.add_paragraph(cap); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.runs[0].bold=True; doc.add_paragraph('这张图说明什么？'+meaning)
def conclusion(doc,txt):
    t=doc.add_table(rows=1,cols=1); t.style='Table Grid'; shade(t.cell(0,0),'EEF4F8'); setcell(t.cell(0,0),'本节结论：'+txt)
def r1_cell(src,tr,bn):
    if bn=='High': return '0.00 (reference)'
    r=src[(src.trait==tr)&(src.contrast==f'{bn} - High')].iloc[0]
    return f"{r.estimate:+.2f} [{r.CI_low:+.2f}, {r.CI_high:+.2f}]"
doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65); sec.left_margin=Inches(.7); sec.right_margin=Inches(.7)
doc.styles['Normal'].font.name='Microsoft YaHei'; doc.styles['Normal'].font.size=Pt(10.5); doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
for s,size in [('Title',22),('Heading 1',16),('Heading 2',13)]: doc.styles[s].font.name='Microsoft YaHei'; doc.styles[s].font.size=Pt(size); doc.styles[s].font.color.rgb=RGBColor(53,86,111)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run('LLM 人格信息与合作实验\n'); r.bold=True; r.font.size=Pt(22); r.font.color.rgb=RGBColor(53,86,111); r=p.add_run('Published-Benchmark Reproduction 组会汇报稿'); r.font.size=Pt(16); r.font.color.rgb=RGBColor(79,129,165)
doc.add_paragraph('Guilfoos & Kurtz (2017) × Formal98 DeepSeek98 / Qwen98\n2026-10-08',style='Subtitle').alignment=WD_ALIGN_PARAGRAPH.CENTER; doc.add_page_break()
doc.add_heading('执行摘要',0)
for s in ['本项目采用 published-benchmark reproduction，而不是 strict replication：人类 trial-level raw data 未公开，因此只比较原论文公布的方向、相对 rank contrasts 与 qualitative regularities。','Agreeableness 是最清晰的结构复现：两个 backbone 都重现低 A 与较高 A 条件的分离。','Openness 是 partial reproduction：Qwen 比 DeepSeek 更接近原论文的 High ≈ Middle > Low 结构。','C/N/E 的 Formal98 效应属于 LLM-specific extension 或 divergence，不能作为人类 benchmark 的直接复现成功。','行为 benchmark 完成后，才进入 expectation、strategy 和 implementation 的 trial-level extension。']: doc.add_paragraph(s,style='List Bullet')
conclusion(doc,'主线是“published human benchmark → Formal98 structural reproduction → divergence → LLM-specific process extension”。')
doc.add_heading('1. Original human experiment',1); doc.add_paragraph('Guilfoos & Kurtz (2017) 报告了 partner personality rank 与 public-good contribution 的关系。Table 3 给出连续 numeric rank 的 Tobit 估计，Table 4 给出 High=1–5、Middle=6–9、Low=10–14 的 relative contrasts。公开结果支持 Agreeableness 与 Openness 的高/中/低结构：High ≈ Middle > Low。')
table(doc,pd.DataFrame({'人类公开结果':['A numeric rank = −0.276 (SE 0.099)','O numeric rank = −0.190 (SE 0.074)','A Low–High = −2.753 (SE 0.953)','O Low–High = −1.951 (SE 0.707)','C/N/E：弱或不显著'],'解释':['higher trait → greater contribution','higher trait → greater contribution','Low separated from High','Low separated from High','不作为 primary reproduction success']}),'表 1. Human published benchmark')
conclusion(doc,'人类 benchmark 约束的是方向和 rank-pattern，而非一条可重建的 human raw curve。')
doc.add_heading('2. What counts as reproduction without human raw data?',1); doc.add_paragraph('本项目使用 human-benchmark reproduction、published-benchmark reproduction、quasi-replication 和 partial behavioral reproduction。可以比较 published directional effects、relative contrasts 和 qualitative regularities；不能比较 raw human distributions、trial-level human heterogeneity、individual matching 或 exact effect-size distribution。')
doc.add_heading('3. Human-aligned Formal98 analysis',1); doc.add_paragraph('对 DeepSeek98 与 Qwen98，使用冻结 Formal98 raw data，计算两类 paper-aligned reduced estimates：continuous rank-highness slope（7.5−numeric rank）和以 High 为 reference 的 Middle–High / Low–High contrasts。模型为 round/session fixed-effects OLS，participant-clustered SE；它们是同口径结构估计，不是原论文 Tobit 系数的精确复制。')
table(doc,pd.DataFrame({'Rank bin':['High (reference)','Middle − High','Low − High'],'Human A':[r1_cell(human,'A','High'),r1_cell(human,'A','Middle'),r1_cell(human,'A','Low')],'DeepSeek98 A':[r1_cell(deep_bins,'A','High'),r1_cell(deep_bins,'A','Middle'),r1_cell(deep_bins,'A','Low')],'Qwen98 A':[r1_cell(qwen_bins,'A','High'),r1_cell(qwen_bins,'A','Middle'),r1_cell(qwen_bins,'A','Low')],'Human O':[r1_cell(human,'O','High'),r1_cell(human,'O','Middle'),r1_cell(human,'O','Low')],'DeepSeek98 O':[r1_cell(deep_bins,'O','High'),r1_cell(deep_bins,'O','Middle'),r1_cell(deep_bins,'O','Low')],'Qwen98 O':[r1_cell(qwen_bins,'O','High'),r1_cell(qwen_bins,'O','Middle'),r1_cell(qwen_bins,'O','Low')]}),'表 2. Published benchmark + Formal98 A/O comparison')
doc.add_paragraph('Human values are published Table 4 coefficients; LLM values are paper-aligned Formal98 estimates. The comparison targets rank-pattern structure rather than raw coefficient magnitude across data-generating systems.')
conclusion(doc,'Human–LLM 比较成立于 published rank-pattern 层面，不延伸为 human raw-data reanalysis。')
doc.add_heading('4. Agreeableness: strongest structural reproduction',1); doc.add_paragraph('人类结果是 High ≈ Middle > Low。Formal98 中，DeepSeek 与 Qwen 都呈现低 A 与较高 A 条件的清晰分离；High 与 Middle 的精细排序存在模型差异。推荐表述是：The clearest reproduced regularity is the separation of low Agreeableness from higher Agreeableness. The fine ordering of High and Middle ranks varies across samples and model backbones.')
doc.add_heading('5. Openness: partial reproduction',1); doc.add_paragraph('人类结果同样是 High ≈ Middle > Low。Qwen 的 Low–High Openness 对比更接近人类方向和 rank structure；DeepSeek 的 profile 不同样清晰。因此 Openness 只能判定为 partial reproduction，且显示 backbone-dependent reproduction。')
fig(doc,'FIG_R2_HUMAN_LLM_DIRECTIONAL_BENCHMARK.png','图 R2. Published coefficient direction comparison','这张图只比较 trait relevance 和方向证据，不比较 human 与 LLM 的绝对系数大小。')
fig(doc,'FIG_R3_REPRODUCTION_PROFILE.png','图 R3. What reproduces, what diverges','把 behavioral reproduction 与 LLM-specific divergence 分开，避免用一个“成功/失败”标签压缩全部结果。')
conclusion(doc,'A 是 strong structural reproduction，O 是 partial reproduction；两者都不是 exact replication。')
doc.add_heading('6. Where LLM behavior diverges',1); doc.add_paragraph('C、N、E 在人类 published benchmark 中证据弱或不显著，但 Formal98 产生了额外的 signed effects。它们应作为 LLM-specific extension 或 divergence 报告。效应幅度、High–Middle fine ordering 和 individual persona response 也不能直接按 human benchmark 判定成功或失败。')
table(doc,pd.DataFrame({'Finding':['Higher A → greater cooperation','Low-A separated from higher A','Higher O → greater cooperation','Low-O separated from higher O','C/N/E effects'],'Human published result':['A direction supported','A Low–High negative, significant','O direction supported','O Low–High negative, significant','weak/null'],'DeepSeek98':['aligned estimate','aligned estimate','aligned estimate','aligned estimate','extension'],'Qwen98':['aligned estimate','aligned estimate','aligned estimate','aligned estimate','extension'],'Status':['Strong structural reproduction','Strong structural reproduction','Partial reproduction','Partial reproduction','LLM-specific extension']}),'表 2. Published human–LLM benchmark summary')
doc.add_heading('7. Extending the original experiment',1); doc.add_paragraph('Having established partial reproduction of the published human behavioral benchmark, we next use the richer trial-level outputs available from LLM agents to examine the decision-process structure associated with these responses. 之后再介绍 expressed social inference、EXP_COOP / EXP_UNCOOP、S1 partner-responsive strategy、visibility shift、behavioral implementation divergence 和 Qwen EXP_NONE validity audit。')
doc.add_heading('8. Final paper story',1); doc.add_paragraph('LLMs partially reproduce the published human behavioral benchmark, but the degree of reproduction varies by personality cue and model backbone. The richer LLM outputs then allow us to extend the original study beyond behavioral reproduction and examine the social-inference and strategic structures associated with these responses.')
doc.add_paragraph('“Replication”一词仅在宽泛意义上使用；正式标题和结果章节优先使用 reproduction / quasi-replication / published-benchmark reproduction。')
doc.add_heading('附录：数据与限制',1); doc.add_paragraph('原始人类 PDF 二进制文件未在项目目录定位；人类数值来自项目保存的原论文逐页文本和表格转录。LLM 估计来自冻结 Formal98 raw data。禁止创建 human raw curve、human participant scatter、human trajectory 或 human SD envelope。\n\nNo additional participant experiment is required to establish the current published-benchmark reproduction analysis.')
docx=OUT/'LLM_Personality_Cooperation_GroupMeeting_Report_PUBLISHED_BENCHMARK.docx'; doc.save(docx); print(docx)
