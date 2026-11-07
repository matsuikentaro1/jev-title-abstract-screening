"""Recalculate reported results from saved outputs. No network or API calls.
Python standard library only; optional --figures uses matplotlib.
"""
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from decimal import Decimal, ROUND_HALF_UP
import argparse,csv,gzip,json,math,statistics as st
ROOT=Path(__file__).resolve().parents[1]
def read_csv(p):
 op=gzip.open if str(p).endswith('.gz') else open
 with op(p,'rt',encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mmm(v):return {'mean':st.mean(v),'min':min(v),'max':max(v)}
def fmt(x):return str(Decimal(str(x)).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP))
def interval(x):return fmt(x['mean']) if fmt(x['min'])==fmt(x['max']) else '%s (%s to %s)'%(fmt(x['mean']),fmt(x['min']),fmt(x['max']))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=ROOT/'results');ap.add_argument('--figures',action='store_true');args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 records=read_csv(ROOT/'data/records.csv');ids={r['record_id'] for r in records};assert len(ids)==4527
 ds={r['record_id']:r['dataset'] for r in records};positive={r['record_id'] for r in records if int(r['reference_retained'])};assert len(positive)==113
 runs=json.loads((ROOT/'data/runs.json').read_text());assert len(runs)==59
 meta={r['run_id']:r for r in runs};groups=defaultdict(list)
 for r in runs:groups[r['condition']].append(r['run_id'])
 for g in groups:groups[g].sort(key=lambda k:meta[k]['rep'])
 assert len(groups)==8
 for g,rr in groups.items():assert len(rr)==(3 if g.startswith('gpt-') else 10)
 predictions=read_csv(ROOT/'data/predictions.csv.gz');requests=read_csv(ROOT/'data/requests.csv.gz');pred=defaultdict(dict);req=defaultdict(list);request_counts=Counter();request_ids=set()
 for r in requests:
  assert r['request_id'] not in request_ids;request_ids.add(r['request_id']);req[r['run_id']].append(r)
 for r in predictions:
  k=r['run_id'];i=r['record_id'];assert i in ids and i not in pred[k] and r['request_id'] in request_ids
  p=float(r['probability_retained']) if r['probability_retained'] else None
  assert p is None or 0<=p<=1
  assert r['decision'] in ['I','E'];pred[k][i]=p if p is not None else r['decision'];request_counts[r['request_id']]+=1
 for r in requests:assert request_counts[r['request_id']]==int(r['records_in_request'])
 for k in meta:assert set(pred[k])==ids and len(req[k])==meta[k]['requests']
 def kept(k,c=.5):return {i for i,p in pred[k].items() if (p!='E' if isinstance(p,str) else p>=c)}
 def metrics(rr,d,c=.5):
  members={i for i in ids if ds[i]==d};pos=members&positive;neg=members-positive;values=[]
  for k in rr:
   retained=kept(k,c)&members;tp=len(retained&pos);fn=len(pos)-tp;fp=len(retained&neg);tn=len(neg)-fp
   values.append({'tp':tp,'fn':fn,'fp':fp,'tn':tn,'sensitivity':100*tp/len(pos),'specificity':100*tn/len(neg),'retained':len(retained),'workload_reduction_percent':100*(len(members)-len(retained))/len(members)})
  return {'cutoff':c,**{f:mmm([v[f] for v in values]) for f in values[0]},'runs_with_100pct_sensitivity':sum(v['fn']==0 for v in values),'per_run':values}
 def best(rr,d):return int(min(pred[k][i] for k in rr for i in positive if ds[i]==d)*100+1e-9)/100
 def varying(rr,c=.5):
  sets=[kept(k,c) for k in rr];return 100*len(set.union(*sets)-set.intersection(*sets))/len(ids)
 summary={};table=[];figure3=[]
 for g,rr in groups.items():
  single=[]
  for k in rr:
   r=req[k];cost=sum(int(x['input_tokens'])*float(x['input_usd_per_million'])+int(x['output_tokens'])*float(x['output_usd_per_million']) for x in r)/1e6
   single.append({'run_id':k,'cost_usd':cost,'wall_seconds':meta[k]['wall_seconds'],'requests':len(r),'records_per_request':len(ids)/len(r),'input_tokens':sum(int(x['input_tokens']) for x in r),'output_tokens':sum(int(x['output_tokens']) for x in r)})
  lat=sorted(float(r['latency_ms']) for k in rr for r in req[k]);q=lambda a:lat[int(a*(len(lat)-1))]
  o=summary[g]={'n_runs':len(rr),'per_run_resources':single,'wall_seconds':mmm([r['wall_seconds'] for r in single]),'cost_usd':mmm([r['cost_usd'] for r in single]),'records_per_request':mmm([r['records_per_request'] for r in single]),'latency_ms':{'median':q(.5),'p5':q(.05),'p95':q(.95)},'output_tokens_per_record':st.mean(r['output_tokens']/len(ids) for r in single),'decision_variation_percent':varying(rr),'datasets':{}}
  for d in ['takeshima','maruki']:
   m=metrics(rr,d);o['datasets'][d]={'at_50_or_llm':m}
   cuts=[(.5,'fixed 50% cutoff')] if not g.startswith('gpt-') else [(None,'categorical')]
   if not g.startswith('gpt-'):
    b=best(rr,d);o['datasets'][d]['highest_cutoff_retaining_all']=metrics(rr,d,b);o['datasets'][d]['grid']=[metrics(rr,d,n/100) for n in range(5,55,5)]
    o['datasets'][d]['transfer_from_other_dataset']=metrics(rr,d,best(rr,'maruki' if d=='takeshima' else 'takeshima'))
    cuts=[(.1,'grid'),(.2,'grid'),(.3,'grid'),(.5,'fixed 50% cutoff'),(b,'data-selected cutoff')]
   if g in ['noul-k1','choice-v4'] or g.startswith('gpt-'):
    for c,kind in cuts:
     v=metrics(rr,d,c if c is not None else .5)
     table.append({'table':'1' if g=='noul-k1' else '2','condition':g,'dataset':d,'cutoff':c if c is not None else '', 'cutoff_type':kind,'sensitivity':interval(v['sensitivity']),'specificity':interval(v['specificity']),'retained':interval(v['retained']),'excluded_reference_positive':interval(v['fn']),'full_sensitivity_runs':v['runs_with_100pct_sensitivity']})
   if g.startswith('noul'):
    figure3.append({'condition':g,'maximum_records_per_request':int(g.split('k')[-1]),'dataset':d,'sensitivity_mean':m['sensitivity']['mean'],'sensitivity_min':m['sensitivity']['min'],'sensitivity_max':m['sensitivity']['max'],'specificity_mean':m['specificity']['mean'],'specificity_min':m['specificity']['min'],'specificity_max':m['specificity']['max'],'wall_mean':o['wall_seconds']['mean'],'wall_min':o['wall_seconds']['min'],'wall_max':o['wall_seconds']['max']})
  if not g.startswith('gpt-'):
   o['decision_variation_mean_over_three_runs']=st.mean(varying(list(c)) for c in combinations(rr,3))
   sd=[st.pstdev(pred[k][i] for k in rr) for i in ids];o['within_record_probability_sd']={'mean':st.mean(sd),'max':max(sd)}
 # Probability intervals: pooled boundaries derived from per-record means in the two analyses.
 bands=[]
 for d in ['takeshima','maruki']:
  members=sorted(i for i in ids if ds[i]==d);means={g:{i:st.mean(pred[k][i] for k in groups[g]) for i in members} for g in ['noul-k1','choice-v4']}
  edges=[0.0];start=0.0
  for n in range(1,11):
   end=n/10;counts=[sum(start<=p and (p<end if end<1 else p<=end) for p in means[g].values()) for g in means]
   if min(counts)>=20:edges.append(end);start=end
  if edges[-1]<1:
   if len(edges)>1:edges[-1]=1.0
   else:edges.append(1.0)
  for g in means:
   for lo,hi in zip(edges[:-1],edges[1:]):
    counts=[];rates=[];positives=[]
    for k in groups[g]:
     selected={i for i in members if lo<=pred[k][i] and (pred[k][i]<hi if hi<1 else pred[k][i]<=hi)};assert selected
     counts.append(len(selected));positives.append(len(selected&positive));rates.append(100*positives[-1]/counts[-1])
    bands.append({'condition':g,'dataset':d,'lower':lo,'upper':hi,'n_mean':st.mean(counts),'positive_mean':st.mean(positives),'retention_mean':st.mean(rates),'retention_min':min(rates),'retention_max':max(rates)})
 mini=groups['gpt-5-mini-default'];astra=groups['gpt-6-astra-default'];miss=lambda rr:[positive-kept(k) for k in rr];a=miss(mini);b=miss(astra);union=set.union(*(a+b))
 summary['overlap']={'mini_excluded_in_every_run':len(set.intersection(*a)),'astra_excluded_in_every_run':len(set.intersection(*b)),'excluded_in_every_run_both':len(set.intersection(*a)&set.intersection(*b)),'excluded_in_any_run_either':len(union),'of_these_choice_mean_probability_below_50':sum(st.mean(pred[k][i] for k in groups['choice-v4'])<.5 for i in union)}
 summary['validation']={'records':len(ids),'positive':len(positive),'runs':len(runs),'predictions':len(predictions),'requests':len(requests),'network_calls':0}
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');write_csv(out/'tables_1_2.csv',table);write_csv(out/'figure_2_data.csv',bands);write_csv(out/'figure_3_data.csv',figure3)
 if args.figures:plot_figures(bands,figure3,out)
 print(json.dumps(summary['validation']));print('Results:',out)

def plot_figures(bands,f3,out):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,2,figsize=(13,9))
 for ri,g in enumerate(['noul-k1','choice-v4']):
  for ci,d in enumerate(['takeshima','maruki']):
   ax=axes[ri,ci];rows=[r for r in bands if r['condition']==g and r['dataset']==d];x=list(range(len(rows)));y=[r['retention_mean'] for r in rows]
   ax.bar(x,y,color=['#0072B2','#D55E00'][ri]);ax.errorbar(x,y,yerr=[[v-r['retention_min'] for v,r in zip(y,rows)],[r['retention_max']-v for v,r in zip(y,rows)]],fmt='none',color='black',capsize=3)
   ax.set_xticks(x,['%d-%s%d%%\nn=%.0f'%(r['lower']*100,'<' if r['upper']<1 else '',r['upper']*100,r['n_mean']) for r in rows],rotation=40,ha='right');ax.set_ylim(0,110);ax.set_ylabel('Human retention (%)');ax.set_title('ABCD'[ri*2+ci]+'  '+('Light therapy' if ci==0 else 'Adjunctive pharmacotherapy')+' / '+('Primary' if ri==0 else 'Secondary'))
 fig.tight_layout();fig.savefig(out/'Figure_2_recalculated.pdf');fig.savefig(out/'Figure_2_recalculated.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(14,4))
 rows=sorted([r for r in f3 if r['dataset']=='takeshima'],key=lambda r:r['maximum_records_per_request']);x=[r['maximum_records_per_request'] for r in rows]
 axes[0].errorbar(x,[r['wall_mean'] for r in rows],yerr=[[r['wall_mean']-r['wall_min'] for r in rows],[r['wall_max']-r['wall_mean'] for r in rows]],marker='o',capsize=3);axes[0].set_title('A  Processing time');axes[0].set_ylabel('Seconds per run')
 for ax,d,title in [(axes[1],'takeshima','B  Light therapy'),(axes[2],'maruki','C  Adjunctive pharmacotherapy')]:
  rr=sorted([r for r in f3 if r['dataset']==d],key=lambda r:r['maximum_records_per_request'])
  for metric,col in [('sensitivity','#0072B2'),('specificity','#D55E00')]:ax.errorbar(x,[r[metric+'_mean'] for r in rr],yerr=[[r[metric+'_mean']-r[metric+'_min'] for r in rr],[r[metric+'_max']-r[metric+'_mean'] for r in rr]],marker='o',capsize=3,label=metric,color=col)
  ax.set_title(title);ax.set_ylabel('Percent');ax.legend()
 for ax in axes:ax.set_xticks(x);ax.set_xlabel('Maximum records per request')
 fig.tight_layout();fig.savefig(out/'Figure_3_recalculated.pdf');fig.savefig(out/'Figure_3_recalculated.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()
