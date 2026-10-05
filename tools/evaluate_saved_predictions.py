"""Optional replay of 18 original point metrics from locally held prediction files."""
import argparse,csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ranks(values):
    out=[0.]*len(values);order=sorted(range(len(values)),key=values.__getitem__);a=0
    while a<len(order):
        b=a+1
        while b<len(order) and values[order[b]]==values[order[a]]:b+=1
        for i in order[a:b]:out[i]=(a+1+b)/2
        a=b
    return out
def corr(x,y):
    a=sum(x)/len(x);b=sum(y)/len(y)
    return sum((v-a)*(w-b) for v,w in zip(x,y))/math.sqrt(sum((v-a)**2 for v in x)*sum((w-b)**2 for w in y))
def metrics(records,col):
    y=[float(r['observed_neutrophils_percent']) for r in records];p=[float(r[col]) for r in records]
    assert all(math.isfinite(v) for v in y+p);m=sum(y)/len(y)
    return dict(R2=1-sum((a-b)**2 for a,b in zip(y,p))/sum((a-m)**2 for a in y),
        MAE=sum(abs(a-b) for a,b in zip(y,p))/len(y),bias=sum(b-a for a,b in zip(y,p))/len(y),rho=corr(ranks(y),ranks(p)))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input-root',required=True,type=Path);a=ap.parse_args()
    d=a.input_root/'measurement_anchor';s=rows(d/'source_OOF.csv');t=rows(d/'heldout_factor_predictions.csv')
    assert len(s)==125 and len({r['sample'] for r in s})==125
    assert len(t)==74 and len({r['subject'] for r in t})==74 and {r['role'] for r in t}=={'heldout_subject'}
    ms=metrics(s,'age_sex_frozen8');mr=metrics(t,'unadapted__locked8');mo=metrics(t,'intercept__locked8');mm=metrics(t,'intercept__official_MCP8')
    numbers=json.loads((ROOT/'results/blood_anchor/manuscript_numbers.json').read_text())
    checks={'SourceRtwo':ms['R2'],'SourceMAE':ms['MAE'],'SourceRho':ms['rho'],
        'TargetRawRtwo':mr['R2'],'TargetRawMAE':mr['MAE'],'TargetRawBias':mr['bias'],
        'TargetRtwo':mo['R2'],'TargetMAE':mo['MAE'],'TargetRho':mo['rho'],'MCPGain':mm['MAE']-mo['MAE']}
    for k,v in checks.items():assert abs(v-numbers[k])<1e-10,(k,v,numbers[k])
    saved=next(r for r in rows(ROOT/'results/blood_anchor/official_performance.csv') if r['model']=='official_MCP8' and r['regime']=='intercept')
    for f,c in [('R2','R2'),('MAE','MAE'),('bias','bias'),('rho','Spearman_r')]:assert abs(mm[f]-float(saved[c]))<1e-10
    preds=rows(a.input_root/'stage26/external_transport_predictions.csv')
    for folder,file,col,cohort in [('stage27','GSE249477_target_relative_predictions.csv','prob_AD','GSE249477'),
            ('stage28','GSE248417_frozen_predictions.csv','prob_AD_uncalibrated','GSE248417')]:
        for r in rows(a.input_root/folder/file):preds.append({'cohort':cohort,'status':'CTL' if r['status']=='Control' else r['status'],'prob_AD':r[col]})
    aucs={}
    for r in rows(ROOT/'results/blood_anchor/clinical_transport.csv'):
        frame=[v for v in preds if v['cohort']==r['cohort'] and v['status'] in {'AD','CTL'}]
        assert len(frame)==int(r['n'])
        pos=[float(v['prob_AD']) for v in frame if v['status']=='AD'];neg=[float(v['prob_AD']) for v in frame if v['status']=='CTL']
        auc=sum(1 if p>n else .5 if p==n else 0 for p in pos for n in neg)/(len(pos)*len(neg))
        assert abs(auc-float(r['AUC']))<1e-10;aucs[r['cohort']]=auc
    print(json.dumps({'passed':True,'point_metrics_recomputed':18,'factor_metrics':{'source_OOF':ms,'external_raw':mr,'external_offset':mo,'official_offset':mm},
        'AUC':aucs,'bootstrap_intervals_recomputed':False,'training_executed':False},indent=2))
if __name__=='__main__':main()
