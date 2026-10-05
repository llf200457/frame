"""Discovery-only post-review selection audit, with fold-local representation."""
from run_revision import *
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

def represent(a,others):
    # Arrays sample x gene; select genes and estimate scaling inside each fitting fold.
    idx=np.argsort(a.var(axis=0,ddof=1))[-5000:]; a=a[:,idx]
    mu=a.mean(axis=0); sd=a.std(axis=0,ddof=1); sd=np.where(sd>0,sd,1)
    z=(a-mu)/sd; pca=PCA(n_components=8,svd_solver='randomized',random_state=2025).fit(z)
    zz=[z]+[(b[:,idx]-mu)/sd for b in others]; out={}
    for k in [4,6,8]:
        v=pca.components_[:k]; assignment=np.argmax(abs(v),axis=0)
        w=np.zeros((len(idx),k))
        for j in range(k):
            mask=assignment==j; w[mask,j]=np.sign(v[j,mask])/mask.sum()
        out[k]=[q@w for q in zz]
    return out

def choose(a,y):
    rows=[]
    for fold,(fi,va) in enumerate(StratifiedKFold(3,shuffle=True,random_state=2025).split(a,y)):
        rep=represent(a[fi],[a[va]])
        for k in [4,6,8]:
            for c in [.1,1,10,30]:
                xt,xv=rep[k]; p=model(C=c).fit(xt,y[fi]).predict_proba(xv)[:,1]
                rows.append({'inner_fold':fold,'K':k,'C':c,'auc':roc_auc_score(y[va],p)})
    tab=pd.DataFrame(rows); rank=tab.groupby(['K','C']).auc.mean().reset_index().sort_values(['auc','K','C'],ascending=[False,True,True]); best=rank.iloc[0]
    return int(best.K),float(best.C),tab

def run():
    t,e=load(); tl=t[t.label_formal.isin(['AD','Control'])]; ev=e[e.status.isin(['AD','CTL'])]
    g1=pd.read_pickle(ROOT/'processed/GSE63060_gene_expression.pkl',compression='gzip')
    g2=pd.read_pickle(ROOT/'processed/GSE63061_gene_expression.pkl',compression='gzip')
    genes=sorted(set(g1.index)&set(g2.index)); a=g1.loc[genes,tl.index].T.to_numpy(); ae=g2.loc[genes,ev.index].T.to_numpy(); y=tl.label_formal.eq('AD').astype(int).to_numpy(); ye=ev.status.eq('AD').astype(int).to_numpy()
    rows=[]; searches=[]; oof=np.zeros(len(y)); foldid=np.zeros(len(y),int)
    for fold,(fi,va) in enumerate(StratifiedKFold(5,shuffle=True,random_state=2025).split(a,y),1):
        k,c,tab=choose(a[fi],y[fi]);tab['outer_fold']=fold;searches.append(tab)
        xt,xv=represent(a[fi],[a[va]])[k]; p=model(C=c).fit(xt,y[fi]).predict_proba(xv)[:,1];oof[va]=p;foldid[va]=fold
        rows.append({'fold':fold,'n_train':len(fi),'n_test':len(va),'K':k,'C':c,**metric(y[va],p)})
        print('outer',fold,'K',k,'C',c,metric(y[va],p),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'nested_cv_folds.csv',index=False); pd.concat(searches).to_csv(OUT/'nested_inner_search.csv',index=False)
    pd.DataFrame({'sample':tl.index,'y':y,'fold':foldid,'probability':oof}).to_csv(OUT/'nested_cv_oof.csv',index=False)
    k,c,tab=choose(a,y);tab.to_csv(OUT/'discovery_final_search.csv',index=False)
    xt,xv=represent(a,[ae])[k];p=model(C=c).fit(xt,y).predict_proba(xv)[:,1]
    pd.DataFrame({'sample':ev.index,'y':ye,'probability':p}).to_csv(OUT/'discovery_selected_external_predictions.csv',index=False)
    result={'analysis':'post-review discovery-only nested selection sensitivity; not retrospective preregistration','outer_folds':5,'inner_folds':3,'K_grid':[4,6,8],'C_grid':[.1,1,10,30],'l1_ratio':.5,'nested_oof':metric(y,oof),'final_K':k,'final_C':c,'external':metric(ye,p),'external_ci':ci(ye,p),'representation':'fold-local variance selection, standardization, PCA and signed-average grouping; no validation profiles used'}
    (OUT/'nested_cv_summary.json').write_text(json.dumps(result,indent=2)); print(result,flush=True)

if __name__=='__main__':run()
