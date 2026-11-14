"""Check prospective SPA calibration formulas against their stated loss.

No downloaded code is executed and no dataset/model labels are read. This is a
scalar expected-risk check, not a reproduction of the paper's image experiments.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import brentq

OUT=Path(__file__).resolve().parents[1]/'outputs/shenzhen_resolution_20261112/method_review_20261112'


def risk(v,eta,k,b,zeta):
    return eta*np.logaddexp(0,-v)+(1-eta)*np.logaddexp(0,k*(v-b))/zeta


def derivative(v,eta,k,b,zeta):
    return -eta*expit(-v)+(1-eta)*k/zeta*expit(k*(v-b))


def risk_link(v,k,b,zeta):
    negative_slope=k/zeta*expit(k*(v-b))
    positive_magnitude=expit(-v)
    return negative_slope/(negative_slope+positive_magnitude)


def paper_printed_link(v,k,b,zeta):
    # Eq17 visually inspected on PDF page5; sigma is increasing per Eq3.
    a=k/zeta*expit(-k*(v-b))
    return a/(a+expit(v))


def released_code_link(v,k,b,zeta):
    # SPAInvLink.forward, retrieved author's code; positive parameters fixed1,1,0.
    e=np.exp(v);t=np.exp(-k*(v-b))
    return (k*e+1)/(zeta+zeta*t+k+k*e)


def main():
    dest=OUT/'spa_link_consistency.json'
    if dest.exists():raise FileExistsError('Preserve numeric audit')
    rows=[]
    for k,b,zeta in [(1.,0.,1.),(3.,.75,5.),(3.,1.,5.)]:
        for eta in [.01,.1,.5,.9,.99]:
            v=float(brentq(lambda z:derivative(z,eta,k,b,zeta),-50,50,xtol=1e-13))
            fd=(risk(v+1e-5,eta,k,b,zeta)-risk(v-1e-5,eta,k,b,zeta))/(2e-5)
            row=dict(k=k,b=b,zeta=zeta,true_eta=eta,risk_minimizer=v,
                derivative=float(derivative(v,eta,k,b,zeta)),finite_difference_derivative=float(fd),
                risk_derived_link=float(risk_link(v,k,b,zeta)),
                printed_equation17_link=float(paper_printed_link(v,k,b,zeta)),
                released_code_link=float(released_code_link(v,k,b,zeta)))
            assert abs(row['risk_derived_link']-eta)<1e-10 and abs(fd)<1e-8
            rows.append(row)
    output=dict(rows=rows,source_paper='https://openaccess.thecvf.com/content/CVPR2024/papers/Cheng_Towards_Calibrated_Multi-label_Deep_Neural_Networks_CVPR_2024_paper.pdf',
        source_code='https://raw.githubusercontent.com/JiachengCheng96/Towards-Calibrated-Multi-label-Deep-Neural-Networks/main/losses.py',
        reviewed='PDF page5 Eq16/Eq17 visually inspected; Eq3 definition and source code read.',
        loss='eta*softplus(-v)+(1-eta)*softplus(k*(v-b))/zeta',
        derived='[k/zeta*sigmoid(k*(v-b))]/([k/zeta*sigmoid(k*(v-b))]+sigmoid(-v))',
        interpretation='Literal printed inverse link and released inverse-link code do not match the conditional-risk minimizer of the stated loss for all tested parameters. This does not reject a corrected SPA construction or assess image experiments.',
        maximum_probability_error={key:max(abs(r[key]-r['true_eta']) for r in rows)
            for key in ['risk_derived_link','printed_equation17_link','released_code_link']},
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'spa_cvpr2024.pdf',OUT/'spa_losses.py.txt',Path(__file__)]},
        target_labels_read=False,implemented_in_shenzhen_model=False)
    dest.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(output['maximum_probability_error'])
    for r in rows:
        if r['true_eta']==.5:print(r)


if __name__=='__main__':main()
