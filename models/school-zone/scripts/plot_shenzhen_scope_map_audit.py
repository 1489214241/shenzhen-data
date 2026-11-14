"""Visualize raw native map paths and map-hit answers, diagnostic only."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import shenzhen_full_inputs as F


def main():
    out=F.REPO/'outputs/shenzhen_resolution_20261112/scope_map_consistency_audit'
    path=out/'贤义信义_原图路径与命中.png'
    if path.exists():raise FileExistsError('Keep audit image')
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
    zones=F.read(F.SZ.DATA/'龙岗区/评价数据/官方学区范围.json')
    sites={s['coordinate_group_id']:s for s in F.SZ._load('龙岗区','小区数据/坐标组.json')}
    gold=F.SZ._load('龙岗区','Gold/小区_学校Gold.json')
    fig,ax=plt.subplots(figsize=(10,8.5),layout='constrained')
    for name,color in [('贤义外国语学校','#1768AC'),('信义实验小学','#C65C00')]:
        z=next(z for z in zones if z['school_name']==name);xy=np.array(z['vertices']);xy=np.vstack([xy,xy[:1]])
        ax.plot(xy[:,0],xy[:,1],color=color,lw=1.4,label=name+'原始路径')
        points=np.array([sites[r['coordinate_group_id']]['representative_point_bd09'] for r in gold if z['school_id'] in r['point_school_ids']])
        ax.scatter(points[:,0],points[:,1],color=color,s=26,alpha=.85,label=f'{name}命中小区：{len(points)}')
    ax.set_title('官方文字为共享范围，现有地图命中却相互分开',fontsize=15,pad=14)
    ax.set_xlabel('经度（官方原生 BD-09）');ax.set_ylabel('纬度（官方原生 BD-09）')
    ax.set_aspect(1/np.cos(np.deg2rad(22.6)));ax.ticklabel_format(useOffset=False);ax.grid(alpha=.18)
    ax.legend(loc='upper left',fontsize=10)
    fig.text(.5,-.03,'共同命中：0个。保留原始自交路径，不修复几何、不改Gold。\n官方指南第3页确认共享；第9页另列优享范围。此图不能独自确认地图路径的政策含义。',
        ha='center',fontsize=10)
    fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    print(path,flush=True)


if __name__=='__main__':main()
