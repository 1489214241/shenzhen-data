"""Extract independent administrative polygons from the already acquired OSM PBF."""
import json
import sys
from pathlib import Path
import shenzhen_full_inputs as F
sys.path.insert(0,str(F.REPO/'tmp/shenzhen_osmium_vendor'))
import osmium
from shapely.geometry import shape, box, mapping


def main():
    F.guard()
    path=F.REPO/'data/深圳市/规则对齐独立资料_20261111/admin_geometry.json'
    if path.exists():return
    source=F.REPO/'tmp/shenzhen_guangdong.osm.pbf'
    factory=osmium.geom.GeoJSONFactory()
    region=box(113.8,22.45,114.7,23.0)
    class Extract(osmium.SimpleHandler):
        def __init__(self):super().__init__();self.features=[];self.errors=[]
        def area(self,a):
            tags=dict(a.tags)
            if tags.get('boundary')!='administrative' or not tags.get('name'):return
            try:
                g=shape(json.loads(factory.create_multipolygon(a)))
                if not g.intersects(region):return
                self.features.append(dict(type='Feature',geometry=mapping(g),properties=dict(tags,osm_area_id=int(a.id),osm_original_id=int(a.orig_id()),from_way=bool(a.from_way()),valid=bool(g.is_valid))))
            except Exception as e:self.errors.append(dict(id=int(a.id),error=type(e).__name__))
    h=Extract();h.apply_file('tmp/shenzhen_guangdong.osm.pbf',locations=True,idx='flex_mem')
    F.write(path,dict(type='FeatureCollection',features=h.features,source='https://download.openstreetmap.fr/extracts/asia/china/guangdong.osm.pbf',source_sha256=F.sha(source),crs_name='EPSG:4326',errors=h.errors,catchment_polygons=False))
    print('ADMIN',len(h.features),[(f['properties']['name'],f['properties'].get('admin_level')) for f in h.features],flush=True)

if __name__=='__main__':main()
