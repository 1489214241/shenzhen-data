
const fs=require('fs'),vm=require('vm');const data=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
class Point{constructor(lng,lat){this.lng=lng;this.lat=lat}equals(p){return this.lng===p.lng&&this.lat===p.lat}}
class Bounds{constructor(sw,ne){this.sw=sw;this.ne=ne}getSouthWest(){return this.sw}getNorthEast(){return this.ne}}
class Polygon{constructor(p){this.p=p}getPath(){return this.p}getBounds(){return new Bounds(new Point(Math.min(...this.p.map(p=>p.lng)),Math.min(...this.p.map(p=>p.lat))),new Point(Math.max(...this.p.map(p=>p.lng)),Math.max(...this.p.map(p=>p.lat))))}}
const fn=vm.runInNewContext('('+data.function+')',{BMap:{Point,Bounds,Polygon}},{timeout:1000});
const ctx={isPointInRect(p,b){return p.lng>=b.sw.lng&&p.lng<=b.ne.lng&&p.lat>=b.sw.lat&&p.lat<=b.ne.lat}};
const call=(pt,v)=>+fn.call(ctx,new Point(...pt),new Polygon(v.map(p=>new Point(...p))));
const sq=[[114,22],[114.01,22],[114.01,22.01],[114,22.01]];
const tests=[[[114.005,22.005],1],[[114,22.005],1],[[114,22],1],[[113.99,22],0]];
for(const [p,e] of tests)if(call(p,sq)!==e)throw Error('Boundary predicate failed');
process.stdout.write(JSON.stringify(data.points.map(p=>data.zones.map(z=>call(p,z)))));
